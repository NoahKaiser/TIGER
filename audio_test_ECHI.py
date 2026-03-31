import os
import argparse
import warnings
import copy
import csv
import random
from itertools import permutations

import torch
import yaml
import torchaudio

warnings.filterwarnings("ignore")

import look2hear.models
import look2hear.datas
from look2hear.losses import PITLossWrapper, pairwise_neg_se_sisdr
from look2hear.utils import (
    tensors_to_device,
    RichProgressBarTheme,
    MyMetricsTextColumn,
    BatchesProcessedColumn,
)

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/mixit_conf.yml",
    help='Path to experiment conf.yml saved under checkpoint/"exp_name".',
)
parser.add_argument(
    "--save_dir",
    required=True,
    help="Base directory where separated outputs will be written (valid/test -> idx*/s*/...).",
)
parser.add_argument(
    "--evaluate_modes",
    nargs="+",
    default=["manifest", "target_sum"],
    choices=["manifest", "target_sum"],
    help=(
        "ECHI mixture variants to evaluate. "
        "'manifest' = original mixtures with background/noise; "
        "'target_sum' = artificial mixtures from target sum (without background noise)."
    ),
)
parser.add_argument(
    "--save_audio_count",
    type=int,
    default=-1,
    help=(
        "How many separated examples to write per split and mixture mode. "
        "-1: save all (default), 0: save none, >0: save that many representative items."
    ),
)
parser.add_argument(
    "--save_audio_strategy",
    choices=["evenly_spaced", "random", "first"],
    default="evenly_spaced",
    help="Strategy to select which examples are written when save_audio_count > 0.",
)
parser.add_argument(
    "--save_audio_seed",
    type=int,
    default=1337,
    help="Random seed used when save_audio_strategy='random'.",
)


class ECHIVariantMetricsTracker:
    """Track ECHI evaluation metrics for one dataset variant.

    Metrics:
    - se_sisdr_all: SE-SI-SDR on complete set (includes silent targets).
    - se_sisdr_all_i: SE-SI-SDR improvement over mixture baseline.
    - sisdr_active: SI-SDR (zero-mean=False) only on active targets.
    - sisdr_active_i: SI-SDR (zero-mean=False) improvement over mixture baseline.
    - residual_loss: normalized residual MSE (only when model returns explicit residual).
    """

    def __init__(self, save_file: str, activity_tau: float = 1e-6, eps: float = 1e-8):
        self.activity_tau = float(activity_tau)
        self.eps = float(eps)

        self.se_sisdr_all_values = []
        self.se_sisdr_all_i_values = []
        self.sisdr_active_values = []
        self.sisdr_active_i_values = []
        self.residual_loss_values = []
        self.num_segments = 0
        self.num_segments_no_active = 0
        self.num_segments_with_model_residual = 0

        self.results_csv = open(save_file, "w", newline="")
        self.csv_fields = [
            "snt_id",
            "n_active",
            "se_sisdr_all",
            "se_sisdr_all_i",
            "sisdr_active",
            "sisdr_active_i",
            "residual_loss",
            "has_model_residual",
        ]
        self.writer = csv.DictWriter(self.results_csv, fieldnames=self.csv_fields)
        self.writer.writeheader()

        self.pit_se_sisdr = PITLossWrapper(
            pairwise_neg_se_sisdr, pit_from="pw_mtx", threshold_byloss=False
        )

    def _pairwise_sisdr_loss_rect(self, estimate: torch.Tensor, clean_active: torch.Tensor) -> torch.Tensor:
        """Return rectangular SI-SDR loss matrix [n_est, n_active]."""
        if estimate.ndim != 2 or clean_active.ndim != 2:
            raise ValueError(
                f"Expected estimate and clean_active to be 2D [n_src, T], got {estimate.shape} and {clean_active.shape}"
            )

        s_estimate = estimate.unsqueeze(1)  # [n_est, 1, T]
        s_target = clean_active.unsqueeze(0)  # [1, n_active, T]
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [n_est, n_active, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + self.eps  # [1, n_active, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [n_est, n_active, T]
        e_noise = s_estimate - pair_wise_proj

        ratio = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + self.eps)
        pair_wise_sisdr = 10.0 * torch.log10(ratio + self.eps)
        return -pair_wise_sisdr

    def _compute_se_sisdr_all_metric(self, estimate: torch.Tensor, clean: torch.Tensor) -> float:
        se_loss = self.pit_se_sisdr(estimate.unsqueeze(0), clean.unsqueeze(0))
        return -float(se_loss.item())

    def _compute_sisdr_active_metric(self, estimate: torch.Tensor, clean_active: torch.Tensor) -> float:
        sisdr_loss_mtx = self._pairwise_sisdr_loss_rect(estimate, clean_active)
        best_active_loss = self._best_rect_assignment_mean(sisdr_loss_mtx)
        return -float(best_active_loss.item())

    @staticmethod
    def _best_rect_assignment_mean(loss_mtx: torch.Tensor) -> torch.Tensor:
        """Find minimum mean assignment loss for rectangular matrix [n_est, n_tgt]."""
        if loss_mtx.ndim != 2:
            raise ValueError(f"Expected 2D loss matrix, got shape {loss_mtx.shape}")

        n_est, n_tgt = loss_mtx.shape
        if n_tgt < 1:
            raise ValueError("n_tgt must be >= 1 for assignment.")
        if n_est < n_tgt:
            raise ValueError(
                f"Number of estimates ({n_est}) must be >= number of active targets ({n_tgt})."
            )

        tgt_idx = torch.arange(n_tgt, device=loss_mtx.device)
        best_loss = None
        for perm in permutations(range(n_est), n_tgt):
            est_idx = torch.as_tensor(perm, device=loss_mtx.device, dtype=torch.long)
            cur = loss_mtx[est_idx, tgt_idx].mean()
            if best_loss is None or cur < best_loss:
                best_loss = cur
        return best_loss

    def _compute_residual_loss(
        self, residual_hat: torch.Tensor, mix: torch.Tensor, clean: torch.Tensor
    ) -> float:
        """Normalized residual loss, matching AudioLightningModuleECHI._compute_residual_loss."""
        residual_target = mix - clean.sum(dim=0)
        num = ((residual_hat - residual_target) ** 2).mean()
        den = (mix ** 2).mean() + self.eps
        return float((num / den).item())

    @staticmethod
    def _mean_or_nan(values: list[float]) -> float:
        if not values:
            return float("nan")
        return float(sum(values) / len(values))

    def __call__(
        self,
        mix: torch.Tensor,
        clean: torch.Tensor,
        estimate: torch.Tensor,
        key: str,
        residual_hat: torch.Tensor | None = None,
    ) -> None:
        self.num_segments += 1

        # 1) SE-SI-SDR on complete set (silent targets included)
        se_sisdr_all = self._compute_se_sisdr_all_metric(estimate=estimate, clean=clean)
        self.se_sisdr_all_values.append(se_sisdr_all)
        baseline_all = mix.unsqueeze(0).repeat(clean.shape[0], 1)
        se_sisdr_all_baseline = self._compute_se_sisdr_all_metric(
            estimate=baseline_all, clean=clean
        )
        se_sisdr_all_i = se_sisdr_all - se_sisdr_all_baseline
        self.se_sisdr_all_i_values.append(se_sisdr_all_i)

        # 2) SI-SDR only for active targets
        target_energy = (clean ** 2).mean(dim=1)
        active_mask = target_energy > self.activity_tau
        n_active = int(active_mask.sum().item())

        sisdr_active = ""
        sisdr_active_i = ""
        if n_active > 0:
            clean_active = clean[active_mask]
            baseline_active = mix.unsqueeze(0).repeat(estimate.shape[0], 1)

            sisdr_active_value = self._compute_sisdr_active_metric(
                estimate=estimate, clean_active=clean_active
            )
            sisdr_active_baseline = self._compute_sisdr_active_metric(
                estimate=baseline_active, clean_active=clean_active
            )
            sisdr_active = sisdr_active_value
            sisdr_active_i = sisdr_active_value - sisdr_active_baseline
            self.sisdr_active_values.append(sisdr_active_value)
            self.sisdr_active_i_values.append(sisdr_active_i)
        else:
            self.num_segments_no_active += 1

        residual_loss = ""
        has_model_residual = residual_hat is not None
        if has_model_residual:
            self.num_segments_with_model_residual += 1
            residual_loss_value = self._compute_residual_loss(
                residual_hat=residual_hat, mix=mix, clean=clean
            )
            residual_loss = residual_loss_value
            self.residual_loss_values.append(residual_loss_value)

        self.writer.writerow(
            {
                "snt_id": key,
                "n_active": n_active,
                "se_sisdr_all": se_sisdr_all,
                "se_sisdr_all_i": se_sisdr_all_i,
                "sisdr_active": sisdr_active,
                "sisdr_active_i": sisdr_active_i,
                "residual_loss": residual_loss,
                "has_model_residual": int(has_model_residual),
            }
        )

    def final(self) -> dict[str, float | int]:
        summary = {
            "num_segments": int(self.num_segments),
            "num_segments_with_active_targets": int(self.num_segments - self.num_segments_no_active),
            "num_segments_without_active_targets": int(self.num_segments_no_active),
            "num_segments_with_model_residual": int(self.num_segments_with_model_residual),
            "se_sisdr_all_mean": self._mean_or_nan(self.se_sisdr_all_values),
            "se_sisdr_all_i_mean": self._mean_or_nan(self.se_sisdr_all_i_values),
            "sisdr_active_mean": self._mean_or_nan(self.sisdr_active_values),
            "sisdr_active_i_mean": self._mean_or_nan(self.sisdr_active_i_values),
            "residual_loss_mean": self._mean_or_nan(self.residual_loss_values),
            "activity_tau": float(self.activity_tau),
        }
        self.results_csv.close()
        return summary


def build_progress(title: str) -> Progress:
    metrics_column = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    return Progress(
        TextColumn(f"[bold blue]{title}", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metrics_column,
    )


def save_estimates(est_sources: torch.Tensor, key: str, save_dir: str, sample_rate: int) -> None:
    filename = os.path.basename(str(key))
    for i in range(est_sources.shape[0]):
        speaker_dir = os.path.join(save_dir, f"s{i + 1}")
        os.makedirs(speaker_dir, exist_ok=True)
        torchaudio.save(
            os.path.join(speaker_dir, filename),
            est_sources[i].unsqueeze(0).cpu(),
            sample_rate,
        )


def select_audio_indices(num_items: int, count: int, strategy: str, seed: int) -> set[int]:
    if num_items <= 0:
        return set()
    if count < 0 or count >= num_items:
        return set(range(num_items))
    if count == 0:
        return set()

    if strategy == "first":
        return set(range(count))
    if strategy == "random":
        rng = random.Random(seed)
        return set(rng.sample(range(num_items), count))

    # evenly_spaced
    if count == 1:
        return {num_items // 2}
    idx = {int(round(i * (num_items - 1) / (count - 1))) for i in range(count)}
    if len(idx) < count:
        for i in range(num_items):
            if i not in idx:
                idx.add(i)
                if len(idx) == count:
                    break
    return idx


def run_split(
    split_name: str,
    dataset,
    model: torch.nn.Module,
    model_device: torch.device,
    save_root: str,
    sample_rate: int,
    metrics: ECHIVariantMetricsTracker | None = None,
    audio_indices: set[int] | None = None,
) -> dict[str, float | int] | None:
    progress = build_progress(f"{split_name.capitalize()} Inference")
    save_all_audio = audio_indices is None
    with progress:
        for idx in progress.track(range(len(dataset))):
            mix, sources, key = tensors_to_device(dataset[idx], device=model_device)
            residual_hat = None
            try:
                model_out = model(mix[None], return_residual=True)
            except TypeError:
                model_out = model(mix[None])
            if isinstance(model_out, tuple):
                est_sources, residual_hat = model_out
                est_sources = est_sources.squeeze(0)
                residual_hat = residual_hat.squeeze(0)
            else:
                est_sources = model_out.squeeze(0)

            if metrics is not None:
                if sources.numel() == 0:
                    raise RuntimeError(
                        f"Cannot compute metrics for split '{split_name}': no target sources present."
                    )
                metrics(
                    mix=mix,
                    clean=sources,
                    estimate=est_sources,
                    key=key,
                    residual_hat=residual_hat,
                )

            if save_all_audio or (audio_indices is not None and idx in audio_indices):
                idx_save_dir = os.path.join(save_root, split_name, f"idx{idx}")
                save_estimates(est_sources, key, idx_save_dir, sample_rate)

    if metrics is not None:
        return metrics.final()
    return None


def main(config):
    train_conf = config["train_conf"]
    base_data_conf = train_conf["datamodule"]["data_config"]
    if train_conf["datamodule"]["data_name"] != "ECHIDataModule":
        raise ValueError(
            "audio_test_ECHI.py expects datamodule.data_name == 'ECHIDataModule'."
        )

    sample_rate = int(base_data_conf["sample_rate"])
    activity_tau = float(train_conf.get("training", {}).get("pit_activity_tau", 1e-6))
    train_conf["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", train_conf["exp"]["exp_name"]
    )
    model_path = os.path.join(train_conf["main_args"]["exp_dir"], "best_model.pth")

    model = getattr(look2hear.models, train_conf["audionet"]["audionet_name"]).from_pretrain(
        model_path,
        sample_rate=sample_rate,
        **train_conf["audionet"]["audionet_config"],
    )
    if train_conf["training"]["gpus"]:
        model.to("cuda")
    model.eval()
    model_device = next(model.parameters()).device

    exp_results_dir = os.path.join(train_conf["main_args"]["exp_dir"], "results")
    os.makedirs(exp_results_dir, exist_ok=True)
    summary_rows = []

    with torch.no_grad():
        for mixture_mode in config["evaluate_modes"]:
            data_conf = copy.deepcopy(base_data_conf)
            data_conf["mixture_mode"] = mixture_mode
            datamodule = getattr(look2hear.datas, train_conf["datamodule"]["data_name"])(
                **data_conf
            )
            datamodule.setup()
            valid_set = getattr(datamodule, "data_val", None)
            test_set = getattr(datamodule, "data_test", None)

            if valid_set is None:
                raise RuntimeError(
                    f"ECHIDataModule did not provide a valid split (data_val is None) for mixture_mode='{mixture_mode}'."
                )

            metrics = ECHIVariantMetricsTracker(
                save_file=os.path.join(exp_results_dir, f"metrics_valid_{mixture_mode}.csv"),
                activity_tau=activity_tau,
            )

            print(f"Using ECHI mixture_mode='{mixture_mode}'.")
            print("Running VALID evaluation: se_sisdr_all + sisdr_active.")

            variant_save_root = os.path.join(config["save_dir"], mixture_mode)
            valid_audio_indices = select_audio_indices(
                num_items=len(valid_set),
                count=int(config["save_audio_count"]),
                strategy=str(config["save_audio_strategy"]),
                seed=int(config["save_audio_seed"]),
            )
            print(
                f"Saving {len(valid_audio_indices)}/{len(valid_set)} VALID examples "
                f"(strategy={config['save_audio_strategy']})."
            )
            summary = run_split(
                split_name="valid",
                dataset=valid_set,
                model=model,
                model_device=model_device,
                save_root=variant_save_root,
                sample_rate=sample_rate,
                metrics=metrics,
                audio_indices=valid_audio_indices,
            )
            if summary is None:
                raise RuntimeError(f"Expected summary for mixture_mode='{mixture_mode}'.")

            summary_with_variant = {"mixture_mode": mixture_mode, **summary}
            summary_rows.append(summary_with_variant)
            print(
                f"[{mixture_mode}] se_sisdr_all_mean={summary['se_sisdr_all_mean']:.4f}, "
                f"se_sisdr_all_i_mean={summary['se_sisdr_all_i_mean']:.4f}, "
                f"sisdr_active_mean={summary['sisdr_active_mean']:.4f}, "
                f"sisdr_active_i_mean={summary['sisdr_active_i_mean']:.4f}, "
                f"residual_loss_mean={summary['residual_loss_mean']:.6f}, "
                f"segments={summary['num_segments']}, "
                f"no_active={summary['num_segments_without_active_targets']}, "
                f"with_model_residual={summary['num_segments_with_model_residual']}"
            )

            if mixture_mode == "manifest":
                if test_set is None:
                    print(
                        "mixture_mode='manifest' but test split is unavailable; skipping TEST inference."
                    )
                else:
                    print("Running TEST inference (no metrics).")
                    test_audio_indices = select_audio_indices(
                        num_items=len(test_set),
                        count=int(config["save_audio_count"]),
                        strategy=str(config["save_audio_strategy"]),
                        seed=int(config["save_audio_seed"]) + 1,
                    )
                    print(
                        f"Saving {len(test_audio_indices)}/{len(test_set)} TEST examples "
                        f"(strategy={config['save_audio_strategy']})."
                    )
                    run_split(
                        split_name="test",
                        dataset=test_set,
                        model=model,
                        model_device=model_device,
                        save_root=variant_save_root,
                        sample_rate=sample_rate,
                        metrics=None,
                        audio_indices=test_audio_indices,
                    )
            else:
                print(
                    "mixture_mode='target_sum': skipping TEST inference because ECHI test split has no targets."
                )

        summary_file = os.path.join(exp_results_dir, "metrics_valid_summary.csv")
        with open(summary_file, "w", newline="") as f:
            fieldnames = [
                "mixture_mode",
                "num_segments",
                "num_segments_with_active_targets",
                "num_segments_without_active_targets",
                "num_segments_with_model_residual",
                "se_sisdr_all_mean",
                "se_sisdr_all_i_mean",
                "sisdr_active_mean",
                "sisdr_active_i_mean",
                "residual_loss_mean",
                "activity_tau",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"Saved summary metrics to: {summary_file}")

        if "manifest" in config["evaluate_modes"] and "target_sum" in config["evaluate_modes"]:
            print(
                "Completed both ECHI variants: "
                "manifest (with noise) and target_sum (without background noise)."
            )
        elif "manifest" in config["evaluate_modes"]:
            print("Completed ECHI manifest variant (with noise).")
        else:
            print("Completed ECHI target_sum variant (without background noise).")


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    arg_dic = dict(vars(args))

    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    main(arg_dic)
