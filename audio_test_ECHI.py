import os
import argparse
import warnings

import torch
import yaml
import torchaudio

warnings.filterwarnings("ignore")

import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
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


def run_split(
    split_name: str,
    dataset,
    model: torch.nn.Module,
    model_device: torch.device,
    save_root: str,
    sample_rate: int,
    metrics: MetricsTracker | None = None,
) -> None:
    progress = build_progress(f"{split_name.capitalize()} Inference")
    with progress:
        for idx in progress.track(range(len(dataset))):
            mix, sources, key = tensors_to_device(dataset[idx], device=model_device)
            est_sources = model(mix[None]).squeeze(0)

            if metrics is not None:
                if sources.numel() == 0:
                    raise RuntimeError(
                        f"Cannot compute metrics for split '{split_name}': no target sources present."
                    )
                metrics(mix=mix, clean=sources, estimate=est_sources, key=key)

            idx_save_dir = os.path.join(save_root, split_name, f"idx{idx}")
            save_estimates(est_sources, key, idx_save_dir, sample_rate)


def main(config):
    train_conf = config["train_conf"]
    data_conf = train_conf["datamodule"]["data_config"]
    mixture_mode = str(data_conf.get("mixture_mode", "manifest"))
    if mixture_mode not in {"manifest", "target_sum"}:
        raise ValueError(
            f"Unsupported mixture_mode='{mixture_mode}'. Expected one of ['manifest', 'target_sum']."
        )

    if train_conf["datamodule"]["data_name"] != "ECHIDataModule":
        raise ValueError(
            "audio_test_ECHI.py expects datamodule.data_name == 'ECHIDataModule'."
        )

    sample_rate = int(data_conf["sample_rate"])
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
    model_device = next(model.parameters()).device

    datamodule = getattr(look2hear.datas, train_conf["datamodule"]["data_name"])(
        **data_conf
    )
    datamodule.setup()
    valid_set = getattr(datamodule, "data_val", None)
    test_set = getattr(datamodule, "data_test", None)

    if valid_set is None:
        raise RuntimeError("ECHIDataModule did not provide a valid split (data_val is None).")

    exp_results_dir = os.path.join(train_conf["main_args"]["exp_dir"], "results")
    os.makedirs(exp_results_dir, exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(exp_results_dir, "metrics_valid.csv"))

    print(f"Using ECHI mixture_mode='{mixture_mode}'.")
    print("Running evaluation and metrics on VALID split only.")

    with torch.no_grad():
        run_split(
            split_name="valid",
            dataset=valid_set,
            model=model,
            model_device=model_device,
            save_root=config["save_dir"],
            sample_rate=sample_rate,
            metrics=metrics,
        )
        metrics.final()

        if mixture_mode == "manifest":
            if test_set is None:
                print(
                    "mixture_mode='manifest' but test split is unavailable; skipping TEST inference."
                )
            else:
                print("Running TEST inference (no metrics).")
                run_split(
                    split_name="test",
                    dataset=test_set,
                    model=model,
                    model_device=model_device,
                    save_root=config["save_dir"],
                    sample_rate=sample_rate,
                    metrics=None,
                )
        else:
            print(
                "mixture_mode='target_sum': skipping TEST inference because ECHI test split has no targets."
            )


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    arg_dic = dict(vars(args))

    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    main(arg_dic)
