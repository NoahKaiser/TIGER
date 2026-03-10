import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # allows usage with non-torch models
    torch = None


def make_torch_source_wrapper(model, source_idx=0, device=None):
    """
    Wrap a PyTorch separator model for check_causality().

    Expected model outputs:
    - [B, S, T]: uses batch 0, source source_idx
    - [B, T]: uses batch 0
    - [T]: uses directly
    - tuple/list containing one of the above in first position
    """
    if torch is None:
        raise ImportError("PyTorch is required for make_torch_source_wrapper.")

    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    def wrapped(sig):
        sig = np.asarray(sig, dtype=np.float32)
        x = torch.from_numpy(sig).to(device)

        with torch.no_grad():
            y = model(x)

        if isinstance(y, (tuple, list)):
            if not y:
                raise ValueError("Model returned an empty tuple/list.")
            y = y[0]

        if not isinstance(y, torch.Tensor):
            raise TypeError("Model output must be a torch.Tensor or tuple/list of tensors.")

        if y.ndim == 3:
            if source_idx < 0 or source_idx >= y.shape[1]:
                raise ValueError(
                    f"source_idx={source_idx} out of range for output shape {tuple(y.shape)}."
                )
            est_sig = y[0, source_idx]
        elif y.ndim == 2:
            est_sig = y[0]
        elif y.ndim == 1:
            est_sig = y
        else:
            raise ValueError(f"Unsupported model output shape: {tuple(y.shape)}.")

        return est_sig.detach().cpu().numpy()

    return wrapped


def check_causality_torch_model(
    model,
    source_idx=0,
    sr=16000,
    algo_lat=0.005,
    device=None,
    num_trials=100,
    min_len_s=2.0,
    max_len_s=8.0,
    seed=None,
):
    """Convenience helper to run check_causality directly on a PyTorch model."""
    wrapped_model = make_torch_source_wrapper(model, source_idx=source_idx, device=device)
    return check_causality(
        wrapped_model,
        sr=sr,
        algo_lat=algo_lat,
        num_trials=num_trials,
        min_len_s=min_len_s,
        max_len_s=max_len_s,
        seed=seed,
    )


def check_causality(
    model,
    sr=16000,
    algo_lat=0.005,
    num_trials=100,
    min_len_s=2.0,
    max_len_s=8.0,
    seed=None,
):
    """
    :param model: callable mapping 1D np.ndarray -> 1D np.ndarray (same length)
    :param sr: sampling rate in Hz
    :param algo_lat: allowed algorithmic latency in seconds
    :param num_trials: number of randomized test examples
    :param min_len_s: minimum signal duration in seconds
    :param max_len_s: maximum signal duration in seconds
    :param seed: optional random seed for reproducibility

    The idea is that we set samples starting from a random position to NaN,
    and a model that peeks into future context propagates NaNs too early.

    Tool is from "STFT-Domain Neural Speech Enhancement with Very Low
    Algorithmic Latency", Wang, Zhong-Qiu and Wichern, Gordon and Watanabe,
    Shinji and {Le Roux}, Jonathan.
    """

    if num_trials <= 0:
        raise ValueError("num_trials must be > 0.")
    if min_len_s <= 0 or max_len_s <= 0:
        raise ValueError("min_len_s and max_len_s must be > 0.")
    if min_len_s > max_len_s:
        raise ValueError("min_len_s must be <= max_len_s.")

    rng = np.random.default_rng(seed)
    algo_lat = int(algo_lat * sr)

    for r in range(num_trials):
        l = rng.uniform(low=min_len_s, high=max_len_s)
        l = int(l * sr)
        sig = rng.standard_normal(l)
        sig = sig / np.max(np.abs(sig)) * 0.9
        p = int(rng.integers(len(sig)))
        sig[p:] = np.nan

        est_sig = model(sig)  # obtain separation results using your model
        assert est_sig.shape == sig.shape  # they should have same length

        if p - algo_lat + 1 >= 1 and np.sum(np.isnan(est_sig[: p - algo_lat + 1])) > 0:
            print(
                "For example %d, your model does NOT satisfy the algorithmic latency requirement!"
                % r
            )
            return False

    print("Your model satisfies the algorithmic latency requirement!")
    return True


def _load_model_from_args(args):
    if torch is None:
        raise ImportError("PyTorch is required to run this script.")

    try:
        import look2hear.models
    except Exception as exc:
        raise ImportError(
            "Could not import look2hear.models. Run from repo root or fix PYTHONPATH."
        ) from exc

    model_name = args.model_name
    model_kwargs = {}
    sample_rate = args.sample_rate

    if args.conf_dir is not None:
        try:
            import yaml
        except Exception as exc:
            raise ImportError("PyYAML is required when --conf_dir is used.") from exc

        with open(args.conf_dir, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_name = cfg["audionet"]["audionet_name"]
        model_kwargs = dict(cfg["audionet"]["audionet_config"])
        cfg_sr = cfg.get("datamodule", {}).get("data_config", {}).get("sample_rate")
        if sample_rate is None:
            sample_rate = cfg_sr

    else:
        if model_name is None:
            raise ValueError("Either --conf_dir or --model_name must be provided.")
        try:
            model_kwargs = json.loads(args.model_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--model_kwargs must be valid JSON. Got: {args.model_kwargs}") from exc
        if not isinstance(model_kwargs, dict):
            raise ValueError("--model_kwargs must decode to a JSON object.")

    if sample_rate is None:
        sample_rate = 16000

    Model = getattr(look2hear.models, model_name)
    ctor_kwargs = {"sample_rate": sample_rate, **model_kwargs}

    if args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = Model.from_pretrain(str(model_path), **ctor_kwargs)
    else:
        model = Model(**ctor_kwargs)

    return model, sample_rate, model_name


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone causality checker for separation models. "
            "Can load model settings from a TIGER YAML config or from explicit model args."
        )
    )
    parser.add_argument(
        "--conf_dir",
        type=str,
        default=None,
        help="Path to YAML config (e.g., configs/causal_tiger4.yml).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model class name in look2hear.models (used when --conf_dir is not provided).",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default="{}",
        help='JSON object for model constructor kwargs, e.g. \'{"num_sources":2,"win":640}\'.',
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional checkpoint path (e.g., best_model.pth created by training).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="Override sample rate for model construction and checker SR.",
    )
    parser.add_argument("--source_idx", type=int, default=0, help="Index of source to test.")
    parser.add_argument(
        "--algo_lat",
        type=float,
        default=0.005,
        help="Allowed algorithmic latency in seconds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device for forward pass: "auto", "cpu", or e.g. "cuda:0".',
    )
    parser.add_argument("--num_trials", type=int, default=100, help="Number of randomized tests.")
    parser.add_argument("--min_len_s", type=float, default=2.0, help="Min random signal length.")
    parser.add_argument("--max_len_s", type=float, default=8.0, help="Max random signal length.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser.parse_args()


def main():
    args = _parse_args()
    model, sr, model_name = _load_model_from_args(args)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if torch is not None:
        model = model.to(device)

    print(
        f"Running causality check for {model_name} | sr={sr} | algo_lat={args.algo_lat}s | "
        f"source_idx={args.source_idx} | device={device}"
    )
    ok = check_causality_torch_model(
        model=model,
        source_idx=args.source_idx,
        sr=sr,
        algo_lat=args.algo_lat,
        device=device,
        num_trials=args.num_trials,
        min_len_s=args.min_len_s,
        max_len_s=args.max_len_s,
        seed=args.seed,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
