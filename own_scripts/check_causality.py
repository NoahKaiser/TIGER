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
    return_details=False,
    method="nan",
    diff_rtol=1e-4,
    diff_atol=1e-6,
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
        return_details=return_details,
        method=method,
        diff_rtol=diff_rtol,
        diff_atol=diff_atol,
    )


def check_causality(
    model,
    sr=16000,
    algo_lat=0.005,
    num_trials=100,
    min_len_s=2.0,
    max_len_s=8.0,
    seed=None,
    return_details=False,
    method="nan",
    diff_rtol=1e-4,
    diff_atol=1e-6,
):
    """
    :param model: callable mapping 1D np.ndarray -> 1D np.ndarray (same length)
    :param sr: sampling rate in Hz
    :param algo_lat: allowed algorithmic latency in seconds
    :param num_trials: number of randomized test examples
    :param min_len_s: minimum signal duration in seconds
    :param max_len_s: maximum signal duration in seconds
    :param seed: optional random seed for reproducibility
    :param return_details: if True, return a report dict instead of only pass/fail bool
    :param method: "nan" (original NaN-propagation test) or "perturb" (finite perturbation test)
    :param diff_rtol: relative tolerance for perturbation output comparison
    :param diff_atol: absolute tolerance for perturbation output comparison

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
    if method not in {"nan", "perturb"}:
        raise ValueError(f'Unsupported method="{method}". Use "nan" or "perturb".')
    if diff_rtol < 0 or diff_atol < 0:
        raise ValueError("diff_rtol and diff_atol must be >= 0.")

    rng = np.random.default_rng(seed)
    allowed_lat_samples = int(algo_lat * sr)

    # Estimate minimal required latency (in samples) across randomized trials.
    # For one trial with NaN boundary p and first NaN output at f:
    #   required_latency_samples = max(0, p - f + 1)
    # Taking the maximum over trials gives a conservative estimate.
    est_lat_samples = 0
    saw_any_output_nan = False
    saw_any_output_diff = False
    worst_case = None

    for r in range(num_trials):
        l = rng.uniform(low=min_len_s, high=max_len_s)
        l = int(l * sr)
        sig = rng.standard_normal(l)
        sig = sig / np.max(np.abs(sig)) * 0.9
        p = int(rng.integers(len(sig)))
        meta = {}

        if method == "nan":
            sig_test = sig.copy()
            sig_test[p:] = np.nan
            est_sig = model(sig_test)  # obtain separation results using your model
            assert est_sig.shape == sig.shape  # they should have same length

            nan_positions = np.flatnonzero(np.isnan(est_sig))
            first_hit = int(nan_positions[0]) if nan_positions.size > 0 else None
            if first_hit is not None:
                saw_any_output_nan = True
                required_lat_samples = max(0, p - first_hit + 1)
            else:
                required_lat_samples = 0
            meta["first_event_kind"] = "first_output_nan"
        else:
            sig_ref = sig.copy()
            sig_alt = sig.copy()
            alt_tail = rng.standard_normal(len(sig_alt) - p)
            if alt_tail.size > 0:
                alt_tail = alt_tail / (np.max(np.abs(alt_tail)) + 1e-12) * 0.9
                sig_alt[p:] = alt_tail.astype(np.float32)

            out_ref = model(sig_ref)
            out_alt = model(sig_alt)
            assert out_ref.shape == sig.shape
            assert out_alt.shape == sig.shape

            # Earliest sample that changes when only future input (>= p) changes.
            # For causal models, this should occur no earlier than p - latency + 1.
            changed = ~np.isclose(out_ref, out_alt, rtol=diff_rtol, atol=diff_atol)
            diff_positions = np.flatnonzero(changed)
            first_hit = int(diff_positions[0]) if diff_positions.size > 0 else None
            if first_hit is not None:
                saw_any_output_diff = True
                required_lat_samples = max(0, p - first_hit + 1)
            else:
                required_lat_samples = 0
            meta["first_event_kind"] = "first_output_diff"

        if required_lat_samples > est_lat_samples:
            est_lat_samples = required_lat_samples
            worst_case = {
                "trial": r,
                "boundary_p": int(p),
                meta["first_event_kind"]: first_hit,
                "required_latency_samples": int(required_lat_samples),
            }

    est_lat_seconds = est_lat_samples / float(sr)
    passes_requirement = est_lat_samples <= allowed_lat_samples

    print(
        "Estimated algorithmic latency: "
        f"{est_lat_samples} samples ({est_lat_seconds * 1000.0:.3f} ms)"
    )
    print(
        "Allowed algorithmic latency: "
        f"{allowed_lat_samples} samples ({algo_lat * 1000.0:.3f} ms)"
    )
    if method == "nan" and not saw_any_output_nan:
        print("Warning: No NaNs were observed in model outputs across all trials.")
    if method == "perturb" and not saw_any_output_diff:
        print("Warning: No output changes were observed across perturbation trials.")
    if passes_requirement:
        print("Your model satisfies the algorithmic latency requirement!")
    else:
        print("Your model does NOT satisfy the algorithmic latency requirement!")
        if worst_case is not None:
            print(
                "Worst case: trial={trial}, p={boundary_p}, "
                "required={required_latency_samples} samples".format(**worst_case)
            )
            if method == "nan":
                print(
                    f"  first_output_nan={worst_case.get('first_output_nan')}"
                )
            else:
                print(
                    f"  first_output_diff={worst_case.get('first_output_diff')}"
                )

    report = {
        "method": method,
        "passes_requirement": bool(passes_requirement),
        "estimated_latency_samples": int(est_lat_samples),
        "estimated_latency_seconds": float(est_lat_seconds),
        "allowed_latency_samples": int(allowed_lat_samples),
        "allowed_latency_seconds": float(algo_lat),
        "num_trials": int(num_trials),
        "sample_rate": int(sr),
        "saw_any_output_nan": bool(saw_any_output_nan),
        "saw_any_output_diff": bool(saw_any_output_diff),
        "diff_rtol": float(diff_rtol),
        "diff_atol": float(diff_atol),
        "worst_case": worst_case,
    }
    if return_details:
        return report
    return bool(passes_requirement)


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
    parser.add_argument(
        "--method",
        type=str,
        choices=["nan", "perturb"],
        default="nan",
        help='Latency estimation method: "nan" (legacy) or "perturb" (recommended for attention models).',
    )
    parser.add_argument(
        "--diff_rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance used in perturbation comparison (method=perturb).",
    )
    parser.add_argument(
        "--diff_atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used in perturbation comparison (method=perturb).",
    )
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
        f"source_idx={args.source_idx} | device={device} | method={args.method}"
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
        method=args.method,
        diff_rtol=args.diff_rtol,
        diff_atol=args.diff_atol,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
