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


def check_causality_torch_model(model, source_idx=0, sr=16000, algo_lat=0.005, device=None):
    """Convenience helper to run check_causality directly on a PyTorch model."""
    wrapped_model = make_torch_source_wrapper(model, source_idx=source_idx, device=device)
    return check_causality(wrapped_model, sr=sr, algo_lat=algo_lat)


def check_causality(model, sr=16000, algo_lat=0.005):
    """
    :param model: callable mapping 1D np.ndarray -> 1D np.ndarray (same length)
    :param sr: sampling rate in Hz
    :param algo_lat: allowed algorithmic latency in seconds

    The idea is that we set samples starting from a random position to NaN,
    and a model that peeks into future context propagates NaNs too early.

    Tool is from "STFT-Domain Neural Speech Enhancement with Very Low
    Algorithmic Latency", Wang, Zhong-Qiu and Wichern, Gordon and Watanabe,
    Shinji and {Le Roux}, Jonathan.
    """

    algo_lat = int(algo_lat * sr)
    sig_len_range = [2.0, 8.0]  # range of signal length in seconds

    R = 100
    for r in range(R):
        l = np.random.uniform(low=sig_len_range[0], high=sig_len_range[1])
        l = int(l * sr)
        sig = np.random.randn(l)
        sig = sig / np.max(np.abs(sig)) * 0.9
        p = np.random.randint(len(sig))
        sig[p:] = np.nan

        est_sig = model(sig)  # obtain separation results using your model
        assert est_sig.shape == sig.shape  # they should have same length

        if p - algo_lat + 1 >= 1 and np.sum(np.isnan(est_sig[: p - algo_lat + 1])) > 0:
            print(
                "For example %d, your model does NOT satisfy the algorithmic latency requirement!"
                % r
            )
            return

    print("Your model satisfies the algorithmic latency requirement!")
