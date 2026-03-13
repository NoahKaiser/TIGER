import torch

from look2hear.losses.matrix import PairwiseNegSISDRSilenceAware
from look2hear.losses.pit_wrapper import perm_reduce_active_soft_mean


def test_active_soft_mean_keeps_signal_for_all_silent_targets():
    pwl_set = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # [B=1, P=2, K=2]
    target_energy = torch.zeros(1, 2)

    reduced = perm_reduce_active_soft_mean(
        pwl_set, target_energy=target_energy, tau=1e-6, gamma=0.2
    )
    expected = torch.tensor([[1.5, 3.5]])

    assert torch.allclose(reduced, expected)


def test_silence_aware_pairwise_penalizes_nonzero_estimate_on_silence():
    loss_fn = PairwiseNegSISDRSilenceAware(
        zero_mean=False,
        take_log=True,
        activity_tau=1e-6,
        activity_beta=8.0,
        silence_weight=0.1,
    )

    targets = torch.zeros(1, 1, 128)
    est_zero = torch.zeros(1, 1, 128)
    est_nonzero = torch.ones(1, 1, 128) * 0.5

    loss_zero = loss_fn(est_zero, targets)
    loss_nonzero = loss_fn(est_nonzero, targets)

    assert torch.isfinite(loss_zero).all()
    assert torch.isfinite(loss_nonzero).all()
    assert loss_nonzero.mean() > loss_zero.mean()
