import torch

from look2hear.models.tiger_tse import TSE_TIGER


def test_tiger_tse_residual_closure_identity():
    torch.manual_seed(0)
    model = TSE_TIGER(
        out_channels=32,
        in_channels=64,
        num_blocks=1,
        upsampling_depth=2,
        win=640,
        stride=160,
        num_sources=4,
        sample_rate=16000,
    )
    model.eval()

    y = torch.randn(1, 3200)
    with torch.no_grad():
        speech_hat, residual_hat = model(y, return_residual=True)

    recon = speech_hat.sum(dim=1) + residual_hat
    max_abs_err = (y - recon).abs().max().item()

    assert speech_hat.shape == (1, 4, 3200)
    assert residual_hat.shape == (1, 3200)
    assert max_abs_err < 1e-4
