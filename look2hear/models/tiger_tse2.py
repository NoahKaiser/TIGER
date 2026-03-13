import torch

from .tiger_tse import TSE_TIGER


class TSE_TIGER2(TSE_TIGER):
    """TSE_TIGER variant with optional explicit residual output stream.

    This baseline keeps the original TIGER-TSE backbone, but can predict
    K speech outputs + 1 residual output and enforce a partitioned mixture
    constraint across all outputs.
    """

    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        att_n_head=4,
        att_hid_chan=4,
        att_kernel_size=8,
        att_stride=1,
        win=2048,
        stride=512,
        num_sources=2,
        sample_rate=44100,
        predict_residual=True,
        enforce_partition=True,
    ):
        self.num_speech_sources = int(num_sources)
        self.predict_residual = bool(predict_residual)
        self.enforce_partition = bool(enforce_partition)
        total_outputs = self.num_speech_sources + (1 if self.predict_residual else 0)
        if total_outputs < 1:
            raise ValueError("total number of outputs must be >= 1")
        if out_channels % total_outputs != 0:
            raise ValueError(
                "out_channels must be divisible by total outputs "
                f"(got out_channels={out_channels}, total_outputs={total_outputs}; "
                "total_outputs = num_sources + int(predict_residual))"
            )

        super().__init__(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            att_n_head=att_n_head,
            att_hid_chan=att_hid_chan,
            att_kernel_size=att_kernel_size,
            att_stride=att_stride,
            win=win,
            stride=stride,
            num_sources=total_outputs,
            sample_rate=sample_rate,
        )

    def forward(
        self,
        input,
        spk_emb=None,
        return_residual=False,
        predict_residual=None,
        enforce_partition=None,
    ):
        del spk_emb  # kept for compatibility with older call sites
        if predict_residual is None:
            predict_residual = self.predict_residual
        if enforce_partition is None:
            enforce_partition = self.enforce_partition

        if input.ndim == 1:
            input = input.unsqueeze(0).unsqueeze(1)
        elif input.ndim == 2:
            input = input.unsqueeze(1)
        elif input.ndim != 3:
            raise ValueError(f"Expected input with 1/2/3 dims, got {input.shape}")

        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)
        mixture_wav = input

        spec = torch.stft(
            input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            return_complex=True,
        )

        spec_ri = torch.stack([spec.real, spec.imag], 1)  # [B*nch, 2, F, T]
        subband_spec_ri = []
        subband_spec = []
        band_idx = 0
        for bw in self.band_width:
            subband_spec_ri.append(spec_ri[:, :, band_idx : band_idx + bw].contiguous())
            subband_spec.append(spec[:, band_idx : band_idx + bw])
            band_idx += bw

        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec_ri[i].view(batch_size * nch, self.band_width[i] * 2, -1)
                )
            )
        subband_feature = torch.stack(subband_feature, 1)

        sep_output = self.separator(
            subband_feature.view(batch_size * nch, self.nband, self.feature_dim, -1)
        )
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(self.nband):
            this_output = self.mask[i](sep_output[:, i]).view(
                batch_size * nch,
                2,
                2,
                self.num_output,
                self.band_width[i],
                -1,
            )
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            this_mask_real = this_mask[:, 0]
            this_mask_imag = this_mask[:, 1]

            if enforce_partition:
                this_mask_real_sum = this_mask_real.sum(1, keepdim=True)
                this_mask_imag_sum = this_mask_imag.sum(1, keepdim=True)
                this_mask_real = this_mask_real - (this_mask_real_sum - 1.0) / self.num_output
                this_mask_imag = this_mask_imag - this_mask_imag_sum / self.num_output

            est_spec_real = (
                subband_spec[i].real.unsqueeze(1) * this_mask_real
                - subband_spec[i].imag.unsqueeze(1) * this_mask_imag
            )
            est_spec_imag = (
                subband_spec[i].real.unsqueeze(1) * this_mask_imag
                + subband_spec[i].imag.unsqueeze(1) * this_mask_real
            )
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        sep_subband_spec = torch.cat(sep_subband_spec, 2)

        output = torch.istft(
            sep_subband_spec.view(batch_size * nch * self.num_output, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            length=nsample,
        )
        output = output.view(batch_size * nch, self.num_output, -1)

        if predict_residual:
            speech_hat = output[:, : self.num_speech_sources, :]
            residual_hat = output[:, self.num_speech_sources, :]
        else:
            speech_hat = output
            residual_hat = mixture_wav - speech_hat.sum(dim=1)

        if return_residual:
            return speech_hat, residual_hat
        return speech_hat
