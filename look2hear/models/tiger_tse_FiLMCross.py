import torch
import torch.nn as nn

from .tiger_tse_SelfCross import TSE_TIGER_SelfCross


class TSE_TIGER_FiLMCross(TSE_TIGER_SelfCross):
    """
    TSE model variant combining:
    - FiLM1-style early conditioning on separator input features
    - SelfCross speaker-token cross-attention in frequency/frame paths
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
        spk_emb_dim=192,
        spk_num_tokens=8,
        spk_token_dim=128,
        spk_tokenizer_mode="linear",
        spk_token_drop=0.0,
        film_hidden=256,
        film_scale=0.1,
        film_init_std=1e-3,
        film_gate_init_logit=-2.0,
    ):
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
            num_sources=num_sources,
            sample_rate=sample_rate,
            spk_emb_dim=spk_emb_dim,
            spk_num_tokens=spk_num_tokens,
            spk_token_dim=spk_token_dim,
            spk_tokenizer_mode=spk_tokenizer_mode,
            spk_token_drop=spk_token_drop,
        )

        self.film_scale = film_scale
        self.film_mlp = nn.Sequential(
            nn.LayerNorm(spk_emb_dim),
            nn.Linear(spk_emb_dim, film_hidden),
            nn.ReLU(),
            # Per-band, per-channel FiLM: gamma/beta for each (band, feature_channel).
            nn.Linear(film_hidden, 2 * self.nband * self.feature_dim),
        )
        last = self.film_mlp[-1]
        half = last.out_features // 2
        with torch.no_grad():
            # Tiny non-zero init on gamma branch so FiLM is not fully inactive at start.
            nn.init.normal_(last.weight[:half], mean=0.0, std=float(film_init_std))
            nn.init.zeros_(last.weight[half:])
            nn.init.zeros_(last.bias)

        # Learnable FiLM contribution gate (sigmoid(logit) in [0,1]).
        self.film_gate_logit = nn.Parameter(torch.tensor(float(film_gate_init_logit)))
        # Runtime warmup multiplier in [0,1], set by LightningModule during training.
        self.register_buffer("film_warmup", torch.tensor(1.0), persistent=False)

    def set_film_warmup(self, value: float):
        v = max(0.0, min(1.0, float(value)))
        self.film_warmup.fill_(v)

    def get_film_gate(self):
        return torch.sigmoid(self.film_gate_logit) * self.film_warmup

    @staticmethod
    def _prepare_spk_emb(spk_emb, batch_size, nch, device, dtype):
        if spk_emb is None:
            raise ValueError("spk_emb must be provided for FiLM+Cross conditioning.")

        if spk_emb.ndim == 1:
            spk_emb = spk_emb.unsqueeze(0)
        elif spk_emb.ndim != 2:
            raise ValueError(
                f"spk_emb must have shape [D], [B, D], or [B*nch, D], got {tuple(spk_emb.shape)}."
            )

        spk_emb = spk_emb.to(device=device, dtype=dtype)

        if spk_emb.shape[0] == batch_size:
            if nch > 1:
                spk_emb = spk_emb.repeat_interleave(nch, dim=0)
        elif spk_emb.shape[0] != batch_size * nch:
            raise ValueError(
                f"spk_emb batch mismatch: got {spk_emb.shape[0]}, expected "
                f"{batch_size} [B,D] or {batch_size*nch} [B*nch,D]."
            )

        return spk_emb

    def forward(self, input, spk_emb=None):
        if input.ndim == 1:
            input = input.unsqueeze(0).unsqueeze(1)
        if input.ndim == 2:
            input = input.unsqueeze(1)

        batch_size, nch, nsample = input.shape

        spk_emb = self._prepare_spk_emb(
            spk_emb=spk_emb,
            batch_size=batch_size,
            nch=nch,
            device=input.device,
            dtype=input.dtype,
        )

        # Speaker prompt tokens for cross-attention branch.
        spk_tokens = self.spk_tokenizer(spk_emb)
        if spk_tokens.shape[0] != batch_size * nch:
            raise ValueError(
                f"spk_tokens batch mismatch: got {spk_tokens.shape[0]}, expected {batch_size * nch}."
            )

        input = input.view(batch_size * nch, -1)

        spec = torch.stft(
            input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            return_complex=True,
        )

        spec_RI = torch.stack([spec.real, spec.imag], 1)  # [B*nch, 2, F, T]
        subband_spec_RI = []
        subband_spec = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec_RI.append(
                spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous()
            )
            subband_spec.append(
                spec[:, band_idx : band_idx + self.band_width[i]]
            )  # [B*nch, BW, T]
            band_idx += self.band_width[i]

        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec_RI[i].view(batch_size * nch, self.band_width[i] * 2, -1)
                )
            )
        subband_feature = torch.stack(subband_feature, 1)  # [B*nch, nband, N, T]

        # FiLM1-style early conditioning at separator input.
        spk_emb = spk_emb.to(device=subband_feature.device, dtype=subband_feature.dtype)
        film_params = self.film_mlp(spk_emb)  # [B*nch, 2*nband*N]
        film_params = film_params.view(batch_size * nch, 2, self.nband, self.feature_dim)
        gamma = 1.0 + self.film_scale * film_params[:, 0]  # [B*nch, nband, N]
        beta = self.film_scale * film_params[:, 1]         # [B*nch, nband, N]
        film_out = gamma[..., None] * subband_feature + beta[..., None]

        # Residual-gated FiLM blend: identity at gate=0, full FiLM at gate=1.
        film_gate = self.get_film_gate().to(dtype=subband_feature.dtype)
        subband_feature = subband_feature + film_gate * (film_out - subband_feature)

        sep_output = self.separator(
            subband_feature.view(batch_size * nch, self.nband, self.feature_dim, -1),
            spk_tokens,
        )
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(self.nband):
            this_output = self.mask[i](sep_output[:, i]).view(
                batch_size * nch, 2, 2, self.num_output, self.band_width[i], -1
            )
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])
            this_mask_real = this_mask[:, 0]
            this_mask_imag = this_mask[:, 1]

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
        return output
