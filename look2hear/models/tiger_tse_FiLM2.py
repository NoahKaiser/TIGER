import inspect
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel
from ..layers import activations, normalizations
"""
Adjustements for Target Speaker Extraction (TSE)-TIGER:
1) make TSE_TIGER accept speaker embedding (spk_emb)
2) deleted block with force-mask-to-one for mixture consistency -> not valid for TSE and ECHI dataset
3) added FiLM (Global per-channel affine transformation) as EARLY conditioning on subband features using a fixed speaker embedding e,
   broadcasted among all sub-band equally

UPDATED HERE:
4) Convert to per-iteration Update-FiLM INSIDE Recurrent:
   x_{i+1} = x_i + FiLM( y_i - u_i ), where y_i = freq_time_process(u_i)
   and u_i is the iteration input (x for i=0, concat_block(mixture + x) for i>0).
   
5) started with setting beta to zero to avoid shift -> only scale modulation: set beta=None in the sperator
"""

def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)

class ATTConvActNorm(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ATTConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias

        if self.padding is None:
            self.padding = 0 if self.stride > 1 else "same"

        if kernel_size > 0:
            conv = nn.Conv2d if is2d else nn.Conv1d

            self.conv = conv(
                in_channels=self.in_chan,
                out_channels=self.out_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
            )
            if self.xavier_init:
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            self.conv = nn.Identity()

        self.act = activations.get(self.act_type)()
        self.norm = normalizations.get(self.norm_type)(
            (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
        )

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(
            hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class InjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat
        return out

class InjectionMulti(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        out = local_feat * sig_act
        return out

class UConvBlock(nn.Module):
    """
    Successive downsampling/upsampling for multi-resolution analysis.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=5,
                    stride=2,
                    groups=in_channels,
                    d=1,
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(InjectionMultiSum(in_channels, in_channels))

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.globalatt = Mlp(in_channels, in_channels, drop=0.1)

        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(InjectionMultiSum(in_channels, in_channels, 5))

    def forward(self, x):
        residual = x.clone()

        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        global_f = torch.zeros(
            output[-1].shape, requires_grad=True, device=output1.device
        )
        for fea in output:
            global_f = global_f + F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            )
        global_f = self.globalatt(global_f)

        x_fused = []
        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)

        return self.res_conv(expanded) + residual

class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "prelu",
        norm_type: str = "LayerNormalization4D",
        dim: int = 3,
        *args,
        **kwargs,
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Keys.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )
            self.Values.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
            )

        self.attn_concat_proj = ATTConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
        )

    def forward(self, x: torch.Tensor):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        batch_size, _, time, freq = x.size()
        residual = x

        all_Q = [q(x) for q in self.Queries]
        all_K = [k(x) for k in self.Keys]
        all_V = [v(x) for v in self.Values]

        Q = torch.cat(all_Q, dim=0)
        K = torch.cat(all_K, dim=0)
        V = torch.cat(all_V, dim=0)

        Q = Q.transpose(1, 2).flatten(start_dim=2)
        K = K.transpose(1, 2).flatten(start_dim=2)
        V = V.transpose(1, 2)
        old_shape = V.shape
        V = V.flatten(start_dim=2)
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        V = torch.matmul(attn_mat, V)
        V = V.reshape(old_shape)
        V = V.transpose(1, 2)
        emb_dim = V.shape[1]

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])
        x = x.transpose(0, 1).contiguous()

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])
        x = self.attn_concat_proj(x)

        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()

        return x


class Recurrent(nn.Module):
    """
    UPDATED: per-iteration Update-FiLM.

    Expected gamma/beta shapes:
      gamma: [B, N, 1, 1]  (broadcast over nband and T)
      beta : [B, N, 1, 1]  (optional; can be None)
    """
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        nband=8,
        upsampling_depth=3,
        n_head=4,
        att_hid_chan=4,
        kernel_size: int = 8,
        stride: int = 1,
        _iter=4
    ):
        super().__init__()
        self.nband = nband

        self.freq_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan,
                                    act_type="prelu", norm_type="LayerNormalization4D", dim=4),
            normalizations.get("LayerNormalization4D")((out_channels, 1))
        ])

        self.frame_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan,
                                    act_type="prelu", norm_type="LayerNormalization4D", dim=4),
            normalizations.get("LayerNormalization4D")((out_channels, 1))
        ])

        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU()
        )

    def forward(self, x, gamma=None, beta=None):
        # input x: [B, nband, N, T]
        B, nband, N, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # state: [B, N, nband, T]
        mixture = x.clone()

        for i in range(self.iter):
            # iteration input u
            if i == 0:
                u = x
            else:
                u = self.concat_block(mixture + x)  # [B, N, nband, T]

            # block output
            y = self.freq_time_process(u, B, nband, N, T)  # [B, N, nband, T]

            # update = y - u
            delta = y - u

            # Update-FiLM: delta <- gamma * delta + beta
            if gamma is not None:
                delta = gamma * delta
            if beta is not None:
                delta = delta + beta

            # update STATE (not u)
            x = x + delta

        return x.permute(0, 2, 1, 3).contiguous()  # [B, nband, N, T]

    def freq_time_process(self, x, B, nband, N, T):
        # Process Frequency Path
        residual_1 = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, N, nband]
        freq_fea = self.freq_path[0](x.view(B * T, N, nband))  # [B*T, N, nband]
        freq_fea = freq_fea.view(B, T, N, nband).permute(0, 2, 1, 3).contiguous()  # [B, N, T, nband]
        freq_fea = self.freq_path[1](freq_fea)  # [B, N, T, nband]
        freq_fea = self.freq_path[2](freq_fea)  # [B, N, T, nband]
        freq_fea = freq_fea.permute(0, 1, 3, 2).contiguous()  # [B, N, nband, T]
        x = freq_fea + residual_1

        # Process Frame Path
        residual_2 = x.clone()
        x2 = x.permute(0, 2, 1, 3).contiguous()  # [B, nband, N, T]
        frame_fea = self.frame_path[0](x2.view(B * nband, N, T))  # [B*nband, N, T]
        frame_fea = frame_fea.view(B, nband, N, T).permute(0, 2, 1, 3).contiguous()  # [B, N, nband, T]
        frame_fea = self.frame_path[1](frame_fea)  # [B, N, nband, T]
        frame_fea = self.frame_path[2](frame_fea)  # [B, N, nband, T]
        x = frame_fea + residual_2

        return x


class TSE_TIGER_FiLM2(BaseModel):
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
        film_hidden=256,
        film_scale=0.1,
    ):
        super(TSE_TIGER_FiLM2, self).__init__(sample_rate=sample_rate)

        self.sample_rate = sample_rate
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = out_channels
        self.film_scale = film_scale
        self.num_output = num_sources
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (25 hop), 1k-2k (100 hop), 2k-4k (250 hop), 4k-8k (500 hop)
        bandwidth_25 = int(np.floor(25 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_250 = int(np.floor(250 / (sample_rate / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sample_rate / 2.) * self.enc_dim))
        self.band_width = [bandwidth_25] * 40
        self.band_width += [bandwidth_100] * 10
        self.band_width += [bandwidth_250] * 8
        self.band_width += [bandwidth_500] * 8
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.band_width[i] * 2, self.eps),
                    nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1),
                )
            )

        self.separator = Recurrent(
            self.feature_dim,
            in_channels,
            self.nband,
            upsampling_depth,
            att_n_head,
            att_hid_chan,
            att_kernel_size,
            att_stride,
            num_blocks,
        )

        self.film_mlp = nn.Sequential(
            nn.LayerNorm(spk_emb_dim),
            nn.Linear(spk_emb_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, 2 * self.feature_dim),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.PReLU(),
                    nn.Conv1d(
                        self.feature_dim,
                        self.band_width[i] * 4 * num_sources,
                        1,
                        groups=num_sources,
                    ),
                )
            )

    def pad_input(self, input, window, stride):
        batch_size, nsample = input.shape
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)
        return input, rest

    def forward(self, input, spk_emb=None):
        # input shape: (B, C, T)
        if input.ndim == 1:
            input = input.unsqueeze(0).unsqueeze(1)
        if input.ndim == 2:
            input = input.unsqueeze(1)

        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)

        spec = torch.stft(
            input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            return_complex=True,
        )

        # concat real and imag, split to subbands
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

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(
                self.BN[i](
                    subband_spec_RI[i].view(
                        batch_size * nch, self.band_width[i] * 2, -1
                    )
                )
            )
        subband_feature = torch.stack(subband_feature, 1)  # [B*nch, nband, N, T]

        # --- speaker embedding checks ---
        if spk_emb is None:
            raise ValueError("spk_emb must be provided for TSE conditioning.")
        if spk_emb.ndim == 1:
            spk_emb = spk_emb.unsqueeze(0)
        elif spk_emb.ndim != 2:
            raise ValueError(
                f"spk_emb must have shape [D], [B, D], or [B*nch, D], got {tuple(spk_emb.shape)}."
            )

        spk_emb = spk_emb.to(device=subband_feature.device, dtype=subband_feature.dtype)

        if spk_emb.shape[0] == batch_size:
            spk_emb = spk_emb.repeat_interleave(nch, dim=0)
        elif spk_emb.shape[0] != batch_size * nch:
            raise ValueError(
                f"spk_emb batch mismatch: got {spk_emb.shape[0]}, expected "
                f"{batch_size} [B,D] or {batch_size*nch} [B*nch,D]."
            )

        # --- compute FiLM params ONCE; reuse per iteration ---
        film_params = self.film_mlp(spk_emb)  # [B*nch, 2N]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [B*nch, N]

        gamma = 1.0 + self.film_scale * gamma
        beta = self.film_scale * beta

        # broadcast for Recurrent internal shape [B, N, nband, T]
        gamma_it = gamma[:, :, None, None]  # [B*nch, N, 1, 1]
        beta_it  = beta[:,  :, None, None]  # [B*nch, N, 1, 1]

        # IMPORTANT:
        # We REMOVE early FiLM on subband_feature to "convert" to per-iteration Update-FiLM.
        # (If you want to keep early FiLM as well, re-enable the next line, but that is double-conditioning.)
        # subband_feature = gamma[:, None, :, None] * subband_feature + beta[:, None, :, None]

        # separator with per-iteration Update-FiLM
        sep_output = self.separator(
            subband_feature.view(batch_size * nch, self.nband, self.feature_dim, -1),
            gamma=gamma_it,
            beta=None,
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

    def get_model_args(self):
        return {"n_sample_rate": 2}