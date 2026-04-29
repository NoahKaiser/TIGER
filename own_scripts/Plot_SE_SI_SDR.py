#!/usr/bin/env python3
"""
Executable script: publication-style two-panel plot (SE-SI-SDR vs SI-SDR)
and save as PNG to a configurable directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ==============================
    # ---- USER CONFIGURATION ------
    # ==============================
    save_dir = "/no_backups/s1495/Plots"  # <-- change to e.g. "/no_backups/s1495/Plots"
    filename = "se_si_sdr_beautiful.png"
    dpi = 300
    eps = 1e-8
    linthresh = 1e-9
    # ==============================

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # journal style
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "figure.dpi": 120,
        "savefig.dpi": dpi,
    })

    # x-axis (explicitly limited to [0, 1e3])
    x_pos = np.logspace(-12, 3, 2000)
    x0 = np.concatenate([[0.0], x_pos])
    x0 = x0[x0 <= 1e3]  # ensure max is 1e3

    # target norms
    t_speech = 1.0
    t_silence = 0.0

    # ----- "SE-SI-SDR" (stabilized numerator like in your snippet) -----
    y_se_sisdr_speech = 20.0 * np.log10((t_speech + eps) / (x0 + eps))
    y_se_sisdr_silence = 20.0 * np.log10((t_silence + eps) / (x0 + eps))

    # ----- "SI-SDR" (your modified definition) -----
    y_sisdr_speech = 20.0 * np.log10((t_speech / (x0 + eps)) + eps)
    y_sisdr_silence = 20.0 * np.log10((t_silence / (x0 + eps)) + eps)

    # Plot (two panels)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 6.2), sharex=True, constrained_layout=True
    )

    lw = 2.0

    ax1.plot(x0, y_se_sisdr_speech, linewidth=lw, label="SE-SI-SDR")
    ax1.plot(x0, y_sisdr_speech, "--", linewidth=lw, label="SI-SDR")
    ax1.set_title(r"Target speaker ($\|s\|=1$)")
    ax1.set_ylabel("Score (dB)")
    ax1.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax1.grid(True, which="minor", linewidth=0.4, alpha=0.18)
    ax1.legend(frameon=False, loc="best")
    ax1.axhline(0.0, linewidth=1.0, alpha=0.25)

    ax2.plot(x0, y_se_sisdr_silence, linewidth=lw, label="SE-SI-SDR")
    ax2.plot(x0, y_sisdr_silence, "--", linewidth=lw, label="SI-SDR")
    ax2.set_title(r"Silent target ($\|s\|=0$)")
    ax2.set_ylabel("Score (dB)")
    ax2.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax2.grid(True, which="minor", linewidth=0.4, alpha=0.18)
    ax2.legend(frameon=False, loc="best")
    ax2.axhline(0.0, linewidth=1.0, alpha=0.25)

    ax2.set_xscale("symlog", linthresh=linthresh)
    ax2.set_xlim(0.0, 1e3)  # (1) force x-axis range 0..1e3

    # (2) set ~5 ticks for x-axis (works well with symlog)
    xticks = [0.0, 1e-9, 1e-3, 1.0, 1e3]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(["0", r"$10^{-9}$", r"$10^{-3}$", r"$10^{0}$", r"$10^{3}$"])

    ax2.set_xlabel(
        r"Residual $l_2$-norm $\left\|\frac{\hat{s}^{\top}s}{\|s\|^2+\varepsilon}s-\hat{s}\right\|$"
    )

    for ax in (ax1, ax2):
        ax.axvline(linthresh, linewidth=1.0, alpha=0.25)

    fig.suptitle(
        r"Comparison of SE-SI-SDR vs. SI-SDR ($\varepsilon=10^{-8}$)",
        y=1.02
    )

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()
