

#!/usr/bin/env python3
"""
Visualize speech-enhancement report CSVs as boxplots + jittered points (matplotlib only).

Usage examples:
  python visualize_reports_matplotlib.py \
    --metrics_csv "/path/to/metrics.csv" \
    --metrics pysepm_fwsegsnr sdr \
    --out "metrics.png"

  # If you only want one metric
  python visualize_reports_matplotlib.py --metrics_csv ./metrics.csv --metrics pysepm_fwsegsnr
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def sort_sessions(sessions: list[str]) -> list[str]:
    def key(s: str):
        m = re.search(r"dev_(\d+)", s)
        return (0, int(m.group(1))) if m else (1, s)
    return sorted(sessions, key=key)


def box_scatter(ax, values_by_group, labels, seed=0, delta=0.22, jitter=0.12):
    centers = np.arange(1, len(labels) + 1)

    box_pos = centers + delta        # box a bit to the right
    pts_pos = centers - delta        # points a bit to the left

    bp = ax.boxplot(
        values_by_group,
        positions=box_pos,
        widths=0.38,
        showfliers=False, #shows outliers as points(fliers)
        patch_artist=True,
        manage_ticks=False,          # we set ticks ourselves
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i in range(len(labels)):
        c = colors[i % len(colors)] if colors else None
        bp["boxes"][i].set_facecolor("none")
        if c is not None:
            bp["boxes"][i].set_edgecolor(c)
            bp["medians"][i].set_color(c)
            bp["whiskers"][2*i].set_color(c); bp["whiskers"][2*i+1].set_color(c)
            bp["caps"][2*i].set_color(c);     bp["caps"][2*i+1].set_color(c)

    rng = np.random.default_rng(seed)
    for i, vals in enumerate(values_by_group):
        if len(vals) == 0:
            continue
        x = pts_pos[i] + rng.uniform(-jitter, jitter, size=len(vals))
        c = colors[i % len(colors)] if colors else None
        ax.scatter(x, vals, s=10, alpha=0.45, c=c)

    # --- Legend (one entry per session) ---
    handles = []
    for i, lab in enumerate(labels):
        c = colors[i % len(colors)] if colors else "C0"
        handles.append(Patch(facecolor=c, edgecolor=c, label=lab))

    ax.legend(
        handles=handles,
        title="Session",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(0, linewidth=1.0)
    ax.set_xlim(0.5, len(labels) + 0.5)


def main():
    ap = argparse.ArgumentParser()
    # Input: a single metrics.csv produced by the test script (MetricsTracker)
    ap.add_argument("--metrics_csv", type=str, default=None, help="Path to a single metrics.csv to plot")
    ap.add_argument("--key_session_regex", type=str, default=r"(dev_\d+)",
                    help="Regex applied to the 'key' column to extract session when 'session' column is absent (default: '(dev_\\d+)')")
    ap.add_argument("--metrics", nargs="+", required=True, help="Column names to plot (stacked as rows)")
    ap.add_argument("--out", type=str, default=None, help="Output PNG path (default: derived from input)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.metrics_csv is None:
        raise ValueError("Provide --metrics_csv")

    # --- Load data ---
    # Single CSV mode (produced by audio_test_EchoSet.py / MetricsTracker)
    data = pd.read_csv(args.metrics_csv)
    # Ensure some metadata columns exist for downstream logic
    if "file" not in data.columns:
        data["file"] = str(args.metrics_csv)
    # Derive session if missing
    if "session" not in data.columns:
        if "key" in data.columns:
            def sess_from_key(k: str):
                m = re.search(args.key_session_regex, str(k))
                return m.group(1) if m else "all"
            data["session"] = data["key"].apply(sess_from_key)
        else:
            data["session"] = "all"

    # Optional filtering if such columns are present
    # NOTE: Filtering by split/device/signal_type removed per requirements

    default_out = Path(args.metrics_csv).with_name("metrics.png")

    # --- Prepare labels and groups ---

    if "session" not in data.columns:
        raise ValueError("Column 'session' is required or must be derivable from 'key'.")
    sessions = sort_sessions([s for s in data["session"].dropna().unique()])
    if not sessions:
        raise ValueError("No sessions found after filtering.")

    # Validate metrics
    missing = [m for m in args.metrics if m not in data.columns]
    if missing:
        raise KeyError(f"Metrics not found in CSV columns: {missing}")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        nrows=len(args.metrics),
        ncols=1,
        figsize=(max(10, 1.2 * len(sessions)), 3.2 * len(args.metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(args.metrics) == 1:
        axes = [axes]
    # Removed signal_type title; keep figure simple
    for ax, metric in zip(axes, args.metrics):
        values = [data.loc[data["session"] == s, metric].dropna().to_numpy() for s in sessions]
        box_scatter(ax, values, sessions, seed=args.seed)
        ax.set_ylabel(metric)

    axes[-1].set_xlabel("Session")

    out = Path(args.out) if args.out else default_out
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
