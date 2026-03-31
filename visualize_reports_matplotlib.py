#!/usr/bin/env python3
"""
Visualize experiment report CSVs with matplotlib.

Supports two workflows:
1) Legacy single-CSV plotting (e.g., EchoSet metrics):
   python visualize_reports_matplotlib.py \
     --metrics_csv /path/to/metrics.csv \
     --metrics sdr sdr_i \
     --out metrics.png

2) ECHI results-directory processing + plotting:
   python visualize_reports_matplotlib.py \
     --results_dir Experiments/checkpoint/TIGER-TSE2-ECHI_without-Noise/results
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch


ECHI_DEFAULT_METRICS = [
    "se_sisdr_all",
    "se_sisdr_all_i",
    "sisdr_active",
    "sisdr_active_i",
    "residual_loss",
]

METRIC_DISPLAY_NAMES = {
    "se_sisdr": "SE-SI-SDR",
    "se_sisdr_i": "SE-SI-SDRi",
    "se_sisdr_all": "SE-SI-SDR (All)",
    "se_sisdr_all_i": "SE-SI-SDRi (All)",
    "sisdr": "SI-SDR",
    "sisdr_i": "SI-SDRi",
    "sisdr_active": "SI-SDR (Active)",
    "sisdr_active_i": "SI-SDRi (Active)",
    "sdr": "SDR",
    "sdr_i": "SDRi",
    "snr": "SNR",
    "snr_i": "SNRi",
    "residual_loss": "Normalized MSE",
}


MODE_DISPLAY_NAMES = {
    "target_sum": "noise-free",
    "manifest": "noisy",
}


def configure_plot_fonts() -> None:
    # STIX is a close matplotlib-native match to TeX-like (txfonts-style) serif text.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "STIX Two Text",
                "STIXGeneral",
                "Times New Roman",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
        }
    )


def _humanize_label(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def metric_display_name(metric: str) -> str:
    name = METRIC_DISPLAY_NAMES.get(metric, _humanize_label(metric))
    if "sdr" in str(metric).lower() and "[dB]" not in name:
        return f"{name} [dB]"
    return name


def session_display_name(session: str) -> str:
    value = str(session)
    m = re.fullmatch(r"dev_(\d+)", value, flags=re.IGNORECASE)
    if m:
        return f"Dev {int(m.group(1))}"
    if value.lower() == "all":
        return "All"
    return _humanize_label(value)


def mode_display_name(mode: str) -> str:
    return MODE_DISPLAY_NAMES.get(mode, _humanize_label(mode))


def save_png_and_pgf(fig, out_path: Path, dpi: int, pfg: bool) -> None:
    out_png = Path(out_path)
    if out_png.suffix.lower() != ".png":
        out_png = out_png.with_suffix(".png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_png}")

    if pfg:
        out_pgf = out_png.with_suffix(".pgf")
        fig.savefig(out_pgf, bbox_inches="tight")
        print(f"Saved: {out_pgf}")


def sort_sessions(sessions: list[str]) -> list[str]:
    def key(s: str):
        m = re.search(r"dev_(\d+)", s)
        return (0, int(m.group(1))) if m else (1, s)

    return sorted([str(s) for s in sessions], key=key)


def box_scatter(
    ax,
    values_by_group,
    labels,
    *,
    seed=0,
    delta=0.22,
    jitter=0.12,
    max_points_per_group=800,
    show_legend=False,
    legend_title="Group",
):
    centers = np.arange(1, len(labels) + 1)
    box_pos = centers + delta
    pts_pos = centers - delta

    bp = ax.boxplot(
        values_by_group,
        positions=box_pos,
        widths=0.38,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = [f"C{i}" for i in range(len(labels))]

    for i in range(len(labels)):
        c = colors[i % len(colors)]
        bp["boxes"][i].set_facecolor("none")
        bp["boxes"][i].set_edgecolor(c)
        bp["medians"][i].set_color(c)
        bp["whiskers"][2 * i].set_color(c)
        bp["whiskers"][2 * i + 1].set_color(c)
        bp["caps"][2 * i].set_color(c)
        bp["caps"][2 * i + 1].set_color(c)

    rng = np.random.default_rng(seed)
    for i, vals in enumerate(values_by_group):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            continue
        if max_points_per_group and len(vals) > max_points_per_group:
            idx = rng.choice(len(vals), size=max_points_per_group, replace=False)
            vals = vals[idx]
        x = pts_pos[i] + rng.uniform(-jitter, jitter, size=len(vals))
        c = colors[i % len(colors)]
        ax.scatter(x, vals, s=10, alpha=0.4, c=c)

    if show_legend:
        handles = []
        for i, lab in enumerate(labels):
            c = colors[i % len(colors)]
            handles.append(Patch(facecolor=c, edgecolor=c, label=lab))
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
            fontsize=8,
            title_fontsize=9,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, linewidth=1.0, color="0.5", alpha=0.5)
    ax.set_xlim(0.5, len(labels) + 0.5)


def summarize_metrics(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    records = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["rows"] = int(len(g))
        for metric in metrics:
            s = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[f"{metric}_count"] = int(s.size)
            if s.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_p10"] = np.nan
                row[f"{metric}_p90"] = np.nan
            else:
                row[f"{metric}_mean"] = float(s.mean())
                row[f"{metric}_std"] = float(s.std(ddof=1)) if s.size > 1 else 0.0
                row[f"{metric}_median"] = float(s.median())
                row[f"{metric}_p10"] = float(s.quantile(0.10))
                row[f"{metric}_p90"] = float(s.quantile(0.90))
        records.append(row)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return out.sort_values(group_cols).reset_index(drop=True)


def ensure_session_column(df: pd.DataFrame, source_col: str, pattern: str) -> pd.DataFrame:
    if "session" in df.columns:
        return df
    if source_col not in df.columns:
        df["session"] = "all"
        return df
    extracted = df[source_col].astype(str).str.extract(pattern, expand=False)
    df["session"] = extracted.fillna("all")
    return df


def load_single_metrics_csv(metrics_csv: str, key_session_regex: str) -> pd.DataFrame:
    data = pd.read_csv(metrics_csv)
    if "file" not in data.columns:
        data["file"] = str(metrics_csv)
    if "session" not in data.columns:
        if "key" in data.columns:
            data["session"] = (
                data["key"]
                .astype(str)
                .str.extract(key_session_regex, expand=False)
                .fillna("all")
            )
        elif "snt_id" in data.columns:
            data = ensure_session_column(data, source_col="snt_id", pattern=r"/(dev_\d+)\.")
        else:
            data["session"] = "all"
    return data


def plot_legacy_single_csv(
    data: pd.DataFrame,
    metrics: list[str],
    out: Path,
    dpi: int,
    pfg: bool,
    show_sample_counts: bool,
    seed: int,
    max_points_per_group: int,
) -> None:
    sessions = sort_sessions([s for s in data["session"].dropna().unique()])
    display_sessions = [session_display_name(s) for s in sessions]
    if not sessions:
        raise ValueError("No sessions found in input CSV.")

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(max(10, 1.2 * len(sessions)), 3.1 * len(metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        values = [pd.to_numeric(data.loc[data["session"] == s, metric], errors="coerce").dropna().to_numpy() for s in sessions]
        box_scatter(
            ax,
            values,
            display_sessions,
            seed=seed + i,
            max_points_per_group=max_points_per_group,
            show_legend=(i == 0),
            legend_title="Session",
        )
        ax.set_ylabel(metric_display_name(metric))
        if show_sample_counts:
            cnt = ", ".join(f"{s}: {len(v)}" for s, v in zip(display_sessions, values))
            ax.text(0.99, 0.98, cnt, ha="right", va="top", fontsize=8, transform=ax.transAxes)

    axes[-1].set_xlabel("Session")
    save_png_and_pgf(fig, out, dpi, pfg)


def discover_variant_files(results_dir: Path) -> list[tuple[str, Path]]:
    variant_files = []
    ignored_stems = {
        "metrics_valid_summary",
        "metrics_valid_combined",
        "metrics_valid_summary_original",
    }
    for p in sorted(results_dir.glob("metrics_valid_*.csv")):
        if p.stem in ignored_stems:
            continue
        mode = p.stem.replace("metrics_valid_", "", 1)
        if not mode:
            continue
        variant_files.append((mode, p))
    return variant_files


def load_echi_results_dir(results_dir: Path, session_regex: str) -> tuple[pd.DataFrame, list[str], pd.DataFrame | None]:
    variant_files = discover_variant_files(results_dir)
    if not variant_files:
        raise FileNotFoundError(
            f"No variant CSVs found in '{results_dir}'. Expected files like metrics_valid_manifest.csv."
        )

    frames = []
    for mode, path in variant_files:
        df = pd.read_csv(path)
        df["mixture_mode"] = mode
        df = ensure_session_column(df, source_col="snt_id", pattern=session_regex)
        for col in [
            "n_active",
            "se_sisdr_all",
            "se_sisdr_all_i",
            "sisdr_active",
            "sisdr_active_i",
            "residual_loss",
            "has_model_residual",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "n_active" in df.columns:
            df["n_active"] = df["n_active"].round().astype("Int64")
        frames.append(df)

    summary_path = results_dir / "metrics_valid_summary.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
    modes = [m for m, _ in variant_files]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, modes, summary_df


def plot_distributions_by_mode(
    df: pd.DataFrame,
    modes: list[str],
    metrics: list[str],
    out_png: Path,
    dpi: int,
    pfg: bool,
    show_sample_counts: bool,
    seed: int,
    max_points_per_group: int,
) -> None:
    display_modes = [mode_display_name(m) for m in modes]
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(max(8, 2.0 * len(modes)), 3.0 * len(metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        values = []
        for mode in modes:
            s = pd.to_numeric(
                df.loc[df["mixture_mode"] == mode, metric],
                errors="coerce",
            ).dropna()
            values.append(s.to_numpy())
        box_scatter(
            ax,
            values,
            display_modes,
            seed=seed + i,
            max_points_per_group=max_points_per_group,
            show_legend=(i == 0),
            legend_title="Mixture mode",
        )
        ax.set_ylabel(metric_display_name(metric))
        if show_sample_counts:
            cnt = ", ".join(f"{m}: {len(v)}" for m, v in zip(display_modes, values))
            ax.text(0.99, 0.98, cnt, ha="right", va="top", fontsize=8, transform=ax.transAxes)

    axes[-1].set_xlabel("Mixture mode")
    save_png_and_pgf(fig, out_png, dpi, pfg)


def plot_trends_by_n_active(
    df: pd.DataFrame,
    modes: list[str],
    metrics: list[str],
    out_png: Path,
    dpi: int,
    pfg: bool,
) -> None:
    if "n_active" not in df.columns:
        return

    n_vals = pd.to_numeric(df["n_active"], errors="coerce").dropna().astype(int).unique()
    n_active_values = sorted(n_vals.tolist())
    if not n_active_values:
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = [f"C{i}" for i in range(len(modes))]

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(max(8, 1.6 * len(n_active_values)), 2.8 * len(metrics)),
        sharex=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax_i, (ax, metric) in enumerate(zip(axes, metrics)):
        for i, mode in enumerate(modes):
            part = df.loc[df["mixture_mode"] == mode, ["n_active", metric]].copy()
            part["n_active"] = pd.to_numeric(part["n_active"], errors="coerce")
            part[metric] = pd.to_numeric(part[metric], errors="coerce")
            part = part.dropna(subset=["n_active"])

            stats = (
                part.groupby("n_active")[metric]
                .agg(
                    median="median",
                    q25=lambda s: s.quantile(0.25),
                    q75=lambda s: s.quantile(0.75),
                    count="count",
                )
                .reset_index()
            )
            stats = stats.loc[stats["count"] > 0]
            if stats.empty:
                continue

            x = stats["n_active"].astype(int).to_numpy()
            y = stats["median"].to_numpy(dtype=float)
            y_lo = stats["q25"].to_numpy(dtype=float)
            y_hi = stats["q75"].to_numpy(dtype=float)
            c = colors[i % len(colors)]

            ax.plot(x, y, marker="o", linewidth=1.8, color=c, label=mode_display_name(mode))
            ax.fill_between(x, y_lo, y_hi, color=c, alpha=0.18, linewidth=0.0)

        ax.set_ylabel(metric_display_name(metric))
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, linewidth=1.0, color="0.5", alpha=0.5)
        if ax_i == 0:
            ax.legend(loc="best", frameon=False, fontsize=9)

    axes[-1].set_xlabel("Number of Active Speakers")
    axes[-1].set_xticks(n_active_values)
    save_png_and_pgf(fig, out_png, dpi, pfg)


def plot_session_delta_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    out_png: Path,
    dpi: int,
    pfg: bool,
) -> bool:
    if "session" not in df.columns or "mixture_mode" not in df.columns:
        return False

    modes = sorted(df["mixture_mode"].dropna().astype(str).unique().tolist())
    if "manifest" in modes and "target_sum" in modes:
        base_mode = "manifest"
        compare_mode = "target_sum"
    elif len(modes) >= 2:
        base_mode = modes[0]
        compare_mode = modes[1]
    else:
        return False

    sessions = sort_sessions(df["session"].dropna().astype(str).unique().tolist())
    if not sessions:
        return False
    display_sessions = [session_display_name(s) for s in sessions]

    means = df.groupby(["session", "mixture_mode"], dropna=False)[metrics].mean()
    matrix = np.full((len(metrics), len(sessions)), np.nan, dtype=float)

    for i, metric in enumerate(metrics):
        for j, session in enumerate(sessions):
            try:
                v_base = float(means.loc[(session, base_mode), metric])
                v_comp = float(means.loc[(session, compare_mode), metric])
            except KeyError:
                continue
            matrix[i, j] = v_comp - v_base

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return False

    vmax = float(np.nanmax(np.abs(finite)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if vmax > 0 else None

    fig, ax = plt.subplots(
        figsize=(max(8, 0.8 * len(sessions)), max(3.5, 0.8 * len(metrics))),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, cmap="coolwarm", aspect="auto", norm=norm)
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(display_sessions, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([metric_display_name(m) for m in metrics])
    ax.set_title(
        "Session mean delta: "
        f"{mode_display_name(compare_mode)} - {mode_display_name(base_mode)}"
    )

    for i in range(len(metrics)):
        for j in range(len(sessions)):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Delta (dB / metric units)")
    save_png_and_pgf(fig, out_png, dpi, pfg)
    return True


def run_legacy_mode(args: argparse.Namespace) -> None:
    if args.metrics_csv is None:
        raise ValueError("Legacy mode requires --metrics_csv.")
    if not args.metrics:
        raise ValueError("Legacy mode requires --metrics.")

    data = load_single_metrics_csv(args.metrics_csv, args.key_session_regex)
    missing = [m for m in args.metrics if m not in data.columns]
    if missing:
        raise KeyError(f"Metrics not found in CSV columns: {missing}")

    out = Path(args.out) if args.out else Path(args.metrics_csv).with_name("metrics.png")
    plot_legacy_single_csv(
        data=data,
        metrics=args.metrics,
        out=out,
        dpi=args.dpi,
        pfg=args.pfg,
        show_sample_counts=args.show_sample_counts,
        seed=args.seed,
        max_points_per_group=args.max_points_per_group,
    )


def run_echi_results_mode(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")
    experiment_name = results_dir.parent.name if results_dir.parent.name else results_dir.name

    combined, modes, summary_df = load_echi_results_dir(
        results_dir=results_dir,
        session_regex=args.session_regex,
    )
    default_metrics = [m for m in ECHI_DEFAULT_METRICS if m in combined.columns]
    metrics = args.metrics if args.metrics else default_metrics
    if not metrics:
        raise ValueError("Could not infer any metrics to plot. Provide --metrics explicitly.")

    missing = [m for m in metrics if m not in combined.columns]
    if missing:
        raise KeyError(f"Metrics not found in combined CSV columns: {missing}")

    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "viz_matplotlib")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_out = out_dir / f"{experiment_name}_metrics_valid_combined.csv"
    combined.to_csv(combined_out, index=False)
    print(f"Saved: {combined_out}")

    by_mode = summarize_metrics(combined, ["mixture_mode"], metrics)
    by_mode_out = out_dir / f"{experiment_name}_summary_by_mode.csv"
    by_mode.to_csv(by_mode_out, index=False)
    print(f"Saved: {by_mode_out}")

    if "n_active" in combined.columns:
        with_n_active = combined.dropna(subset=["n_active"]).copy()
        by_mode_n_active = summarize_metrics(with_n_active, ["mixture_mode", "n_active"], metrics)
        by_mode_n_active_out = out_dir / f"{experiment_name}_summary_by_mode_n_active.csv"
        by_mode_n_active.to_csv(by_mode_n_active_out, index=False)
        print(f"Saved: {by_mode_n_active_out}")

    if "session" in combined.columns:
        by_mode_session = summarize_metrics(combined, ["mixture_mode", "session"], metrics)
        by_mode_session_out = out_dir / f"{experiment_name}_summary_by_mode_session.csv"
        by_mode_session.to_csv(by_mode_session_out, index=False)
        print(f"Saved: {by_mode_session_out}")

    if summary_df is not None:
        summary_copy = out_dir / f"{experiment_name}_metrics_valid_summary_original.csv"
        summary_df.to_csv(summary_copy, index=False)
        print(f"Saved: {summary_copy}")

    plot_distributions_by_mode(
        df=combined,
        modes=modes,
        metrics=metrics,
        out_png=out_dir / f"{experiment_name}_distributions_by_mode.png",
        dpi=args.dpi,
        pfg=args.pfg,
        show_sample_counts=args.show_sample_counts,
        seed=args.seed,
        max_points_per_group=args.max_points_per_group,
    )
    plot_trends_by_n_active(
        df=combined,
        modes=modes,
        metrics=metrics,
        out_png=out_dir / f"{experiment_name}_trends_by_n_active.png",
        dpi=args.dpi,
        pfg=args.pfg,
    )
    _ = plot_session_delta_heatmap(
        df=combined,
        metrics=metrics,
        out_png=out_dir / f"{experiment_name}_session_delta_heatmap.png",
        dpi=args.dpi,
        pfg=args.pfg,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--metrics_csv",
        type=str,
        default=None,
        help="Path to a single metrics CSV (legacy mode).",
    )
    group.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to ECHI experiment results dir containing metrics_valid_*.csv files.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric columns to plot. Optional in --results_dir mode (defaults to ECHI metrics).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Legacy mode only: output PNG path.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Results-dir mode only: output directory for processed CSVs + figures.",
    )
    ap.add_argument(
        "--key_session_regex",
        type=str,
        default=r"(dev_\d+)",
        help=(
            "Legacy mode: regex on 'key' to derive session if 'session' column is absent "
            "(default: '(dev_\\d+)')."
        ),
    )
    ap.add_argument(
        "--session_regex",
        type=str,
        default=r"/(dev_\d+)\.",
        help="Results-dir mode: regex on 'snt_id' to derive session (default: '/(dev_\\d+)\\.').",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pfg",
        action="store_true",
        help="Also write PGF outputs (default: PNG only).",
    )
    ap.add_argument(
        "--max_points_per_group",
        type=int,
        default=900,
        help="Cap number of jittered points per group to keep plots readable.",
    )
    ap.add_argument(
        "--show_sample_counts",
        action="store_true",
        help="Show sample-count annotation in the top-right of box/scatter subplots.",
    )
    return ap


def main():
    args = build_parser().parse_args()
    configure_plot_fonts()
    if args.results_dir:
        run_echi_results_mode(args)
    else:
        run_legacy_mode(args)


if __name__ == "__main__":
    main()
