#!/usr/bin/env python3
"""Plot calibration probability-change vs success-lift correlations.

This script reads the paper-spec full SGAGSN metrics.jsonl and generates:
1. Scatter plots with Pearson r annotated.
2. Iteration line plots for the paired signals.
3. Quartile trend plots grouped by the probability-change signal.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    REPO_ROOT
    / "outputs/exp_debug/"
    / "rwd-grp-b_stb-rwd-5x_128-epi_paper-spec_latefus/metrics.jsonl"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "generated"


@dataclass(frozen=True)
class PairSpec:
    name: str
    x_key: str
    y_key: str
    x_label: str
    y_label: str


PAIR_SPECS = (
    PairSpec(
        name="prob_delta_mean_vs_success_lift",
        x_key="calibrator/prob_delta_mean",
        y_key="outcome/success_lift_vs_dataset",
        x_label="Mean probability change after refinement",
        y_label="Success lift vs dataset",
    ),
    PairSpec(
        name="prob_delta_positive_rate_vs_success_lift",
        x_key="calibrator/prob_delta_positive_rate",
        y_key="outcome/success_lift_vs_dataset",
        x_label="Fraction with positive probability change",
        y_label="Success lift vs dataset",
    ),
)


def read_metrics(path: Path) -> list[tuple[int, dict[str, float]]]:
    rows: list[tuple[int, dict[str, float]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append((int(item["step"]), item.get("stats", {})))
    return rows


def finite_float(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def extract_xy(
    rows: Iterable[tuple[int, dict[str, float]]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    for step, stats in rows:
        x = stats.get(x_key)
        y = stats.get(y_key)
        if finite_float(x) and finite_float(y):
            steps.append(step)
            xs.append(float(x))
            ys.append(float(y))
    return np.asarray(steps), np.asarray(xs), np.asarray(ys)


def pearson_r(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 3:
        return float("nan")
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def as_percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")


def add_regression_line(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    if len(xs) < 2 or float(np.std(xs)) == 0.0:
        return
    slope, intercept = np.polyfit(xs, ys, deg=1)
    grid = np.linspace(float(xs.min()), float(xs.max()), 100)
    ax.plot(grid, slope * grid + intercept, color="#2f4b7c", linewidth=2.0)


def plot_scatter_grid(
    rows: list[tuple[int, dict[str, float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    domains = (
        ("train", ""),
        ("validation", "validation/"),
    )
    for row_idx, spec in enumerate(PAIR_SPECS):
        for col_idx, (domain_name, prefix) in enumerate(domains):
            ax = axes[row_idx, col_idx]
            steps, xs, ys = extract_xy(rows, prefix + spec.x_key, prefix + spec.y_key)
            r = pearson_r(xs, ys)
            ax.scatter(
                xs,
                ys,
                s=28,
                alpha=0.68,
                color="#1f77b4" if domain_name == "train" else "#ff9f1c",
                edgecolors="white",
                linewidths=0.35,
            )
            add_regression_line(ax, xs, ys)
            ax.axhline(0.0, color="#8a8a8a", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.set_title(f"{domain_name}: {spec.name.replace('_', ' ')}")
            ax.set_xlabel(spec.x_label)
            ax.set_ylabel(spec.y_label)
            ax.grid(True, linewidth=0.55, alpha=0.28)
            as_percent_axis(ax)
            ax.text(
                0.04,
                0.96,
                f"Pearson r = {r:.3f}\nn = {len(xs)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10.5,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "alpha": 0.9,
                },
            )
    fig.suptitle("Probability-Change Signals vs Success Lift", fontsize=15)
    fig.savefig(out_dir / "scatter_prob_delta_success_lift_corr.png", dpi=220)
    fig.savefig(out_dir / "scatter_prob_delta_success_lift_corr.pdf")
    plt.close(fig)


def plot_line_grid(
    rows: list[tuple[int, dict[str, float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), constrained_layout=True)
    domains = (
        ("train", "", 15),
        ("validation", "validation/", 7),
    )
    for row_idx, spec in enumerate(PAIR_SPECS):
        for col_idx, (domain_name, prefix, window) in enumerate(domains):
            ax = axes[row_idx, col_idx]
            steps, xs, ys = extract_xy(rows, prefix + spec.x_key, prefix + spec.y_key)
            r = pearson_r(xs, ys)
            ax2 = ax.twinx()
            ax.plot(
                steps,
                xs,
                color="#1f77b4",
                alpha=0.18,
                linewidth=0.8,
            )
            ax.plot(
                steps,
                rolling_mean(xs, window),
                color="#1f77b4",
                linewidth=2.2,
                label=spec.x_label,
            )
            ax2.plot(
                steps,
                ys,
                color="#d62728",
                alpha=0.16,
                linewidth=0.8,
            )
            ax2.plot(
                steps,
                rolling_mean(ys, window),
                color="#d62728",
                linewidth=2.2,
                label=spec.y_label,
            )
            ax.axhline(0.0, color="#8a8a8a", linewidth=0.8, linestyle="--", alpha=0.55)
            ax2.axhline(0.0, color="#8a8a8a", linewidth=0.8, linestyle="--", alpha=0.35)
            ax.set_title(f"{domain_name}: rolling line, r={r:.3f}, n={len(xs)}")
            ax.set_xlabel("PPO iteration")
            ax.set_ylabel(spec.x_label, color="#1f77b4")
            ax2.set_ylabel(spec.y_label, color="#d62728")
            ax.tick_params(axis="y", labelcolor="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#d62728")
            ax.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
            ax2.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
            ax.grid(True, linewidth=0.55, alpha=0.28)
    fig.suptitle("Probability-Change Signals and Success Lift over Training", fontsize=15)
    fig.savefig(out_dir / "line_prob_delta_success_lift_corr.png", dpi=220)
    fig.savefig(out_dir / "line_prob_delta_success_lift_corr.pdf")
    plt.close(fig)


def quartile_summary(
    rows: list[tuple[int, dict[str, float]]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, xs, ys = extract_xy(rows, x_key, y_key)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    chunks_x: list[float] = []
    chunks_y: list[float] = []
    chunks_y_std: list[float] = []
    for chunk in np.array_split(np.arange(len(xs)), 4):
        chunks_x.append(float(np.mean(xs[chunk])))
        chunks_y.append(float(np.mean(ys[chunk])))
        chunks_y_std.append(float(np.std(ys[chunk])))
    return np.asarray(chunks_x), np.asarray(chunks_y), np.asarray(chunks_y_std)


def plot_quartile_grid(
    rows: list[tuple[int, dict[str, float]]],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    domains = (
        ("train", ""),
        ("validation", "validation/"),
    )
    q_labels = ["Q1 low", "Q2", "Q3", "Q4 high"]
    for row_idx, spec in enumerate(PAIR_SPECS):
        for col_idx, (domain_name, prefix) in enumerate(domains):
            ax = axes[row_idx, col_idx]
            qx, qy, qstd = quartile_summary(rows, prefix + spec.x_key, prefix + spec.y_key)
            xpos = np.arange(4)
            ax.errorbar(
                xpos,
                qy,
                yerr=qstd,
                color="#2f4b7c",
                marker="o",
                markersize=7,
                linewidth=2.2,
                capsize=4,
            )
            for idx, (x_val, y_val) in enumerate(zip(qx, qy)):
                ax.text(
                    idx,
                    y_val,
                    f"x={x_val * 100:.1f}%\ny={y_val * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax.axhline(0.0, color="#8a8a8a", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.set_xticks(xpos, q_labels)
            ax.set_title(f"{domain_name}: grouped by {spec.x_label}")
            ax.set_ylabel("Mean success lift vs dataset")
            ax.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
            ax.grid(True, axis="y", linewidth=0.55, alpha=0.28)
    fig.suptitle("Success Lift by Probability-Change Quartile", fontsize=15)
    fig.savefig(out_dir / "quartile_prob_delta_success_lift.png", dpi=220)
    fig.savefig(out_dir / "quartile_prob_delta_success_lift.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = read_metrics(args.metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter_grid(rows, args.out_dir)
    plot_line_grid(rows, args.out_dir)
    plot_quartile_grid(rows, args.out_dir)

    print(f"Read metrics: {args.metrics}")
    print(f"Rows: {len(rows)}")
    print(f"Saved figures to: {args.out_dir}")
    for spec in PAIR_SPECS:
        for domain_name, prefix in (("train", ""), ("validation", "validation/")):
            _, xs, ys = extract_xy(rows, prefix + spec.x_key, prefix + spec.y_key)
            print(f"{domain_name:10s} {spec.name:46s} r={pearson_r(xs, ys):.6f} n={len(xs)}")


if __name__ == "__main__":
    main()
