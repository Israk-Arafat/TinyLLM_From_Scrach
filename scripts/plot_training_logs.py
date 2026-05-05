"""Create report-ready training plots from one or more CSV log files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TinyLLM training logs")
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[
            "/Users/xanderdufour/Downloads/training_log_first_half.csv",
            "/Users/xanderdufour/Downloads/training_log.csv",
        ],
        help="CSV log paths in chronological order",
    )
    parser.add_argument(
        "--output-dir",
        default="report_plots",
        help="Directory where PNG figures will be written",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Rolling window for smoothed train loss / grad norm / throughput",
    )
    return parser.parse_args()


def load_and_merge_logs(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, path_str in enumerate(paths):
        path = Path(path_str).expanduser()
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["phase"] = index + 1
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "opt_step",
        "tokens_seen",
        "train_loss",
        "val_loss",
        "lr",
        "grad_norm",
        "tokens_per_sec",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("opt_step").drop_duplicates(subset="opt_step", keep="last").reset_index(drop=True)

    # `tokens_seen` resets after the restart in the second CSV, so derive a stable
    # cumulative token estimate from opt_step instead of trusting the raw column.
    tokens_per_step = infer_tokens_per_step(df)
    df["tokens_seen_reconstructed"] = df["opt_step"] * tokens_per_step
    df["tokens_billions"] = df["tokens_seen_reconstructed"] / 1e9
    df["train_loss_smooth"] = df["train_loss"].rolling(window=25, min_periods=1).mean()
    df["grad_norm_smooth"] = df["grad_norm"].rolling(window=25, min_periods=1).mean()
    nonzero_tps = df["tokens_per_sec"].replace(0, np.nan)
    df["tokens_per_sec_smooth"] = nonzero_tps.rolling(window=25, min_periods=1).mean()
    return df


def infer_tokens_per_step(df: pd.DataFrame) -> float:
    deltas = df[["opt_step", "tokens_seen"]].dropna().diff()
    valid = deltas[(deltas["opt_step"] > 0) & (deltas["tokens_seen"] > 0)]
    if valid.empty:
        raise ValueError("Could not infer tokens-per-step from the provided logs.")
    return float((valid["tokens_seen"] / valid["opt_step"]).median())


def save_plot(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["opt_step"], df["train_loss"], color="#b7c4d3", linewidth=1.0, alpha=0.55, label="Train loss")
    ax.plot(df["opt_step"], df["train_loss_smooth"], color="#0f4c81", linewidth=2.4, label="Train loss (smoothed)")

    val_df = df.dropna(subset=["val_loss"])
    if not val_df.empty:
        ax.plot(
            val_df["opt_step"],
            val_df["val_loss"],
            color="#d1495b",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Validation loss",
        )

    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_plot(fig, output_dir, "loss_vs_step.png")


def plot_loss_vs_tokens(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["tokens_billions"], df["train_loss_smooth"], color="#0f4c81", linewidth=2.4, label="Train loss (smoothed)")

    val_df = df.dropna(subset=["val_loss"])
    if not val_df.empty:
        ax.plot(
            val_df["tokens_billions"],
            val_df["val_loss"],
            color="#d1495b",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Validation loss",
        )

    ax.set_title("Loss vs Tokens Seen")
    ax.set_xlabel("Tokens seen (billions, reconstructed from step count)")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_plot(fig, output_dir, "loss_vs_tokens.png")


def plot_learning_rate(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["opt_step"], df["lr"], color="#2a9d8f", linewidth=2.2)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.25)
    save_plot(fig, output_dir, "learning_rate.png")


def plot_grad_norm(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["opt_step"], df["grad_norm"], color="#f4a261", linewidth=1.0, alpha=0.45, label="Gradient norm")
    ax.plot(df["opt_step"], df["grad_norm_smooth"], color="#e76f51", linewidth=2.3, label="Gradient norm (smoothed)")
    ax.set_title("Gradient Norm During Training")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Gradient norm")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_plot(fig, output_dir, "grad_norm.png")


def plot_throughput(df: pd.DataFrame, output_dir: Path) -> None:
    throughput_df = df.dropna(subset=["tokens_per_sec_smooth"])
    if throughput_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        throughput_df["opt_step"],
        throughput_df["tokens_per_sec_smooth"],
        color="#6d597a",
        linewidth=2.2,
    )
    ax.set_title("Training Throughput")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Tokens per second")
    ax.grid(alpha=0.25)
    save_plot(fig, output_dir, "throughput.png")


def plot_phase_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1d3557", "#d62828", "#2a9d8f", "#6a4c93"]
    for phase, phase_df in df.groupby("phase"):
        ax.plot(
            phase_df["opt_step"],
            phase_df["train_loss_smooth"],
            linewidth=2.2,
            color=colors[(phase - 1) % len(colors)],
            label=f"Phase {phase}: {phase_df['source_file'].iloc[0]}",
        )

    ax.set_title("Smoothed Training Loss by Log File")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Train loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_plot(fig, output_dir, "phase_comparison.png")


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-whitegrid")

    df = load_and_merge_logs(args.logs)

    # Respect the CLI smoothing argument after merge-time defaults are available.
    df["train_loss_smooth"] = df["train_loss"].rolling(window=args.rolling_window, min_periods=1).mean()
    df["grad_norm_smooth"] = df["grad_norm"].rolling(window=args.rolling_window, min_periods=1).mean()
    nonzero_tps = df["tokens_per_sec"].replace(0, np.nan)
    df["tokens_per_sec_smooth"] = nonzero_tps.rolling(window=args.rolling_window, min_periods=1).mean()

    output_dir = Path(args.output_dir)
    plot_loss_curves(df, output_dir)
    plot_loss_vs_tokens(df, output_dir)
    plot_learning_rate(df, output_dir)
    plot_grad_norm(df, output_dir)
    plot_throughput(df, output_dir)
    plot_phase_comparison(df, output_dir)

    print(f"Wrote plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
