"""Compute and plot multi-layer angular distances for a Hugging Face model."""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

try:
    from .utils.short_hf import ShortHFModel
except ImportError:
    from utils.short_hf import ShortHFModel


def normalize_rows_robustly(data: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Percentile-clip and normalize each row to [0, 1]."""
    normalized = np.zeros_like(data, dtype=float)
    for index, row in enumerate(data):
        low = np.nanpercentile(row, p_low)
        high = np.nanpercentile(row, p_high)
        if low == high:
            continue
        clipped = np.clip(row, low, high)
        span = np.max(clipped) - np.min(clipped)
        if span > 0:
            normalized[index] = (clipped - np.min(clipped)) / span
    return normalized


def plot_angular_distance_heatmap(
    data: np.ndarray,
    layer_count: int,
    output_path: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Save a triangular heatmap of normalized angular distances."""
    row_count, column_count = data.shape
    layer_indices = np.arange(column_count)
    offsets = np.arange(row_count)
    mask = layer_indices[np.newaxis, :] + offsets[:, np.newaxis] + 1 >= layer_count
    masked_data = np.where(mask, np.nan, data)

    fig, axis = plt.subplots(figsize=(6, 4))
    mesh = axis.pcolormesh(
        np.arange(column_count + 1),
        np.arange(row_count + 1),
        masked_data,
        cmap="viridis_r",
        edgecolors="white",
        linewidth=0.2,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_aspect("equal")
    axis.set_xlabel(r"Layer Index $\ell$")
    axis.set_ylabel(r"Subsequent $n^{th}$ Layer")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(bottom=False, left=False)
    axis.set_xticks(np.arange(0, column_count, 5 if column_count > 5 else 1))
    axis.set_yticks(np.arange(0, row_count, 4 if row_count > 4 else 1))
    fig.colorbar(mesh, ax=axis, fraction=0.046, pad=0.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """Run the angular-distance sweep."""
    model = ShortHFModel(
        model_name=args.model,
        layers_path=args.layers_path,
        n_prune_layers=1,
        device=args.device,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    layer_count = len(model.layers)
    all_distances = []
    for offset in range(1, layer_count):
        model.importances = [0.0 for _ in range(layer_count)]
        for index, record in enumerate(tqdm(dataset, desc=f"offset={offset}")):
            if index >= args.samples:
                break
            model.eval_importance(
                prompts=[record["text"]],
                max_seq_len=args.max_seq_len,
                stride=args.stride,
                angular=True,
                n=offset,
            )
        all_distances.append(model.importances)

    distances = np.asarray(all_distances) / max(args.samples, 1)
    normalized = normalize_rows_robustly(distances)
    plot_angular_distance_heatmap(normalized, layer_count, args.output)
    mpl.rcParams.update(mpl.rcParamsDefault)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layers-path", default="model.layers")
    parser.add_argument("--dataset", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--output", default="qwen3_angular_distance.pdf")
    parser.add_argument("--device")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
