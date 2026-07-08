"""Write generated trajectories to parquet and a JSON manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars to Python built-ins."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ensure_serializable(v) for v in obj]
    return obj


def write_split(
    output_dir: Path,
    split: str,
    rows: list[dict[str, Any]],
) -> Path:
    """Write one split (train/val) to parquet and return the file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{split}.parquet"

    df = pd.DataFrame(rows)
    if df.empty:
        # Write an empty parquet with the expected schema so downstream tools
        # can still read the file without schema errors.
        df = pd.DataFrame(
            columns=[
                "data_source",
                "messages",
                "ability",
                "reward_model",
                "extra_info",
            ]
        )

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    return parquet_path


def write_manifest(
    output_dir: Path,
    task: str,
    seed: int,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> Path:
    """Write a manifest summarizing the generated dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    def _count_turns(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            key = f"{row['extra_info']['n_turns']}_turn"
            counts[key] = counts.get(key, 0) + 1
        return counts

    total_rows = train_rows + val_rows
    success_rate = 1.0 if total_rows else 0.0
    avg_turns = (
        sum(r["extra_info"]["n_turns"] for r in total_rows) / len(total_rows)
        if total_rows
        else 0.0
    )

    manifest = {
        "task": task,
        "data_source": task,
        "seed": seed,
        "train": len(train_rows),
        "val": len(val_rows),
        "success_rate": success_rate,
        "avg_turns": avg_turns,
        "train_turn_counts": _count_turns(train_rows),
        "val_turn_counts": _count_turns(val_rows),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def build_row(
    task_name: str,
    messages: list[dict[str, Any]],
    split: str,
    index: int,
    seed: int,
    n_turns: int,
    has_error: bool,
    ground_truth_program: str,
) -> dict[str, Any]:
    """Construct a single parquet row in the expected schema."""
    return {
        "data_source": task_name,
        "messages": _ensure_serializable(messages),
        "ability": "agent",
        "reward_model": {
            "style": "sim_code",
            "ground_truth": {"program": ground_truth_program},
        },
        "extra_info": {
            "split": split,
            "index": index,
            "seed": seed,
            "n_turns": n_turns,
            "has_error": has_error,
            "source": "synthetic",
        },
    }
