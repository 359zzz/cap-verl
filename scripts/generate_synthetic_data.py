"""Generate synthetic SFT data for Cap-X + VeRL first-phase tasks.

Example:
    python scripts/generate_synthetic_data.py \
        --task franka_lift_code_env \
        --train-size 100 \
        --val-size 20 \
        --seed 42

The script uses a visual cap-x environment to extract the model prompt and a
privileged environment to generate and validate oracle code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure ``scripts/synthetic_data/`` is importable whether this file is run as
# ``python scripts/generate_synthetic_data.py`` or ``python -m scripts.generate_synthetic_data``.
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import numpy as np

from synthetic_data.config import TaskConfig, get_task_config
from synthetic_data.env_utils import extract_initial_messages, load_code_env
from synthetic_data.perturbations import apply_perturbation
from synthetic_data.trajectory_builder import (
    build_error_correction,
    build_multi_turn_success,
    build_single_turn_success,
)
from synthetic_data.validator import (
    validate_error_correction,
    validate_multi_turn_success,
    validate_success,
)
from synthetic_data.writer import build_row, write_manifest, write_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for cap-x code tasks."
    )
    parser.add_argument("--task", required=True, help="Task name from TASK_REGISTRY")
    parser.add_argument(
        "--train-size", type=int, required=True, help="Number of training samples"
    )
    parser.add_argument(
        "--val-size", type=int, default=0, help="Number of validation samples"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for parquet and manifest",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (currently only 1 is fully supported)",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=5,
        help="Stop after this many attempts per split (multiplier of target size)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip privileged-env validation (for quick debugging only)",
    )
    parser.add_argument(
        "--save-failed",
        action="store_true",
        help="Write failed seeds to failed_samples.jsonl",
    )
    return parser.parse_args()


def _generate_split(
    config: TaskConfig,
    visual_env: Any,
    privileged_env: Any,
    split: str,
    target_size: int,
    seed_base: int,
    index_start: int,
    max_attempts: int,
    verify: bool,
    failed_records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Generate ``target_size`` validated samples for one split."""
    rows: list[dict[str, Any]] = []
    idx = index_start
    attempts = 0

    while len(rows) < target_size and attempts < max_attempts:
        seed = seed_base + idx
        try:
            # Synchronise both envs to the same underlying scene.
            visual_env.reset(seed=seed)
            initial_messages = extract_initial_messages(visual_env)

            rng = np.random.default_rng(seed)
            generated = config.code_generator.generate(rng)

            if rng.random() < config.error_correction_ratio:
                # ---- error / correction pair (always 2 turns) ----
                error_code, perturbation = apply_perturbation(
                    generated.full_code,
                    config.perturbation_specs,
                    rng,
                )
                correction_code = generated.full_code

                if verify:
                    err_result, corr_result = validate_error_correction(
                        privileged_env, error_code, correction_code, seed
                    )
                    if not err_result.ok:
                        raise RuntimeError(
                            f"error code unexpectedly succeeded (reward={err_result.reward})"
                        )
                    if not corr_result.ok:
                        raise RuntimeError(
                            f"correction code failed (reward={corr_result.reward})"
                        )

                messages = build_error_correction(
                    initial_messages,
                    error_code,
                    correction_code,
                    perturbation,
                    config.task_name,
                )
                n_turns = 2
                has_error = True
                ground_truth = correction_code
            else:
                # ---- successful trajectory ----
                n_turns = config.sample_turn_count(rng)

                if n_turns == 1:
                    code = generated.full_code
                    if verify:
                        result = validate_success(privileged_env, code, seed)
                        if not result.ok:
                            raise RuntimeError(
                                f"success code failed (reward={result.reward})"
                            )
                    messages = build_single_turn_success(initial_messages, code)
                    ground_truth = code
                else:
                    turn_codes = config.code_generator.split_turns(generated, n_turns)
                    if verify:
                        result = validate_multi_turn_success(
                            privileged_env, turn_codes, seed
                        )
                        if not result.ok:
                            raise RuntimeError(
                                f"multi-turn code failed at turn {result.turn} "
                                f"(reward={result.reward})"
                            )
                    messages = build_multi_turn_success(
                        initial_messages, turn_codes, config.task_name
                    )
                    ground_truth = generated.full_code

                has_error = False

            row = build_row(
                task_name=config.task_name,
                messages=messages,
                split=split,
                index=len(rows),
                seed=seed,
                n_turns=n_turns,
                has_error=has_error,
                ground_truth_program=ground_truth,
            )
            rows.append(row)
            logger.info(
                "%s %d/%d generated (seed=%d, turns=%d, error=%s)",
                split,
                len(rows),
                target_size,
                seed,
                n_turns,
                has_error,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Seed %d failed: %s", seed, exc)
            if failed_records is not None:
                failed_records.append(
                    {
                        "split": split,
                        "seed": seed,
                        "error": str(exc),
                    }
                )

        idx += 1
        attempts += 1

    if len(rows) < target_size:
        logger.error(
            "Only generated %d/%d %s samples after %d attempts",
            len(rows),
            target_size,
            split,
            attempts,
        )
        raise RuntimeError(
            f"Failed to generate enough {split} samples ({len(rows)}/{target_size})"
        )

    return rows


def main() -> None:
    args = _parse_args()
    config = get_task_config(args.task)

    logger.info("Loading visual environment from %s", config.visual_yaml)
    visual_env = load_code_env(config.visual_yaml)

    logger.info("Loading privileged environment from %s", config.privileged_yaml)
    privileged_env = load_code_env(config.privileged_yaml)

    output_dir = args.output_dir / args.task
    failed_records: list[dict[str, Any]] = []
    failed_path = output_dir / "failed_samples.jsonl" if args.save_failed else None

    train_rows = _generate_split(
        config=config,
        visual_env=visual_env,
        privileged_env=privileged_env,
        split="train",
        target_size=args.train_size,
        seed_base=args.seed,
        index_start=0,
        max_attempts=args.train_size * args.max_attempts_multiplier,
        verify=not args.no_verify,
        failed_records=failed_records if args.save_failed else None,
    )

    val_offset = 10_000_000  # keep val seeds far away from train seeds
    val_rows = _generate_split(
        config=config,
        visual_env=visual_env,
        privileged_env=privileged_env,
        split="val",
        target_size=args.val_size,
        seed_base=args.seed + val_offset,
        index_start=0,
        max_attempts=args.val_size * args.max_attempts_multiplier,
        verify=not args.no_verify,
        failed_records=failed_records if args.save_failed else None,
    )

    logger.info("Writing outputs to %s", output_dir)
    write_split(output_dir, "train", train_rows)
    write_split(output_dir, "val", val_rows)
    write_manifest(output_dir, args.task, args.seed, train_rows, val_rows)

    if failed_path is not None:
        import json

        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_path.open("w") as f:
            for rec in failed_records:
                f.write(json.dumps(rec) + "\n")
        logger.info("Wrote %d failed records to %s", len(failed_records), failed_path)

    logger.info(
        "Done. Generated %d train and %d val samples.", len(train_rows), len(val_rows)
    )


if __name__ == "__main__":
    main()
