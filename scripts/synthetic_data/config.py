"""Task registry and configuration for synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from synthetic_data.code_generators import LiftCodeGenerator, NutAssemblyCodeGenerator
from synthetic_data.perturbations import (
    PerturbationSpec,
    remove_line_containing,
    replace_argument,
    replace_assignment,
)
from synthetic_data.types import CodeGenerator


@dataclass
class TaskConfig:
    """Configuration bundle for a synthetic-data task.

    Adding a new task only requires:
      1. Two YAML env configs (visual + privileged).
      2. A ``CodeGenerator`` implementation.
      3. A list of ``PerturbationSpec`` for error/correction pairs.
      4. Registration in ``TASK_REGISTRY`` below.
    """

    task_name: str
    visual_yaml: Path
    privileged_yaml: Path
    category: str  # e.g. "single_turn", "multi_turn_simple"
    turn_distribution: dict[int, float]
    error_correction_ratio: float
    code_generator: CodeGenerator
    perturbation_specs: list[PerturbationSpec] = field(default_factory=list)

    def sample_turn_count(self, rng: np.random.Generator) -> int:
        """Sample a turn count for a *successful* trajectory."""
        counts = list(self.turn_distribution.keys())
        weights = np.asarray([self.turn_distribution[c] for c in counts], dtype=float)
        weights /= weights.sum()
        return int(rng.choice(counts, p=weights))


# -----------------------------------------------------------------------------
# Task registry
# -----------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[2]


TASK_REGISTRY: dict[str, TaskConfig] = {}


def register_task(config: TaskConfig) -> None:
    """Register a task configuration."""
    TASK_REGISTRY[config.task_name] = config


def get_task_config(task_name: str) -> TaskConfig:
    """Look up a registered task configuration."""
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task {task_name!r}. Registered tasks: {list(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_name]

# -----------------------------------------------------------------------------
# Task registry
# -----------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parents[2]


TASK_REGISTRY: dict[str, TaskConfig] = {}


def register_task(config: TaskConfig) -> None:
    """Register a task configuration."""
    TASK_REGISTRY[config.task_name] = config


def get_task_config(task_name: str) -> TaskConfig:
    """Look up a registered task configuration."""
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task {task_name!r}. Registered tasks: {list(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task_name]


# -----------------------------------------------------------------------------
# Register first-phase tasks
# -----------------------------------------------------------------------------

_ROOT = _project_root()

register_task(
    TaskConfig(
        task_name="franka_lift_code_env",
        visual_yaml=_ROOT / "env_configs" / "synthetic" / "franka_lift_visual.yaml",
        privileged_yaml=_ROOT
        / "env_configs"
        / "synthetic"
        / "franka_lift_privileged.yaml",
        category="single_turn",
        turn_distribution={1: 1.0},
        error_correction_ratio=0.2,
        code_generator=LiftCodeGenerator(),
        perturbation_specs=[
            PerturbationSpec(
                name="missing_open_gripper",
                weight=0.3,
                description="Omit open_gripper() before grasping",
                apply=remove_line_containing("open_gripper()"),
            ),
            PerturbationSpec(
                name="z_approach_zero",
                weight=0.35,
                description="Set the first z_approach to 0.0 so the gripper crashes into the cube",
                apply=replace_argument("z_approach", "0.0"),
            ),
            PerturbationSpec(
                name="lift_height_too_low",
                weight=0.35,
                description="Set lift height too low to count as a successful lift",
                apply=replace_assignment(
                    "lift_offset", "np.array([0.0, 0.0, 0.030])"
                ),
            ),
        ],
    )
)

register_task(
    TaskConfig(
        task_name="franka_nut_assembly_code_env",
        visual_yaml=_ROOT
        / "env_configs"
        / "synthetic"
        / "franka_nut_assembly_visual.yaml",
        privileged_yaml=_ROOT
        / "env_configs"
        / "synthetic"
        / "franka_nut_assembly_privileged.yaml",
        category="multi_turn_simple",
        turn_distribution={2: 0.1, 3: 0.7, 4: 0.2},
        error_correction_ratio=0.3,
        code_generator=NutAssemblyCodeGenerator(),
        perturbation_specs=[
            PerturbationSpec(
                name="grasp_z_approach_zero",
                weight=0.4,
                description="Set the grasp z_approach to 0.0 so the gripper misses the handle",
                apply=replace_argument("z_approach", "0.0"),
            ),
            PerturbationSpec(
                name="missing_home",
                weight=0.3,
                description="Omit goto_home_joint_position() before computing the insertion pose",
                apply=remove_line_containing("goto_home_joint_position()"),
            ),
            PerturbationSpec(
                name="shallow_insertion",
                weight=0.3,
                description="Remove the final downward insertion offset",
                apply=replace_assignment("final_pos", "desired_handle_pos"),
            ),
        ],
    )
)
