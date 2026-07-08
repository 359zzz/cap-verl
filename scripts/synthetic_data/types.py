"""Shared types for the synthetic data generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class GeneratedCode:
    """A generated successful program and its decomposed logical blocks."""

    full_code: str
    blocks: list[str]


class CodeGenerator(Protocol):
    """Protocol for task-specific synthetic code generators."""

    def generate(self, rng: np.random.Generator) -> GeneratedCode:
        """Generate the full successful code for one episode.

        Returns both the executable full program and a list of logical blocks
        that can be grouped into multi-turn trajectories.
        """
        ...

    def split_turns(self, generated: GeneratedCode, n_turns: int) -> list[str]:
        """Group ``generated.blocks`` into ``n_turns`` executable turn strings."""
        ...
