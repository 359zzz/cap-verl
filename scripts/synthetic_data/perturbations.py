"""Perturbation strategies for building error/correction pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PerturbationSpec:
    """Specification for one type of synthetic error."""

    name: str
    weight: float
    description: str
    apply: Callable[[str, np.random.Generator], str]


def remove_line_containing(substring: str) -> Callable[[str, np.random.Generator], str]:
    """Return an applicator that removes the first line containing ``substring``."""

    def _apply(code: str, _rng: np.random.Generator) -> str:
        lines = code.splitlines(keepends=True)
        filtered = [ln for ln in lines if substring not in ln]
        return "".join(filtered)

    return _apply


def replace_assignment(
    var_name: str,
    new_value: str | Callable[[np.random.Generator], str],
) -> Callable[[str, np.random.Generator], str]:
    """Replace ``var_name = <value>`` with a new value.

    ``var_name`` is treated as a literal Python identifier and matched with
    word boundaries. ``new_value`` may be a static string or a callable that
    receives the RNG and returns a string.
    """

    def _apply(code: str, rng: np.random.Generator) -> str:
        value = new_value(rng) if callable(new_value) else new_value
        pattern = rf"(\b{re.escape(var_name)}\s*=\s*)[^\n#]+"
        return re.sub(pattern, rf"\g<1>{value}", code, count=1)

    return _apply


def replace_argument(
    arg_name: str,
    new_value: str | Callable[[np.random.Generator], str],
) -> Callable[[str, np.random.Generator], str]:
    """Replace the first occurrence of ``arg_name=<value>`` with a new value."""

    def _apply(code: str, rng: np.random.Generator) -> str:
        value = new_value(rng) if callable(new_value) else new_value
        # Match arg_name=<number or expression> on a single line.
        pattern = rf"\b{re.escape(arg_name)}\s*=\s*([^\n,)]+)"
        return re.sub(pattern, f"{arg_name}={value}", code, count=1)

    return _apply


def replace_string(
    old: str,
    new: str,
) -> Callable[[str, np.random.Generator], str]:
    """Replace the first occurrence of a literal string with another string."""

    def _apply(code: str, _rng: np.random.Generator) -> str:
        if old not in code:
            raise ValueError(f"String {old!r} not found in code")
        return code.replace(old, new, 1)

    return _apply


def choose_perturbation(
    specs: list[PerturbationSpec],
    rng: np.random.Generator,
) -> PerturbationSpec:
    """Sample one perturbation according to spec weights."""
    if not specs:
        raise ValueError("No perturbation specs provided")
    weights = np.asarray([s.weight for s in specs], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(specs), p=weights))
    return specs[idx]


def apply_perturbation(
    code: str,
    specs: list[PerturbationSpec],
    rng: np.random.Generator,
) -> tuple[str, PerturbationSpec]:
    """Apply a sampled perturbation to ``code`` and return the result plus spec."""
    spec = choose_perturbation(specs, rng)
    return spec.apply(code, rng), spec
