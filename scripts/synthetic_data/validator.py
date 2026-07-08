"""Execute generated code in a privileged environment and validate outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Result of validating one trajectory."""

    ok: bool
    reward: float
    info: dict[str, Any]
    turn: int | None = None  # 1-based turn index where outcome was recorded


def _step_ok(reward: float, info: dict[str, Any]) -> bool:
    """Return True if a single step executed without error and succeeded."""
    return bool(info.get("sandbox_rc", 1) == 0 and reward >= 1.0)


def _step_executed(reward: float, info: dict[str, Any]) -> bool:
    """Return True if a single step ran without a sandbox exception."""
    return bool(info.get("sandbox_rc", 1) == 0)


def validate_success(
    env: Any,
    code: str,
    seed: int,
) -> ValidationResult:
    """Validate a single-turn successful program."""
    env.reset(seed=seed)
    _obs, reward, _terminated, _truncated, info = env.step(code)
    return ValidationResult(
        ok=_step_ok(reward, info),
        reward=float(reward),
        info=info,
    )


def validate_multi_turn_success(
    env: Any,
    turn_codes: list[str],
    seed: int,
) -> ValidationResult:
    """Validate a multi-turn successful program.

    Intermediate turns must execute without exceptions. The final turn must
    yield reward == 1.0.
    """
    env.reset(seed=seed)
    for idx, turn_code in enumerate(turn_codes[:-1]):
        _obs, reward, _terminated, _truncated, info = env.step(turn_code)
        if not _step_executed(reward, info):
            return ValidationResult(
                ok=False,
                reward=float(reward),
                info=info,
                turn=idx + 1,
            )

    _obs, reward, _terminated, _truncated, info = env.step(turn_codes[-1])
    return ValidationResult(
        ok=_step_ok(reward, info),
        reward=float(reward),
        info=info,
        turn=len(turn_codes),
    )


def validate_error_correction(
    env: Any,
    error_code: str,
    correction_code: str,
    seed: int,
) -> tuple[ValidationResult, ValidationResult]:
    """Validate an error/correction pair.

    The error code must fail (reward < 1.0 or exception), and the correction
    code must succeed after resetting to the same initial state.
    """
    env.reset(seed=seed)
    _obs, err_reward, _terminated, _truncated, err_info = env.step(error_code)
    error_ok = not _step_ok(err_reward, err_info)
    error_result = ValidationResult(
        ok=error_ok,
        reward=float(err_reward),
        info=err_info,
    )

    env.reset(seed=seed)
    _obs, corr_reward, _terminated, _truncated, corr_info = env.step(correction_code)
    correction_ok = _step_ok(corr_reward, corr_info)
    correction_result = ValidationResult(
        ok=correction_ok,
        reward=float(corr_reward),
        info=corr_info,
    )

    return error_result, correction_result
