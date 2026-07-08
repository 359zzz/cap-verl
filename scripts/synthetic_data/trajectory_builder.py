"""Build ``messages`` conversation trajectories from generated code."""

from __future__ import annotations

from typing import Any

from synthetic_data.perturbations import PerturbationSpec


def _success_feedback(turn_idx: int, total_turns: int, task_name: str) -> str:
    """Generate a concise user feedback after a successful intermediate turn."""
    if turn_idx + 1 < total_turns:
        return (
            f"Code executed successfully (step {turn_idx + 1}/{total_turns}). "
            "Please continue with the next step."
        )
    return "The previous step completed successfully. Please finish the task."


def build_single_turn_success(
    initial_messages: list[dict[str, Any]],
    code: str,
) -> list[dict[str, Any]]:
    """Build a single-turn successful trajectory."""
    messages = [msg.copy() for msg in initial_messages]
    messages.append({"role": "assistant", "content": code})
    return messages


def build_multi_turn_success(
    initial_messages: list[dict[str, Any]],
    turn_codes: list[str],
    task_name: str,
) -> list[dict[str, Any]]:
    """Build a multi-turn successful trajectory.

    Each intermediate turn is followed by a short user feedback message;
    the final assistant turn is the last message in the conversation.
    """
    messages = [msg.copy() for msg in initial_messages]
    total = len(turn_codes)
    for idx, turn_code in enumerate(turn_codes):
        messages.append({"role": "assistant", "content": turn_code})
        if idx + 1 < total:
            messages.append(
                {
                    "role": "user",
                    "content": _success_feedback(idx, total, task_name),
                }
            )
    return messages


def build_error_correction(
    initial_messages: list[dict[str, Any]],
    error_code: str,
    correction_code: str,
    perturbation: PerturbationSpec,
    task_name: str,
) -> list[dict[str, Any]]:
    """Build a 2-turn error/correction trajectory."""
    messages = [msg.copy() for msg in initial_messages]
    messages.append({"role": "assistant", "content": error_code})
    feedback = (
        f"The previous code failed: {perturbation.description} "
        "Please regenerate the full code to complete the task, or type FINISH if done."
    )
    messages.append({"role": "user", "content": feedback})
    messages.append({"role": "assistant", "content": correction_code})
    return messages
