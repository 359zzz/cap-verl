"""Environment loading and prompt extraction utilities."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any


def _capx_root() -> Path:
    """Return the absolute path to the ``cap-x`` submodule root.

    ``cap-x`` uses relative paths in several default controller configs, so we
    need to be in the ``cap-x`` root while instantiating environments.
    """
    return Path(__file__).resolve().parents[2] / "cap-x"


@contextlib.contextmanager
def _capx_working_dir():
    """Temporarily switch the working directory to ``cap-x`` root."""
    capx_root = _capx_root()
    if not capx_root.exists():
        raise FileNotFoundError(f"cap-x submodule not found at {capx_root}")
    original = os.getcwd()
    try:
        os.chdir(capx_root)
        yield
    finally:
        os.chdir(original)


def load_code_env(yaml_path: Path) -> Any:
    """Load a ``CodeExecutionEnvBase`` from a Hydra-style YAML factory.

    The YAML must contain a top-level ``env`` key with ``_target_`` and ``cfg``
    fields, matching the existing cap-x launcher conventions.
    """
    # Defer cap-x imports so this module can be imported for --help / unit
    # tests in environments where cap-x is not installed.
    from capx.envs.configs.instantiate import instantiate as capx_instantiate
    from capx.envs.configs.loader import DictLoader
    from capx.envs.tasks.base import CodeExecutionEnvBase

    if not yaml_path.exists():
        raise FileNotFoundError(f"Environment config not found: {yaml_path}")

    configs_dict = DictLoader.load([str(yaml_path)])
    if "env" not in configs_dict:
        raise ValueError(f"YAML config {yaml_path} must contain an 'env' key")

    env_factory = configs_dict["env"]

    # cap-x default controller configs use relative paths; run instantiation
    # from the cap-x root directory.
    with _capx_working_dir():
        env = capx_instantiate(env_factory)

    if not isinstance(env, CodeExecutionEnvBase):
        raise TypeError(
            f"Instantiated env is {type(env).__name__}, expected CodeExecutionEnvBase"
        )
    return env


def extract_initial_messages(env: Any) -> list[dict[str, Any]]:
    """Extract the system + first-user messages from a freshly built env.

    The user message is produced by ``env._get_complete_prompt()``, which
    concatenates the task instruction with API documentation. The system
    message comes from the env's configured system prompt.
    """
    system_content = env._system_prompt
    user_content = env._get_complete_prompt()
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
