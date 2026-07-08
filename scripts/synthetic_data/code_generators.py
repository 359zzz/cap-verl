"""Task-specific synthetic code generators.

Each generator produces executable Python code that uses the same high-level
API functions the target model will see at inference time. When executed in the
privileged environment the API calls resolve to ground-truth states, giving us
a 100% successful oracle program without hard-coding coordinates.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import numpy as np

from synthetic_data.types import GeneratedCode


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _indent(lines: str, prefix: str = "    ") -> str:
    return textwrap.indent(lines, prefix)


def _group_blocks(blocks: Sequence[str], n_turns: int) -> list[str]:
    """Group ``blocks`` into ``n_turns`` adjacent chunks.

    Blocks are concatenated in order; the grouping is greedy (earlier turns get
    any remainder). ``n_turns`` must be <= len(blocks).
    """
    if n_turns > len(blocks):
        raise ValueError(
            f"Cannot split {len(blocks)} code block(s) into {n_turns} turns"
        )

    base = len(blocks) // n_turns
    extra = len(blocks) % n_turns
    turns: list[str] = []
    idx = 0
    for turn_idx in range(n_turns):
        size = base + (1 if turn_idx < extra else 0)
        turns.append("\n\n".join(blocks[idx : idx + size]).strip())
        idx += size
    return turns


# -----------------------------------------------------------------------------
# Franka lift
# -----------------------------------------------------------------------------


class LiftCodeGenerator:
    """Code generator for franka_lift_code_env.

    The oracle behavior is a single grasp-and-lift motion. Randomized control
    parameters (approach height, lift height) create diversity without changing
    the task semantics.
    """

    DEFAULT_Z_APPROACH_RANGE = (0.05, 0.15)
    DEFAULT_LIFT_HEIGHT_RANGE = (0.08, 0.15)

    def __init__(
        self,
        z_approach_range: tuple[float, float] | None = None,
        lift_height_range: tuple[float, float] | None = None,
    ) -> None:
        self.z_approach_range = z_approach_range or self.DEFAULT_Z_APPROACH_RANGE
        self.lift_height_range = lift_height_range or self.DEFAULT_LIFT_HEIGHT_RANGE

    def generate(self, rng: np.random.Generator) -> GeneratedCode:
        z_approach = float(rng.uniform(*self.z_approach_range))
        lift_height = float(rng.uniform(*self.lift_height_range))

        full_code = textwrap.dedent(
            f"""\
            import numpy as np

            # Get a grasp pose for the red cube
            grasp_pos, grasp_quat = sample_grasp_pose("red cube")

            # Open the gripper before approaching
            open_gripper()

            # Approach the grasp pose from above
            goto_pose(grasp_pos, grasp_quat, z_approach={z_approach:.3f})

            # Move to the exact grasp pose
            goto_pose(grasp_pos, grasp_quat)

            # Close the gripper to grasp the cube
            close_gripper()

            # Lift the cube to a safe height
            lift_offset = np.array([0.0, 0.0, {lift_height:.3f}])
            lift_pos = grasp_pos + lift_offset
            goto_pose(lift_pos, grasp_quat)
            """
        ).strip()
        return GeneratedCode(full_code=full_code, blocks=[full_code])

    def split_turns(self, generated: GeneratedCode, n_turns: int) -> list[str]:
        # Lift is treated as a single-turn task; error/correction pairs use the
        # full code directly.
        if n_turns != 1:
            raise ValueError("franka_lift_code_env only supports single-turn success")
        return [generated.full_code]


# -----------------------------------------------------------------------------
# Franka nut assembly
# -----------------------------------------------------------------------------


class NutAssemblyCodeGenerator:
    """Code generator for franka_nut_assembly_code_env.

    The oracle behavior is split into four logical blocks:
      1. imports + helper functions
      2. grasp the nut handle
      3. compute handle-to-center transform and move above the peg
      4. insert and release

    Multi-turn trajectories group adjacent blocks (2, 3, or 4 turns).
    """

    DEFAULT_Z_APPROACH_RANGE = (0.03, 0.08)
    DEFAULT_INSERT_OFFSET_RANGE = (0.015, 0.030)

    def __init__(
        self,
        z_approach_range: tuple[float, float] | None = None,
        insert_offset_range: tuple[float, float] | None = None,
    ) -> None:
        self.z_approach_range = z_approach_range or self.DEFAULT_Z_APPROACH_RANGE
        self.insert_offset_range = insert_offset_range or self.DEFAULT_INSERT_OFFSET_RANGE

    def generate(self, rng: np.random.Generator) -> GeneratedCode:
        z_approach = float(rng.uniform(*self.z_approach_range))
        insert_offset = float(rng.uniform(*self.insert_offset_range))

        imports_block = textwrap.dedent(
            """\
            import numpy as np
            from scipy.spatial.transform import Rotation as R

            def pose_to_matrix(pos, quat):
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy expects xyzw
                mat = np.eye(4)
                mat[:3, :3] = rot.as_matrix()
                mat[:3, 3] = pos
                return mat

            def matrix_to_pose(mat):
                pos = mat[:3, 3]
                rot = R.from_matrix(mat[:3, :3])
                quat_xyzw = rot.as_quat()
                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                return pos, quat_wxyz

            def flip_xy_axis(mat):
                flip_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                return mat @ flip_mat
            """
        ).strip()

        perception_block = textwrap.dedent(
            f"""\
            # Sample grasp pose for the nut handle and query object poses
            handle_pos, handle_quat = sample_grasp_pose("extruded handle of the brown square nut")
            nut_pos, _ = get_object_pose("white hollow center of the brown square nut")
            peg_pos, peg_quat = get_object_pose("square block")

            # Ensure the handle x-axis points toward the nut center
            v_world = nut_pos - handle_pos
            handle_orientation = R.from_quat([handle_quat[1], handle_quat[2], handle_quat[3], handle_quat[0]]).as_matrix()
            if v_world @ handle_orientation[:, 0] < 0:
                handle_orientation = flip_xy_axis(handle_orientation)
                handle_xyzw = R.from_matrix(handle_orientation).as_quat()
                handle_quat = np.array([handle_xyzw[3], handle_xyzw[0], handle_xyzw[1], handle_xyzw[2]])
            """
        ).strip()

        grasp_block = textwrap.dedent(
            f"""\
            # Open the gripper and approach the handle
            open_gripper()
            goto_pose(handle_pos, handle_quat, z_approach={z_approach:.3f})

            # Grasp the handle
            close_gripper()
            """
        ).strip()

        align_block = textwrap.dedent(
            """\
            # Return to home for a better IK solution before inserting
            goto_home_joint_position()

            # Compute the rigid transform from handle grasp to nut center
            T_handle = pose_to_matrix(handle_pos, handle_quat)
            T_nut = pose_to_matrix(nut_pos, handle_quat)
            T_handle_to_center = np.linalg.inv(T_nut) @ T_handle

            # Compute the desired handle pose for insertion onto the peg
            T_peg = pose_to_matrix(peg_pos, peg_quat)
            T_desired_handle = T_peg @ T_handle_to_center
            desired_handle_pos, desired_handle_quat = matrix_to_pose(T_desired_handle)

            # Move above the peg
            goto_pose(desired_handle_pos, desired_handle_quat, z_approach=0.05)
            """
        ).strip()

        insert_block = textwrap.dedent(
            f"""\
            # Final insertion
            final_pos = desired_handle_pos + np.array([0.0, 0.0, -{insert_offset:.3f}])
            goto_pose(final_pos, desired_handle_quat, z_approach=0.0)

            # Release the nut
            open_gripper()
            """
        ).strip()

        blocks = [imports_block, perception_block, grasp_block, align_block, insert_block]
        full_code = "\n\n".join(blocks)
        return GeneratedCode(full_code=full_code, blocks=blocks)

    def split_turns(self, generated: GeneratedCode, n_turns: int) -> list[str]:
        if n_turns not in (2, 3, 4):
            raise ValueError(
                "franka_nut_assembly_code_env supports 2, 3, or 4 turn success trajectories"
            )
        return _group_blocks(generated.blocks, n_turns)
