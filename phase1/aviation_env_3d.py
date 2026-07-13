"""
Phase 1 - lightweight 3D aviation environment for reproducible baseline runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from phase2_qaoa.qubo_encoder import PARAM_GRID


ROLLOUT_HORIZON: int = 60  # timesteps per encounter simulation
ACTIONS: Tuple[str, ...] = ("maintain", "climb", "descend", "turn_left", "turn_right")
FAILURE_TYPES = {"separation_loss", "near_miss"}


def state_to_indices(state: Dict[str, float]) -> Tuple[int, int, int, int]:
    return (
        PARAM_GRID["int_heading"].index(float(state["int_heading"])),
        PARAM_GRID["int_altitude"].index(float(state["int_altitude"])),
        PARAM_GRID["int_speed"].index(float(state["int_speed"])),
        PARAM_GRID["int_x_offset"].index(float(state["int_x_offset"])),
    )


def normalize_state(state: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            (float(state["int_heading"]) - 97.5) / 22.5,
            (float(state["int_altitude"]) - 30000.0) / 2500.0,
            (float(state["int_speed"]) - 4.5) / 1.5,
            float(state["int_x_offset"]) / 20.0,
        ],
        dtype=float,
    )


def sample_state(rng: np.random.Generator) -> Dict[str, float]:
    return {
        name: float(values[int(rng.integers(0, len(values)))])
        for name, values in PARAM_GRID.items()
    }


def _ego_profile(action: str) -> Tuple[float, float]:
    if action == "climb":
        return 0.0, 20.0
    if action == "descend":
        return 0.0, -20.0
    if action == "turn_left":
        return -18.0, 0.0
    if action == "turn_right":
        return 18.0, 0.0
    return 0.0, 0.0


def simulate_encounter(state: Dict[str, float], action: str = "maintain", horizon: int = ROLLOUT_HORIZON) -> Dict[str, Any]:
    ego_x, ego_y, ego_z = 20.0, 100.0, 31000.0
    ego_heading_deg, ego_speed = _ego_profile(action)[0], 4.5
    ego_climb_rate = _ego_profile(action)[1]
    intruder_x = 88.0 + float(state["int_x_offset"])
    intruder_y = 10.0
    intruder_z = float(state["int_altitude"])
    intruder_heading_deg = float(state["int_heading"])
    intruder_speed = float(state["int_speed"])

    ex, ey = ego_x, ego_y
    ix, iy = intruder_x, intruder_y
    ez = ego_z
    min_h_sep = float("inf")
    min_v_sep = float("inf")
    separation_loss = False
    near_miss = False

    # NOTE: ego applies climb_rate per step (from _ego_profile). The qaoa_runner.py
    # evaluate_state() uses a simplified model with fixed ego altitude. Results from
    # phase2_qaoa are not directly comparable to phase1 due to this difference.
    for _ in range(horizon):
        ex += ego_speed * math.cos(math.radians(ego_heading_deg))
        ey += ego_speed * math.sin(math.radians(ego_heading_deg))
        ez += ego_climb_rate

        ix += intruder_speed * math.cos(math.radians(intruder_heading_deg))
        iy += intruder_speed * math.sin(math.radians(intruder_heading_deg))

        h_sep = math.hypot(ex - ix, ey - iy)
        v_sep = abs(ez - intruder_z)
        min_h_sep = min(min_h_sep, h_sep)
        min_v_sep = min(min_v_sep, v_sep)

        if h_sep < 5.0 and v_sep < 1000.0:
            separation_loss = True
        if h_sep < 7.5 and v_sep < 1500.0:
            near_miss = True

    if separation_loss:
        failure_type = "separation_loss"
    elif near_miss:
        failure_type = "near_miss"
    else:
        failure_type = "safe"

    reward = (
        0.25 * min_h_sep
        + 0.001 * min_v_sep
        - (20.0 if separation_loss else 0.0)
        - (8.0 if (near_miss and not separation_loss) else 0.0)
    )
    return {
        "params": state,
        "action": action,
        "min_h_sep_nm": round(min_h_sep, 3),
        "min_v_sep_ft": round(min_v_sep, 1),
        "separation_loss": separation_loss,
        "near_miss": near_miss,
        "failure": failure_type in FAILURE_TYPES,
        "failure_type": failure_type,
        "reward": round(float(reward), 6),
    }


@dataclass
class AviationEnv3D:
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.current_state: Dict[str, float] | None = None

    def reset(self) -> np.ndarray:
        self.current_state = sample_state(self.rng)
        return normalize_state(self.current_state)

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.current_state is None:
            self.reset()
        action = ACTIONS[int(action_index)]
        outcome = simulate_encounter(self.current_state, action=action)
        next_obs = self.reset()
        done = True
        return next_obs, float(outcome["reward"]), done, outcome

    @property
    def action_space_n(self) -> int:
        return len(ACTIONS)

