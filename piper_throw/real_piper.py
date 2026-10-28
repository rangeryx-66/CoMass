"""Future hardware adapter contract; importing this module never imports the SDK."""
from typing import Protocol
import numpy as np


class TrajectorySink(Protocol):
    def execute(self, time_s: np.ndarray, joints_rad: np.ndarray,
                gripper_opening_m: np.ndarray) -> None: ...


class RealPiper:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Hardware execution is disabled in Phase 1. Simulation only.")
