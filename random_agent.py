from __future__ import annotations

import numpy as np


class RandomAgent:
    """
    Simple baseline agent for Paper.io.

    Every turn, it picks a random action from 0 to 3.
    """

    def __init__(self, agent_id: str, grid_size: int) -> None:
        self.agent_id = agent_id
        self.grid_size = grid_size
        self._rng = np.random.default_rng()

    def act(self, env) -> int:
        return int(self._rng.integers(4))