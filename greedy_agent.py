"""
greedy_agent.py  —  Rule-based greedy baseline agent for Paper.io
==================================================================
Kept separate from the environment so the env module stays self-contained.

Usage
-----
>>> from paper_io_env import PaperIOEnv
>>> from greedy_agent import GreedyAgent
>>> env = PaperIOEnv(grid_size=20)
>>> agent = GreedyAgent("player_0", env.grid_size)
>>> action = agent.act(env)
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np

from paperio_env import (
    AGENT_IDX,
    AGENT_IDS,
    DR,
    NEUTRAL,
    TERRITORY,
    TRAIL,
)

if TYPE_CHECKING:
    from paper_io_env import PaperIOEnv


class GreedyAgent:
    """
    Rule-based greedy agent that can act as a training opponent or baseline.

    Strategy
    --------
    * While it has an active trail (is outside its territory):
        BFS toward the nearest cell of its own territory, avoiding
        its own trail.
    * While safely inside its territory:
        BFS toward the nearest neutral cell to expand territory.
    * Falls back to a random safe move if BFS finds no path.

    Usage
    -----
    >>> agent = GreedyAgent("player_0", env.grid_size)
    >>> action = agent.act(env)
    """

    def __init__(self, agent_id: str, grid_size: int) -> None:
        self.agent_id = agent_id
        self.grid_size = grid_size
        self._rng = np.random.default_rng()

    def act(self, env: PaperIOEnv) -> int:
        """Choose and return an action (0–3) given the current env state."""
        if self.agent_id not in env.possible_agents:
            return int(self._rng.integers(4))

        idx = AGENT_IDX[self.agent_id]
        r, c = env.pos[self.agent_id]

        # Only avoid the agent's own trail — stepping on it is self-elimination.
        # The opponent's trail is NOT dangerous; stepping on it kills the opponent.
        danger = {TRAIL[idx]}

        opp_trail = TRAIL[1 - idx]

        if env.has_trail[self.agent_id]:
            # Head back to own territory as quickly as possible.
            # Also treat adjacent opponent-trail cells as valid targets so the
            # agent will cut the opponent's trail if it's directly on the way.
            target = {TERRITORY[idx], opp_trail}
        else:
            # Venture out; prioritise cutting the opponent's trail over neutral.
            target = {opp_trail, NEUTRAL}

        action = self._bfs(env.grid, r, c, self.grid_size, target, danger)
        if action is None:
            action = self._safe_random(env.grid, r, c, self.grid_size, danger)
        return action if action is not None else int(self._rng.integers(4))

    @staticmethod
    def _bfs(
        grid: np.ndarray,
        sr: int,
        sc: int,
        G: int,
        targets: set[int],
        blocked: set[int],
    ) -> Optional[int]:
        """
        BFS from (sr, sc) to the nearest cell whose value is in `targets`.
        Cells whose value is in `blocked` are not traversed.
        Returns the first action on the shortest path, or None if unreachable.
        """
        visited = np.zeros((G, G), dtype=bool)
        visited[sr, sc] = True
        # queue items: (row, col, first_action_taken)
        q: deque[tuple[int, int, int]] = deque()

        for action, (dr, dc) in DR.items():
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < G and 0 <= nc < G and not visited[nr, nc]:
                cell = int(grid[nr, nc])
                if cell in blocked:
                    continue
                visited[nr, nc] = True
                if cell in targets:
                    return action
                q.append((nr, nc, action))

        while q:
            r, c, first_action = q.popleft()
            for _, (dr, dc) in DR.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < G and 0 <= nc < G and not visited[nr, nc]:
                    cell = int(grid[nr, nc])
                    if cell in blocked:
                        continue
                    visited[nr, nc] = True
                    if cell in targets:
                        return first_action
                    q.append((nr, nc, first_action))

        return None

    @staticmethod
    def _safe_random(
        grid: np.ndarray,
        r: int,
        c: int,
        G: int,
        blocked: set[int],
    ) -> Optional[int]:
        """Return a random action that avoids walls and blocked cells."""
        safe = [
            a
            for a, (dr, dc) in DR.items()
            if 0 <= r + dr < G
            and 0 <= c + dc < G
            and int(grid[r + dr, c + dc]) not in blocked
        ]
        if safe:
            return safe[np.random.randint(len(safe))]
        return None
