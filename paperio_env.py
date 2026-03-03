# File containing implementation of paperio environment

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from collections import deque

# Directions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DR = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Cell states
NEUTRAL = 0
P0_TERRITORY = 1
P1_TERRITORY = 2
P0_TRAIL = 3
P1_TRAIL = 4

TERRITORY  = [P0_TERRITORY, P1_TERRITORY]
TRAIL = [P0_TRAIL, P1_TRAIL]

# Agents
AGENT_IDS = ["player_0", "player_1"]
AGENT_IDX = {"player_0": 0, "player_1": 1}

class PaperIOEnv(ParallelEnv):
    
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, grid_size=20, view_size=11, reward_type=1, render_mode = None, max_steps=500):
        super().__init__()

        self.grid_size = grid_size
        self.view_size = view_size
        self.reward_mode = reward_type
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.possible_agents: list[str] = AGENT_IDS[:]
        self.agents: list[str] = []

        # observations for agent
        _obs_shape = (view_size, view_size, 5)
        self.observation_spaces = {
            a: spaces.Box(0.0, 1.0, shape=_obs_shape, dtype=np.float32)
            for a in self.possible_agents
        }

        # actions for agent
        self.action_spaces = {
            a: spaces.Discrete(4) for a in self.possible_agents
        }

        # runtime info
        self.grid: np.ndarray = np.empty(0)
        self.pos: dict[str, list[int]] = {}
        self.direction: dict[str, int] = {}
        self.alive: dict[str, bool] = {}
        self.has_trail: dict[str, bool] = {}
        self.step_count = 0

    def reset(
        self,
        seed = None,
        options = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        # resetting state info
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        self.pos = {}
        self.direction = {}
        self.alive = {}
        self.has_trail = {}

        # Each player spawns in an opposite corner with a small territory block
        init_half = 2
        margin = init_half + 1 # distance from grid edge to spawn centre

        spawn_cfg = {
            "player_0": (margin, margin, RIGHT, P0_TERRITORY),
            "player_1": (self.grid_size - margin - 1,
                         self.grid_size - margin - 1,
                         LEFT, P1_TERRITORY),
        }

        for agent, (r0, c0, init_dir, tv) in spawn_cfg.items():
            self.pos[agent] = [r0, c0]
            self.direction[agent] = init_dir
            self.alive[agent] = True
            self.has_trail[agent] = False

            # Stamp initial territory block
            for dr in range(-init_half, init_half + 1):
                for dc in range(-init_half, init_half + 1):
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        self.grid[nr, nc] = tv

        observations = {a: self._obs(a) for a in self.agents}
        info = {a: {} for a in self.agents}
        return observations, info

    def _obs(self, agent):

        idx = AGENT_IDX[agent]
        r, c = self.pos[agent]
        L = self.local_view_size
        half = L // 2
        G = self.grid_size

        # Pad with neutral so agents see out=of-bounds area as neutral
        padded = np.full((G + 2 * half, G + 2 * half), NEUTRAL, dtype=np.int8)
        padded[half: half + G, half: half + G] = self.grid

        # generate observation
        local = padded[r: r + L, c: c + L]
        obs = np.zeros((L, L, 5), dtype=np.float32)
        obs[:, :, 0] = (local == TERRITORY[idx]).astype(np.float32)
        obs[:, :, 1] = (local == TERRITORY[1 - idx]).astype(np.float32)
        obs[:, :, 2] = (local == TRAIL[idx]).astype(np.float32)
        obs[:, :, 3] = (local == TRAIL[1 - idx]).astype(np.float32)
        obs[:, :, 4] = self.direction[agent] / 3.0
        return obs

    def step(self, actions):

        self.step_count += 1

        prev_terr = {
            a: int(np.sum(self.grid == TERRITORY[AGENT_IDX[a]]))
            for a in self.agents
        }

        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # New agent positions
        new_pos: dict[str, list[int]] = {}
        for agent in self.agents:
            action = int(actions.get(agent, self.direction[agent]))
            self.direction[agent] = action
            r, c = self.pos[agent]
            dr, dc = DR[action]
            new_pos[agent] = [r + dr, c + dc]

        # Agent moves
        # check for trail collisions against the cell they already occupy
        killed: set[str] = set()
        moved: set[str] = set()
        for agent in self.agents:
            nr, nc = new_pos[agent]
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                self.pos[agent] = new_pos[agent]
                moved.add(agent)
            # otherwise agent stays put; skip trail/collision check for this step

        # Detect trail collision for agents that moved
        # Collect all kills before applying any so both can detect each other
        trail_kills: set[str] = set()
        for agent in moved:
            idx = AGENT_IDX[agent]
            r, c = self.pos[agent]
            cell = int(self.grid[r, c])
            if cell == TRAIL[1 - idx]: # stepped on opponent's trail
                trail_kills.add(AGENT_IDS[1 - idx])
            elif cell == TRAIL[idx]: # stepped on own trail
                trail_kills.add(agent)
        killed |= trail_kills

        # Death handling
        if killed:
            for agent in killed:
                self.alive[agent] = False
                idx = AGENT_IDX[agent]
                self.grid[self.grid == TRAIL[idx]] = NEUTRAL
                self.has_trail[agent] = False
            # Any death ends episode
            for a in self.agents:
                terminations[a] = True

        # Update trails and owned cells
        enclosure_kills: set[str] = set()
        for agent in self.agents:
            if agent in killed:
                continue
            idx = AGENT_IDX[agent]
            r, c = self.pos[agent]
            cell = int(self.grid[r, c])
            tv = TERRITORY[idx]
            tr = TRAIL[idx]

            if cell == tv:
                # Agent returned to its own territory → close trail
                if self.has_trail[agent]:
                    opp_caught = self._flood_claim(idx)
                    if opp_caught:
                        enclosure_kills.add(AGENT_IDS[1 - idx])
                    self.has_trail[agent] = False
            else:
                # Agent is outside territory → leave trail
                if cell != tr:
                    self.grid[r, c] = tr
                self.has_trail[agent] = True

        # Handle opponents killed by enclosure
        for agent in enclosure_kills:
            if agent not in killed:
                killed.add(agent)
                self.alive[agent] = False
                idx = AGENT_IDX[agent]
                self.grid[self.grid == TRAIL[idx]] = NEUTRAL
                self.has_trail[agent] = False
                for a in self.agents:
                    terminations[a] = True

        # Calculate reward
        grid_area = self.grid_size ** 2
        for agent in self.agents:
            idx = AGENT_IDX[agent]
            opp = AGENT_IDS[1 - idx]
            won = agent not in killed and opp in killed
            lost = agent in killed

            if self.reward_mode == "sparse":
                if won:
                    rewards[agent] = 1.0
                elif lost:
                    rewards[agent] = -1.0

            elif self.reward_mode == "dense":
                curr = int(np.sum(self.grid == TERRITORY[idx]))
                opp_curr = int(np.sum(self.grid == TERRITORY[1 - idx]))
                delta = (curr - prev_terr[agent]) / grid_area
                opp_delta = (opp_curr - prev_terr[opp]) / grid_area
                rewards[agent] = delta - 0.5 * opp_delta
                if won:
                    rewards[agent] += 1.0
                elif lost:
                    rewards[agent] -= 1.0

            else:  # "mixed"
                curr = int(np.sum(self.grid == TERRITORY[idx]))
                delta = (curr - prev_terr[agent]) / grid_area
                survival = 0.0 if lost else 0.005
                rewards[agent] = delta + survival
                if won:
                    rewards[agent] += 1.0
                elif lost:
                    rewards[agent] -= 1.0
        # truncation
        if self.step_count >= self.max_steps:
            for a in self.agents:
                truncations[a] = True

        # return observations
        _zero_obs = np.zeros(
            (self.local_view_size, self.local_view_size, 5), dtype=np.float32
        )
        observations: dict[str, np.ndarray] = {}
        for a in self.agents:
            observations[a] = (
                _zero_obs if (terminations[a] or truncations[a])
                else self._obs(a)
            )

        self.agents = [
            a for a in self.agents
            if not (terminations[a] or truncations[a])
        ]

        return observations, rewards, terminations, truncations, infos
    
    def _flood_claim(self, agent_idx: int) -> bool:
        # when an agent closes loop in trail
        # determines which cells are captured, and if opponent was killedns True if the opponent's trail was enclosed (opponent dies)
        tv  = TERRITORY[agent_idx]
        tr  = TRAIL[agent_idx]
        otr = TRAIL[1 - agent_idx]
        G   = self.grid_size

        # Step 1 – convert this agent's trail to territory
        self.grid[self.grid == tr] = tv

        # Step 2 – BFS from every border cell that is NOT self-territory
        visited = np.zeros((G, G), dtype=bool)
        q: deque[tuple[int, int]] = deque()

        def _seed_border(r: int, c: int) -> None:
            if self.grid[r, c] != tv and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))

        for i in range(G):
            _seed_border(0, i)
            _seed_border(G - 1, i)
            _seed_border(i, 0)
            _seed_border(i, G - 1)

        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < G and 0 <= nc < G
                    and not visited[nr, nc]
                    and self.grid[nr, nc] != tv
                ):
                    visited[nr, nc] = True
                    q.append((nr, nc))

        # claim all unreachable  non-self cells
        enclosed_mask = ~visited & (self.grid != tv)

        # Opponent is eliminated if their trail OR their position is enclosed
        opp_agent = AGENT_IDS[1 - agent_idx]
        opp_r, opp_c = self.pos[opp_agent]
        opp_eliminated = (
            bool(np.any(self.grid[enclosed_mask] == otr))
            or bool(enclosed_mask[opp_r, opp_c])
        )

        self.grid[enclosed_mask] = tv

        return opp_eliminated

    def __render__(self):
        if self.render_mode == None:
            return None
        # pygame stuff
    
    def __close__(self):
        # clsoe pygame
        return