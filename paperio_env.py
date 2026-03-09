# File containing implementation of paperio environment

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from collections import deque
from typing import Optional
import functools

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

    # Number of pixels per grid cell in the Pygame window
    _CELL_PX = 22

    # Colours
    _PALETTE: dict[int, tuple[int, int, int]] = {
        NEUTRAL:      (210, 210, 210),
        P0_TERRITORY: (100, 149, 237), #light blue
        P1_TERRITORY: (255,  99,  71), # light red
        P0_TRAIL:     ( 20,  60, 220), # deep blue
        P1_TRAIL:     (200,  30,  30), # deep red
    }
    _HEAD_COLOR: dict[str, tuple[int, int, int]] = {
        "player_0": (  0,   0, 180),
        "player_1": (180,   0,   0),
    }

    def __init__(self,
        grid_size: int = 20,
        reward_mode: str = "mixed",
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
    ):
        super().__init__()
        assert reward_mode in ("sparse", "dense", "mixed"), (
            "reward_mode must be 'sparse', 'dense', or 'mixed'"
        )
        assert render_mode in (None, "human", "rgb_array")

        self.grid_size = grid_size # probably keep at 20
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.possible_agents: list[str] = AGENT_IDS[:]
        self.agents: list[str] = []

        _obs_shape = (grid_size, grid_size, 7)
        self.observation_spaces = {
            a: spaces.Box(0.0, 1.0, shape=_obs_shape, dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(4) for a in self.possible_agents
        }

        # rendering
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

        # state info
        self.grid: np.ndarray = np.empty(0)
        self.pos: dict[str, list[int]] = {}
        self.direction: dict[str, int] = {}
        self.alive: dict[str, bool] = {}
        self.has_trail: dict[str, bool] = {}
        self.step_count: int = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        self.pos = {}
        self.direction = {}
        self.alive = {}
        self.has_trail = {}

        # Each player spawns in an opposite corner with a small territory block
        init_half = 2
        margin = init_half + 1

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

            # Initial territory
            for dr in range(-init_half, init_half + 1):
                for dc in range(-init_half, init_half + 1):
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        self.grid[nr, nc] = tv

        observations = {a: self._obs(a) for a in self.agents}
        info: dict[str, dict] = {a: {} for a in self.agents}
        return observations, info

    def _obs(self, agent: str) -> np.ndarray:
        # return array representing grid_state
        idx = AGENT_IDX[agent]
        opp = AGENT_IDS[1 - idx]
        G = self.grid_size

        obs = np.zeros((G, G, 7), dtype=np.float32)
        obs[:, :, 0] = (self.grid == TERRITORY[idx]).astype(np.float32)
        obs[:, :, 1] = (self.grid == TERRITORY[1 - idx]).astype(np.float32)
        obs[:, :, 2] = (self.grid == TRAIL[idx]).astype(np.float32)
        obs[:, :, 3] = (self.grid == TRAIL[1 - idx]).astype(np.float32)

        r, c = self.pos[agent]
        obs[r, c, 4] = 1.0

        if self.alive.get(opp, False):
            or_, oc = self.pos[opp]
            obs[or_, oc, 5] = 1.0

        obs[:, :, 6] = self.direction[agent] / 3.0
        return obs

    def step(self,
             actions: dict[str, int]
    ):
        self.step_count += 1

        prev_terr = {
            a: int(np.sum(self.grid == TERRITORY[AGENT_IDX[a]]))
            for a in self.agents
        }

        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}
        infos: dict[str, dict] = {a: {} for a in self.agents}

        moved = self._apply_actions(actions)

        # Collect trail-collision kills
        killed = self._detect_trail_kills(moved)
        self._apply_kills(killed, terminations)

        # Trail extension / territory closure; may produce enclosure kills
        enclosure_kills = self._update_trails_and_territory(killed)
        new_kills = enclosure_kills - killed
        self._apply_kills(new_kills, terminations)
        killed |= new_kills

        rewards = {
            a: self._compute_reward(a, prev_terr, killed)
            for a in self.agents
        }

        if self.step_count >= self.max_steps:
            for a in self.agents:
                truncations[a] = True

        _zero_obs = np.zeros((self.grid_size, self.grid_size, 7), dtype=np.float32)
        observations = {
            a: _zero_obs if (terminations[a] or truncations[a]) else self._obs(a)
            for a in self.agents
        }

        self.agents = [
            a for a in self.agents
            if not (terminations[a] or truncations[a])
        ]

        return observations, rewards, terminations, truncations, infos

    # update directions and move agents
    def _apply_actions(self, actions: dict[str, int]):
        moved: set[str] = set()
        for agent in self.agents:
            action = int(actions.get(agent, self.direction[agent]))
            self.direction[agent] = action
            r, c = self.pos[agent]
            dr, dc = DR[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                self.pos[agent] = [nr, nc]
                moved.add(agent)
        return moved

    # check for trail collision (which kills agent)s
    def _detect_trail_kills(self, moved: set[str]):
        kills: set[str] = set()
        for agent in moved:
            idx = AGENT_IDX[agent]
            r, c = self.pos[agent]
            cell = int(self.grid[r, c])
            if cell == TRAIL[1 - idx]:    # stepped on opponent's trail
                kills.add(AGENT_IDS[1 - idx])
            elif cell == TRAIL[idx]:       # stepped on own trail
                kills.add(agent)
        return kills

    # erase killed agent trail, end game if kill occurs
    def _apply_kills(self, killed: set[str], terminations: dict[str, bool]):
        for agent in killed:
            self.alive[agent] = False
            idx = AGENT_IDX[agent]
            self.grid[self.grid == TRAIL[idx]] = NEUTRAL
            self.has_trail[agent] = False
        if killed:
            for a in self.agents:
                terminations[a] = True

    
    # close or extend trail, eliminates opponents contained within 
    def _update_trails_and_territory(self, killed: set[str]):
        enclosure_kills: set[str] = set()
        for agent in self.agents:
            if agent in killed:
                continue
            idx = AGENT_IDX[agent]
            r, c = self.pos[agent]
            cell = int(self.grid[r, c])

            if cell == TERRITORY[idx]:
                # Returned to own territory so close the trail
                if self.has_trail[agent]:
                    if self._flood_claim(idx):
                        enclosure_kills.add(AGENT_IDS[1 - idx])
                    self.has_trail[agent] = False
            else:
                # Outside territory — leave a trail
                if cell != TRAIL[idx]:
                    self.grid[r, c] = TRAIL[idx]
                self.has_trail[agent] = True
        return enclosure_kills

    def _compute_reward(self,
        agent: str,
        prev_terr: dict[str, int],
        killed: set[str],
    ):
        # calculates reward for agent
        idx = AGENT_IDX[agent]
        opp = AGENT_IDS[1 - idx]
        won  = agent not in killed and opp in killed
        lost = agent in killed

        if self.reward_mode == "sparse":
            if won:
                return 1.0
            if lost:
                return -1.0
            return 0.0

        if self.reward_mode == "dense":
            grid_area = self.grid_size ** 2
            curr     = int(np.sum(self.grid == TERRITORY[idx]))
            opp_curr = int(np.sum(self.grid == TERRITORY[1 - idx]))
            delta     = (curr     - prev_terr[agent]) / grid_area
            opp_delta = (opp_curr - prev_terr[opp])   / grid_area
            reward = delta - 0.5 * opp_delta
            if won:
                reward += 1.0
            elif lost:
                reward -= 1.0
            return reward

        # "mixed"
        curr     = int(np.sum(self.grid == TERRITORY[idx]))
        delta    = curr - prev_terr[agent]  # / grid_area
        survival = 0.0 if lost else 0.001
        return delta + survival
    
    def _flood_claim(self, agent_idx: int) -> bool:
        # When an agent closes its trail loop, determines which cells are
        # captured. Returns True if the opponent's trail was enclosed (opponent dies).
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

    def render(self):
    # pygame rendering
        if self.render_mode is None:
            return None

        px = self._CELL_PX
        W = H = self.grid_size * px

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("Paper.io RL")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((W, H))

        self._screen.fill((30, 30, 30))

        # Grid cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = self._PALETTE.get(int(self.grid[r, c]), (210, 210, 210))
                pygame.draw.rect(
                    self._screen, color,
                    (c * px, r * px, px - 1, px - 1),
                )

        # Agent heads and direction indicator
        for agent in self.possible_agents:
            if not self.alive.get(agent, False):
                continue
            r, c = self.pos[agent]
            cx, cy = c * px + px // 2, r * px + px // 2
            head_color = self._HEAD_COLOR[agent]
            pygame.draw.circle(self._screen, head_color, (cx, cy), px // 2 - 1)
            # Small arrow in direction of travel
            dr, dc = DR[self.direction[agent]]
            ax, ay = cx + dc * (px // 3), cy + dr * (px // 3)
            pygame.draw.line(self._screen, (255, 255, 255), (cx, cy), (ax, ay), 2)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(10)
            return None
        else:
            # rgb_array: (H, W, 3)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)),
                axes=(1, 0, 2),
            )
    
    def close(self):
        # close pygame
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None


if __name__ == "__main__":
    import time
    from greedy_agent import GreedyAgent

    env = PaperIOEnv(
        grid_size=20,
        reward_mode="mixed",
        render_mode="human",
        max_steps=300,
    )
    greedy_agents = [GreedyAgent(aid, env.grid_size) for aid in env.possible_agents]

    total_eps = 3
    for ep in range(total_eps):
        obs, info = env.reset(seed=ep)
        ep_rewards = {a: 0.0 for a in env.possible_agents}
        steps = 0

        while env.agents:
            actions = {
                ga.agent_id: ga.act(env)
                for ga in greedy_agents
                if ga.agent_id in env.agents
            }
            obs, rew, term, trunc, info = env.step(actions)
            for a, r in rew.items():
                ep_rewards[a] += r
            env.render()
            steps += 1

        terr = env.territory_counts()
        print(
            f"  Episode {ep + 1:>2d} | steps={steps:>4d} | "
            f"rewards={ep_rewards} | territory={terr}"
        )

    env.close()
    print("Done.")
