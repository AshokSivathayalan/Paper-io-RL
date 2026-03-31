import copy
import os
import random

import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from paperio_env import PaperIOEnv, P0_TERRITORY
from greedy_agent import GreedyAgent


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # Stride-1 convolutions preserve spatial resolution so the policy
        # retains precise positional information (trail location, head position).
        # MaxPool is avoided for the same reason.
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))


# Maps an action produced from a 180°-rotated observation back to the real board.
# Rotating 180° flips both axes, so UP↔DOWN and LEFT↔RIGHT.
_ACTION_FLIP = {0: 1, 1: 0, 2: 3, 3: 2}

def _p1_obs(env: PaperIOEnv) -> np.ndarray:
    """Return player_1's observation rotated 180° into player_0's spatial frame.

    The full-board observation is not egocentric, so player_1 always appears in
    the bottom-right corner while the policy was only trained from the top-left.
    Rotating 180° maps player_1's corner to the top-left, making the spatial
    layout match what the policy saw during training.  The caller must flip the
    returned action with _ACTION_FLIP to convert it back to real board directions.
    """
    obs_hwc = env._obs("player_1").copy()                  # (H, W, C)
    obs_hwc = np.rot90(obs_hwc, k=2, axes=(0, 1)).copy()  # rotate spatial dims
    # Direction channels (6-9) encode (dr, dc) as (v+1)/2.
    # A 180° rotation negates both dr and dc: flip around 0.5.
    for ch in (6, 7, 8, 9):
        obs_hwc[:, :, ch] = 1.0 - obs_hwc[:, :, ch]
    return np.transpose(obs_hwc, (2, 0, 1))               # → (C, H, W)


class OpponentPool:
    """Pool of frozen policy snapshots used as self-play opponents.

    All SingleAgentWrapper instances in the VecEnv hold a reference to the
    same OpponentPool, so a single update() call affects every parallel env
    simultaneously without any inter-process communication.

    Maintains up to `max_pool` historical snapshots and samples uniformly
    from them each call, forcing the learner to generalise across different
    play styles rather than overfitting to a single frozen policy.

    Before the first snapshot is taken the greedy agent is used as a
    warm-start opponent, so the learner isn't playing against a random policy.
    """

    def __init__(self, grid_size: int = 20, max_pool: int = 10):
        self._policies: list = []
        self._max_pool = max_pool
        self._greedy = GreedyAgent("player_1", grid_size)

    def update(self, policy) -> None:
        """Deep-copy the current live policy and add it to the pool."""
        self._policies.append(copy.deepcopy(policy))
        if len(self._policies) > self._max_pool:
            self._policies.pop(0)

    def act(self, env: PaperIOEnv) -> int:
        if not self._policies:
            return self._greedy.act(env)
        policy = random.choice(self._policies)
        obs = _p1_obs(env)
        action, _ = policy.predict(obs, deterministic=False)
        return _ACTION_FLIP[int(action)]


class SingleAgentWrapper(gym.Env):
    def __init__(self, opponent_pool: OpponentPool, grid_size: int = 20, reward_mode: str = "mixed"):
        super().__init__()

        self.env = PaperIOEnv(grid_size=grid_size, reward_mode=reward_mode)
        self.agent_id = "player_0"
        self.opponent_pool = opponent_pool

        orig_space = self.env.observation_space(self.agent_id)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(orig_space.shape[2], orig_space.shape[0], orig_space.shape[1]),
            dtype=np.float32
        )
        self.action_space = self.env.action_space(self.agent_id)

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)
        obs = np.transpose(obs[self.agent_id], (2, 0, 1))
        return obs, {}

    def set_terminal_scale(self, scale: float) -> None:
        self.env.terminal_scale = scale

    def step(self, action):
        opp_action = self.opponent_pool.act(self.env)

        obs, rewards, term, trunc, infos = self.env.step({
            self.agent_id: action,
            "player_1": opp_action,
        })

        terminated = term[self.agent_id]
        truncated = trunc[self.agent_id]
        obs = np.transpose(obs[self.agent_id], (2, 0, 1))
        # Pass outcome through on terminal steps so the callback can track win rate
        info = infos[self.agent_id] if (terminated or truncated) else {}
        return obs, rewards[self.agent_id], terminated, truncated, info


class SelfPlayCallback(BaseCallback):
    """Snapshots the policy into OpponentPool every `update_freq` steps,
    and logs per-episode rewards from the Monitor wrapper.

    For the first `warmup_steps` timesteps the OpponentPool falls back to the
    greedy agent automatically (no snapshot is taken), giving the policy time
    to learn basic competence before self-play begins.
    """

    def __init__(self, opponent_pool: OpponentPool, update_freq: int = 10_000,
                 warmup_steps: int = 100_000):
        super().__init__()
        self.opponent_pool = opponent_pool
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self._last_update = 0
        self.episode_rewards: list = []
        self.episode_wins: list = []   # True/False per completed episode

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if "outcome" in info:
                self.episode_wins.append(info["outcome"] == "win")

        if self.num_timesteps >= self.warmup_steps:
            # Take a snapshot immediately when warmup ends so self-play begins
            # without a gap where the opponent is still the greedy agent.
            if self._last_update == 0:
                self.opponent_pool.update(self.model.policy)
                self._last_update = self.num_timesteps
                print(f"[SelfPlay] Warmup done at step {self.num_timesteps}, switching to self-play")

            elif self.num_timesteps - self._last_update >= self.update_freq:
                self.opponent_pool.update(self.model.policy)
                self._last_update = self.num_timesteps
                print(f"[SelfPlay] Opponent snapshot updated at step {self.num_timesteps}")

        return True


def moving_avg(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode="valid")


def evaluate(model, episodes: int = 50):
    wins = 0
    territories = []
    survival = []

    env = PaperIOEnv()
    agent = GreedyAgent("player_1", env.grid_size)

    for _ in range(episodes):
        obs, _ = env.reset()
        steps = 0
        last_infos = {}

        while env.agents:
            obs_agent = np.transpose(obs["player_0"], (2, 0, 1))
            action, _ = model.predict(obs_agent, deterministic=True)

            obs, _, term, trunc, infos = env.step({
                "player_0": int(action),
                "player_1": agent.act(env),
            })
            last_infos = infos
            steps += 1

        if last_infos.get("player_0", {}).get("outcome") == "win":
            wins += 1

        p0 = int(np.sum(env.grid == P0_TERRITORY))
        territories.append(p0)
        survival.append(steps)

    env.close()
    return wins / episodes, np.mean(territories), np.mean(survival)


class RewardCurriculumCallback(BaseCallback):
    """Linearly anneals terminal_scale from 0 → 1 over `anneal_steps` timesteps.

    During the early phase the win/loss bonus is suppressed, so the agent
    learns territory expansion before it is penalised for dying.
    """

    def __init__(self, anneal_steps: int = 200_000):
        super().__init__()
        self.anneal_steps = anneal_steps

    def _on_step(self) -> bool:
        scale = min(1.0, self.num_timesteps / self.anneal_steps)
        self.training_env.env_method("set_terminal_scale", scale)
        return True


class CheckpointCallback(BaseCallback):
    """Periodically saves the model and tracks the best by recent win rate.

    Saves a numbered checkpoint every `save_freq` timesteps into `save_dir`.
    Also maintains a `best_model` save based on the rolling win rate from
    a companion SelfPlayCallback.
    """

    def __init__(self, save_freq: int, save_dir: str = "checkpoints",
                 selfplay_cb: "SelfPlayCallback | None" = None,
                 win_window: int = 200):
        super().__init__()
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.selfplay_cb = selfplay_cb
        self.win_window = win_window
        self._last_save = 0
        self._best_winrate = -1.0

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            path = os.path.join(self.save_dir, f"model_{self.num_timesteps}")
            self.model.save(path)
            print(f"[Checkpoint] Saved {path}")

            # Track best model by win rate
            if self.selfplay_cb is not None:
                wins = self.selfplay_cb.episode_wins
                if len(wins) >= self.win_window:
                    recent = np.mean(wins[-self.win_window:])
                    if recent > self._best_winrate:
                        self._best_winrate = recent
                        best_path = os.path.join(self.save_dir, "best_model")
                        self.model.save(best_path)
                        print(f"[Checkpoint] New best model (win rate {recent:.2%})")

        return True


if __name__ == "__main__":
    GRID_SIZE = 20
    N_ENVS = 4
    TOTAL_TIMESTEPS = 4_000_000
    SELFPLAY_UPDATE_FREQ = 20_000
    WARMUP_STEPS = 150_000
    ANNEAL_STEPS = 400_000

    opponent_pool = OpponentPool(grid_size=GRID_SIZE)

    def make_env():
        return Monitor(SingleAgentWrapper(opponent_pool, grid_size=GRID_SIZE, reward_mode="mixed"))

    env = make_vec_env(make_env, n_envs=N_ENVS)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=lambda progress: 3e-4 * max(progress, 0.1),
        gamma=0.99,
        ent_coef=0.015,
        policy_kwargs=dict(
            normalize_images=False,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    )

    curriculum_cb = RewardCurriculumCallback(anneal_steps=ANNEAL_STEPS)
    selfplay_cb = SelfPlayCallback(opponent_pool, update_freq=SELFPLAY_UPDATE_FREQ,
                                   warmup_steps=WARMUP_STEPS)
    checkpoint_cb = CheckpointCallback(save_freq=200_000, save_dir="checkpoints",
                                       selfplay_cb=selfplay_cb)

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=[curriculum_cb, selfplay_cb, checkpoint_cb],
                progress_bar=True)
    model.save("ppo_selfplay_model")

    # Reward curve
    if selfplay_cb.episode_rewards:
        plt.figure()
        plt.plot(moving_avg(selfplay_cb.episode_rewards))
        plt.title("Training Reward Curve (Self-Play)")
        plt.xlabel("Episode")
        plt.ylabel("Reward (moving avg, w=10)")
        plt.savefig("reward_curve.png")

    # Win rate curve
    if selfplay_cb.episode_wins:
        plt.figure()
        plt.plot(moving_avg(selfplay_cb.episode_wins, w=100))
        plt.ylim(0, 1)
        plt.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
        plt.title("Win Rate vs Opponent (Self-Play)")
        plt.xlabel("Episode")
        plt.ylabel("Win rate (moving avg, w=100)")
        plt.savefig("win_rate.png")

    # Evaluation vs greedy
    win, terr, surv = evaluate(model)
    print(f"Win Rate:  {win:.2%}")
    print(f"Territory: {terr:.1f}")
    print(f"Survival:  {surv:.1f} steps")

    plt.figure()
    plt.bar(["Win Rate", "Territory", "Survival"], [win, terr, surv])
    plt.title("Evaluation Metrics (vs Greedy)")
    plt.savefig("evaluation.png")

    plt.show()
