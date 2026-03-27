import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from paperio_env import PaperIOEnv
from greedy_agent import GreedyAgent

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        # Fully connected layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs))


class SingleAgentWrapper(gym.Env):
    def __init__(self, grid_size=20, reward_mode="mixed"):
        super().__init__()

        self.env = PaperIOEnv(grid_size=grid_size, reward_mode=reward_mode)

        self.agent_id = "player_0"
        self.opponent = GreedyAgent("player_1", grid_size)

        orig_space = self.env.observation_space(self.agent_id)

        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(orig_space.shape[2], orig_space.shape[0], orig_space.shape[1]),
            dtype=np.float32
        )

        self.action_space = self.env.action_space(self.agent_id)

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)
        # Convert to channel-first format for CNN
        obs = np.transpose(obs[self.agent_id], (2, 0, 1))
        return obs, {}

    def step(self, action):
         # Opponent uses fixed greedy policy
        opp_action = self.opponent.act(self.env)

        obs, rewards, term, trunc, _ = self.env.step({
            self.agent_id: action,
            "player_1": opp_action
        })

        done = term[self.agent_id] or trunc[self.agent_id]
        obs = np.transpose(obs[self.agent_id], (2, 0, 1))

        return obs, rewards[self.agent_id], done, False, {}


from stable_baselines3.common.callbacks import BaseCallback

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        # Extract episode reward from Monitor wrapper
        for info in self.locals["infos"]:
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True



def make_env():
    return Monitor(SingleAgentWrapper())

env = make_vec_env(make_env, n_envs=4)

# PPO Model Setup
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,
    policy_kwargs=dict(
        normalize_images=False,
        features_extractor_class=CustomCNN,
    )
)

callback = RewardLogger()

model.learn(total_timesteps=100_000, callback=callback)

model.save("ppo_model")


# Reward Plot
def moving_avg(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure()
plt.plot(moving_avg(callback.rewards))
plt.title("Training Reward Curve")
plt.savefig("reward_curve.png")


# Evaluation
def evaluate(model, episodes=50):
    wins = 0
    territories = []
    survival = []

    env = PaperIOEnv()
    agent = GreedyAgent("player_1", env.grid_size)

    for _ in range(episodes):
        obs, _ = env.reset()
        steps = 0

        while env.agents:
            obs_agent = np.transpose(obs["player_0"], (2, 0, 1))
            action, _ = model.predict(obs_agent, deterministic=True)

            obs, rewards, term, trunc, _ = env.step({
                "player_0": action,
                "player_1": agent.act(env)
            })

            steps += 1

        if env.alive["player_0"]:
            wins += 1

        territories.append(np.sum(env.grid == 1))
        survival.append(steps)

    return wins/episodes, np.mean(territories), np.mean(survival)


win, terr, surv = evaluate(model)

print("Win Rate:", win)
print("Territory:", terr)
print("Survival:", surv)


# Plot metrics
plt.figure()
plt.bar(["Win", "Territory", "Survival"], [win, terr, surv])
plt.title("Evaluation Metrics")
plt.savefig("evaluation.png")

plt.show()