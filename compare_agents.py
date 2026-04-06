import os
import json
import csv
import argparse
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from paperio_env import PaperIOEnv
from greedy_agent import GreedyAgent
from random_agent import RandomAgent


class SB3AgentWrapper:
    def __init__(self, model_path, agent_id, name):
        self.model = PPO.load(model_path)
        self.agent_id = agent_id
        self.name = name

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def _make_random(agent_id, grid_size):
    agent = RandomAgent(agent_id, grid_size)
    agent.name = "random"
    return agent


def _make_greedy(agent_id, grid_size):
    agent = GreedyAgent(agent_id, grid_size)
    agent.name = "greedy"
    return agent


def save_csv(results, path):
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_agent_action(agent, env, obs_dict, agent_id):
    if agent_id not in env.agents:
        return None

    if isinstance(agent, (GreedyAgent, RandomAgent)):
        return int(agent.act(env))

    obs = obs_dict[agent_id]
    obs = np.transpose(obs, (2, 0, 1))
    return int(agent.act(env, obs))


def run_episode(env, agent_a, agent_b, seed=None, render=False):
    obs, info = env.reset(seed=seed)

    agent_map = {
        "player_0": agent_a,
        "player_1": agent_b,
    }

    total_rewards = {
        "player_0": 0.0,
        "player_1": 0.0,
    }

    steps = 0
    last_infos = {a: {} for a in env.possible_agents}

    while env.agents:
        actions = {}

        for agent_id in env.agents:
            action = get_agent_action(agent_map[agent_id], env, obs, agent_id)
            if action is not None:
                actions[agent_id] = action

        obs, rew, term, trunc, infos = env.step(actions)

        for a in rew:
            total_rewards[a] += rew[a]

        last_infos = infos
        steps += 1

        if render:
            env.render()

    territory = env.territory_counts()

    p0_outcome = last_infos.get("player_0", {}).get("outcome", "draw")
    p1_outcome = last_infos.get("player_1", {}).get("outcome", "draw")

    if p0_outcome == "win":
        winner = "player_0"
    elif p1_outcome == "win":
        winner = "player_1"
    else:
        winner = "draw"

    result = {
        "episode": None,
        "matchup_requested": None,
        "steps_survived": steps,
        "player_0_agent": getattr(agent_a, "name", agent_a.__class__.__name__),
        "player_1_agent": getattr(agent_b, "name", agent_b.__class__.__name__),
        "player_0_reward": total_rewards["player_0"],
        "player_1_reward": total_rewards["player_1"],
        "player_0_territory": territory["player_0"],
        "player_1_territory": territory["player_1"],
        "player_0_outcome": p0_outcome,
        "player_1_outcome": p1_outcome,
        "winner": winner,
    }

    return result


def summarize_results(results):
    by_agent = defaultdict(lambda: {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_reward": 0.0,
        "total_territory": 0.0,
        "total_steps": 0.0,
    })

    for r in results:
        p0 = r["player_0_agent"]
        p1 = r["player_1_agent"]

        by_agent[p0]["games"] += 1
        by_agent[p1]["games"] += 1

        by_agent[p0]["total_reward"] += r["player_0_reward"]
        by_agent[p1]["total_reward"] += r["player_1_reward"]

        by_agent[p0]["total_territory"] += r["player_0_territory"]
        by_agent[p1]["total_territory"] += r["player_1_territory"]

        by_agent[p0]["total_steps"] += r["steps_survived"]
        by_agent[p1]["total_steps"] += r["steps_survived"]

        if r["player_0_outcome"] == "win":
            by_agent[p0]["wins"] += 1
            by_agent[p1]["losses"] += 1
        elif r["player_1_outcome"] == "win":
            by_agent[p1]["wins"] += 1
            by_agent[p0]["losses"] += 1
        else:
            by_agent[p0]["draws"] += 1
            by_agent[p1]["draws"] += 1

    overall_summary = {}
    for agent_name, stats in by_agent.items():
        games = stats["games"]
        overall_summary[agent_name] = {
            "games": games,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "win_rate": stats["wins"] / games if games else 0.0,
            "avg_reward": stats["total_reward"] / games if games else 0.0,
            "avg_territory": stats["total_territory"] / games if games else 0.0,
            "avg_steps_survived": stats["total_steps"] / games if games else 0.0,
        }

    return overall_summary


def compute_matchup_summary(results):
    summary = defaultdict(lambda: {
        "games": 0,
        "agent_1_wins": 0,
        "agent_2_wins": 0,
        "draws": 0,
        "agent_1_rewards": [],
        "agent_2_rewards": [],
        "agent_1_territory": [],
        "agent_2_territory": [],
        "steps": [],
    })

    for r in results:
        key = f"{r['player_0_agent']}_vs_{r['player_1_agent']}"

        s = summary[key]
        s["games"] += 1
        s["agent_1_rewards"].append(r["player_0_reward"])
        s["agent_2_rewards"].append(r["player_1_reward"])
        s["agent_1_territory"].append(r["player_0_territory"])
        s["agent_2_territory"].append(r["player_1_territory"])
        s["steps"].append(r["steps_survived"])

        if r["winner"] == "player_0":
            s["agent_1_wins"] += 1
        elif r["winner"] == "player_1":
            s["agent_2_wins"] += 1
        else:
            s["draws"] += 1

    final = {}
    for k, v in summary.items():
        g = v["games"]
        final[k] = {
            "games": g,
            "agent_1_wins": v["agent_1_wins"],
            "agent_2_wins": v["agent_2_wins"],
            "draws": v["draws"],
            "agent_1_win_rate": v["agent_1_wins"] / g if g else 0.0,
            "agent_2_win_rate": v["agent_2_wins"] / g if g else 0.0,
            "avg_steps": float(np.mean(v["steps"])) if v["steps"] else 0.0,
            "agent_1_avg_reward": float(np.mean(v["agent_1_rewards"])) if v["agent_1_rewards"] else 0.0,
            "agent_2_avg_reward": float(np.mean(v["agent_2_rewards"])) if v["agent_2_rewards"] else 0.0,
            "agent_1_avg_territory": float(np.mean(v["agent_1_territory"])) if v["agent_1_territory"] else 0.0,
            "agent_2_avg_territory": float(np.mean(v["agent_2_territory"])) if v["agent_2_territory"] else 0.0,
        }

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--reward-mode", type=str, default="mixed", choices=["sparse", "dense", "mixed"])
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--terminal-scale", type=float, default=1.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    os.makedirs("evaluation_results", exist_ok=True)

    def make_env():
        return PaperIOEnv(
            grid_size=args.grid_size,
            reward_mode=args.reward_mode,
            render_mode="human" if args.render else None,
            max_steps=args.max_steps,
            terminal_scale=args.terminal_scale,
        )

    agents = {
        "random": lambda agent_id: _make_random(agent_id, args.grid_size),
        "greedy": lambda agent_id: _make_greedy(agent_id, args.grid_size),
        "ppo_vs_greedy": lambda agent_id: SB3AgentWrapper(
            "ppo_selfplay_model_on_greedy.zip", agent_id, "ppo_vs_greedy"
        ),
        "ppo_selfplay": lambda agent_id: SB3AgentWrapper(
            "ppo_selfplay_model_selfplay_8m.zip", agent_id, "ppo_selfplay"
        ),
    }

    matchups = [
        ("random", "random"),
        ("greedy", "random"),
        ("ppo_vs_greedy", "random"),
        ("ppo_selfplay", "random"),
        ("greedy", "greedy"),
        ("ppo_vs_greedy", "greedy"),
        ("ppo_selfplay", "greedy"),
        ("ppo_vs_greedy", "ppo_selfplay"),
        ("ppo_vs_greedy", "ppo_vs_greedy"),
        ("ppo_selfplay", "ppo_selfplay"),
    ]

    all_results = []
    episode_counter = 1

    for left_name, right_name in matchups:
        print(f"\nRunning matchup: {left_name} vs {right_name}")

        for ep in range(args.episodes):
            env = make_env()

            if ep % 2 == 0:
                player_0_agent = agents[left_name]("player_0")
                player_1_agent = agents[right_name]("player_1")
            else:
                player_0_agent = agents[right_name]("player_0")
                player_1_agent = agents[left_name]("player_1")

            result = run_episode(
                env=env,
                agent_a=player_0_agent,
                agent_b=player_1_agent,
                seed=args.seed + episode_counter,
                render=args.render,
            )

            result["episode"] = episode_counter
            result["matchup_requested"] = f"{left_name}_vs_{right_name}"

            all_results.append(result)

            print(
                f"Episode {episode_counter:>4} | "
                f"{result['player_0_agent']} vs {result['player_1_agent']} | "
                f"winner={result['winner']} | "
                f"steps={result['steps_survived']} | "
                f"territory=({result['player_0_territory']}, {result['player_1_territory']}) | "
                f"reward=({result['player_0_reward']:.2f}, {result['player_1_reward']:.2f})"
            )

            env.close()
            episode_counter += 1

    save_csv(all_results, "evaluation_results/episode_results.csv")

    overall_summary = summarize_results(all_results)
    matchup_summary = compute_matchup_summary(all_results)

    save_json(overall_summary, "evaluation_results/overall_summary.json")
    save_json(matchup_summary, "evaluation_results/matchup_summary.json")

    print("\n=== OVERALL SUMMARY ===")
    for agent_name, stats in overall_summary.items():
        print(
            f"{agent_name}: "
            f"games={stats['games']}, "
            f"wins={stats['wins']}, "
            f"losses={stats['losses']}, "
            f"draws={stats['draws']}, "
            f"win_rate={stats['win_rate']:.3f}, "
            f"avg_reward={stats['avg_reward']:.3f}, "
            f"avg_territory={stats['avg_territory']:.3f}, "
            f"avg_steps={stats['avg_steps_survived']:.3f}"
        )

    print("\nSaved files:")
    print("  evaluation_results/episode_results.csv")
    print("  evaluation_results/overall_summary.json")
    print("  evaluation_results/matchup_summary.json")


if __name__ == "__main__":
    main()