"""
watch.py — watch the trained PPO model play against the greedy agent.

Usage
-----
    python watch.py                        # uses ppo_selfplay_model.zip
    python watch.py path/to/model.zip      # custom model path

Controls
--------
    +/-        speed up / slow down
    R          restart
    ESC / Q    quit
"""

import sys
import argparse

import numpy as np
import pygame
from stable_baselines3 import PPO

from paperio_env import PaperIOEnv
from greedy_agent import GreedyAgent

# ── Layout ────────────────────────────────────────────────────────────────────
HUD_H   = 56
MIN_FPS =  2
MAX_FPS = 20
DEF_FPS =  6

# ── Colours ───────────────────────────────────────────────────────────────────
COL_P0  = (100, 149, 237)   # blue (PPO)
COL_P1  = (255,  99,  71)   # red  (greedy)
COL_HUD = ( 22,  22,  22)

# Flip UP↔DOWN and LEFT↔RIGHT to convert an action from a 180°-rotated
# observation back to real board directions (see train_ppo.py for explanation).
_ACTION_FLIP = {0: 1, 1: 0, 2: 3, 3: 2}

KEY_QUIT    = {pygame.K_ESCAPE, pygame.K_q}
KEY_RESTART = {pygame.K_r}
KEY_FASTER  = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}
KEY_SLOWER  = {pygame.K_MINUS, pygame.K_KP_MINUS}


def draw_hud(screen, font, env, fps, grid_px, p1_label_str="PPO"):
    W = screen.get_width()
    screen.fill(COL_HUD, (0, grid_px, W, HUD_H))

    terr  = {a: int(np.sum(env.grid == (i + 1))) for i, a in enumerate(env.possible_agents)}
    total = env.grid_size ** 2

    # Territory bar
    BAR_Y = grid_px + HUD_H - 10
    BAR_H = 6
    bar_w = W - 4
    p0_w  = int(bar_w * terr["player_0"] / total)
    p1_w  = int(bar_w * terr["player_1"] / total)

    pygame.draw.rect(screen, (55, 55, 55), (2, BAR_Y, bar_w, BAR_H))
    pygame.draw.rect(screen, COL_P0,       (2, BAR_Y, p0_w,  BAR_H))
    pygame.draw.rect(screen, COL_P1,       (W - 2 - p1_w, BAR_Y, p1_w, BAR_H))

    p0_label   = font.render(f"PPO            (blue) {terr['player_0']:>4} cells", True, COL_P0)
    p1_label   = font.render(f"{p1_label_str:<14} (red)  {terr['player_1']:>4} cells", True, COL_P1)
    info_label = font.render(
        f"step {env.step_count:>4}/{env.max_steps}   {fps} fps   +/- speed",
        True, (130, 130, 130),
    )

    screen.blit(p0_label,   (8, grid_px + 6))
    screen.blit(p1_label,   (8, grid_px + 24))
    screen.blit(info_label, (W - info_label.get_width() - 8, grid_px + 15))


def show_result(screen, font_lg, font_sm, clock, env):
    terr   = {a: int(np.sum(env.grid == (i + 1))) for i, a in enumerate(env.possible_agents)}
    p0, p1 = terr["player_0"], terr["player_1"]
    alive0 = env.alive.get("player_0", False)
    alive1 = env.alive.get("player_1", False)

    if alive0 and not alive1:
        headline, hl_col = "PPO wins!",  ( 80, 220,  80)
    elif alive1 and not alive0:
        headline, hl_col = "Greedy wins!", (220,  80,  80)
    elif p0 > p1:
        headline, hl_col = "PPO wins!",  ( 80, 220,  80)
    elif p1 > p0:
        headline, hl_col = "Greedy wins!", (220,  80,  80)
    else:
        headline, hl_col = "Draw!",      (220, 220,  80)

    detail = f"PPO: {p0} cells   |   Greedy: {p1} cells"
    prompt = "R  to restart        ESC / Q  to quit"

    W, H = screen.get_size()
    cx, cy = W // 2, H // 2

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_RESTART:
                    return "restart"
                if event.key in KEY_QUIT:
                    return "quit"

        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((10, 10, 10, 180))
        screen.blit(overlay, (0, 0))

        hl_surf = font_lg.render(headline, True, hl_col)
        screen.blit(hl_surf, (cx - hl_surf.get_width() // 2, cy - 80))

        for i, (txt, col) in enumerate([
            (detail, (190, 190, 190)),
            ("",     (0, 0, 0)),
            (prompt, (130, 130, 130)),
        ]):
            s = font_sm.render(txt, True, col)
            screen.blit(s, (cx - s.get_width() // 2, cy - 10 + i * 28))

        pygame.display.flip()
        clock.tick(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="ppo_selfplay_model",
                        help="Path to a saved SB3 model (default: ppo_selfplay_model)")
    parser.add_argument("--vs-greedy", action="store_true",
                        help="Play against the greedy agent instead of itself")
    args = parser.parse_args()

    print(f"Loading model from '{args.model_path}' ...")
    model = PPO.load(args.model_path)
    print("Model loaded.")

    env = PaperIOEnv(grid_size=20, reward_mode="mixed",
                     render_mode="rgb_array", max_steps=200)
    greedy = GreedyAgent("player_1", env.grid_size) if args.vs_greedy else None

    opponent_label = "Greedy" if args.vs_greedy else "PPO"
    pygame.display.set_caption(f"Paper.io — PPO (blue) vs {opponent_label} (red)")

    GRID_PX = env.grid_size * env._CELL_PX

    pygame.init()
    screen  = pygame.display.set_mode((GRID_PX, GRID_PX + HUD_H))
    font_sm = pygame.font.SysFont("monospace", 15)
    font_lg = pygame.font.SysFont("monospace", 40, bold=True)
    clock   = pygame.time.Clock()

    fps = DEF_FPS

    while True:
        obs, _ = env.reset()
        quit_requested = False

        while env.agents:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY_QUIT:
                        quit_requested = True
                    elif event.key in KEY_FASTER:
                        fps = min(fps + 2, MAX_FPS)
                    elif event.key in KEY_SLOWER:
                        fps = max(fps - 2, MIN_FPS)

            if quit_requested:
                break

            actions = {}
            if "player_0" in env.agents:
                obs_p0 = np.transpose(obs["player_0"], (2, 0, 1))
                action, _ = model.predict(obs_p0, deterministic=True)
                actions["player_0"] = int(action)
            if "player_1" in env.agents:
                if greedy is not None:
                    actions["player_1"] = greedy.act(env)
                else:
                    obs_p1 = np.rot90(obs["player_1"], k=2, axes=(0, 1)).copy()
                    for ch in (6, 7, 8, 9):
                        obs_p1[:, :, ch] = 1.0 - obs_p1[:, :, ch]
                    obs_p1 = np.transpose(obs_p1, (2, 0, 1))
                    action, _ = model.predict(obs_p1, deterministic=True)
                    actions["player_1"] = _ACTION_FLIP[int(action)]

            obs, _, _, _, _ = env.step(actions)

            frame = env.render()
            grid_surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            screen.blit(grid_surf, (0, 0))
            draw_hud(screen, font_sm, env, fps, GRID_PX, opponent_label)
            pygame.display.flip()
            clock.tick(fps)

        if quit_requested:
            break

        if show_result(screen, font_lg, font_sm, clock, env) == "quit":
            break

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
