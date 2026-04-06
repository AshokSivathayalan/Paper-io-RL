# A script to let the user play against the trained PPO model

import sys
import argparse

import numpy as np
import pygame
from stable_baselines3 import PPO

from paperio_env import DOWN, LEFT, RIGHT, UP, PaperIOEnv

HUD_H   = 56   # height of the info strip below the grid
MIN_FPS =  2
MAX_FPS = 20
DEF_FPS =  8

COL_P0      = (100, 149, 237)   # blue (player)
COL_P1      = (255,  99,  71)   # red (PPO model)
COL_NEUTRAL = (150, 150, 150)
COL_BG      = ( 30,  30,  30)
COL_HUD     = ( 22,  22,  22)

# Flip UP↔DOWN and LEFT↔RIGHT to convert an action from a 180°-rotated
# observation back to real board directions (see watch.py / train_ppo.py).
_ACTION_FLIP = {0: 1, 1: 0, 2: 3, 3: 2}

# Controls
KEY_DIR = {
    pygame.K_UP:    UP,    pygame.K_w: UP,
    pygame.K_DOWN:  DOWN,  pygame.K_s: DOWN,
    pygame.K_LEFT:  LEFT,  pygame.K_a: LEFT,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
}
KEY_QUIT    = {pygame.K_ESCAPE, pygame.K_q}
KEY_RESTART = {pygame.K_r}
KEY_FASTER  = {pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS}
KEY_SLOWER  = {pygame.K_MINUS, pygame.K_KP_MINUS}


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: PaperIOEnv,
    fps: int,
    grid_px: int,
) -> None:
    """Draw the information strip beneath the game grid."""
    W = screen.get_width()
    screen.fill(COL_HUD, (0, grid_px, W, HUD_H))

    terr  = {a: int(np.sum(env.grid == (i + 1))) for i, a in enumerate(env.possible_agents)}
    total = env.grid_size ** 2

    # ── Territory bar ─────────────────────────────────────────────────────────
    BAR_Y = grid_px + HUD_H - 10
    BAR_H = 6
    bar_w = W - 4
    p0_w  = int(bar_w * terr["player_0"] / total)
    p1_w  = int(bar_w * terr["player_1"] / total)

    pygame.draw.rect(screen, (55, 55, 55),  (2, BAR_Y, bar_w, BAR_H))
    pygame.draw.rect(screen, COL_P0,        (2, BAR_Y, p0_w,  BAR_H))
    pygame.draw.rect(screen, COL_P1,        (W - 2 - p1_w, BAR_Y, p1_w, BAR_H))

    # ── Text labels ───────────────────────────────────────────────────────────
    p0_label   = font.render(f"You  (blue) {terr['player_0']:>4} cells", True, COL_P0)
    p1_label   = font.render(f"PPO  (red)  {terr['player_1']:>4} cells", True, COL_P1)
    info_label = font.render(
        f"step {env.step_count:>4}/{env.max_steps}   {fps} fps   +/- speed",
        True, (130, 130, 130),
    )

    screen.blit(p0_label,   (8, grid_px + 6))
    screen.blit(p1_label,   (8, grid_px + 24))
    screen.blit(info_label, (W - info_label.get_width() - 8, grid_px + 15))

def show_result(
    screen: pygame.Surface,
    font_lg: pygame.font.Font,
    font_sm: pygame.font.Font,
    clock: pygame.time.Clock,
    env: PaperIOEnv,
) -> str:
    """
    Overlay the outcome on top of the final grid frame.
    Returns "restart" or "quit".
    """
    terr   = {a: int(np.sum(env.grid == (i + 1))) for i, a in enumerate(env.possible_agents)}
    p0, p1 = terr["player_0"], terr["player_1"]
    alive0 = env.alive.get("player_0", False)
    alive1 = env.alive.get("player_1", False)

    # Determine headline
    if not alive0 and alive1:
        headline, hl_col = "PPO wins!",  (220,  80,  80)
    elif alive0 and not alive1:
        headline, hl_col = "You win!",   ( 80, 220,  80)
    elif p0 > p1:
        headline, hl_col = "You win!",   ( 80, 220,  80)
    elif p1 > p0:
        headline, hl_col = "PPO wins!",  (220,  80,  80)
    else:
        headline, hl_col = "Draw!",      (220, 220,  80)

    detail = f"Your territory: {p0}  |  PPO territory: {p1}"
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

        # Semi-transparent overlay
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default="ppo_selfplay_model",
                        help="Path to saved SB3 model (default: ppo_selfplay_model)")
    args = parser.parse_args()

    print(f"Loading model from '{args.model_path}' ...")
    model = PPO.load(args.model_path)
    print("Model loaded.")

    env = PaperIOEnv(
        grid_size=20,
        reward_mode="mixed",
        render_mode="rgb_array",
        max_steps=500,
    )

    GRID_PX = env.grid_size * env._CELL_PX

    pygame.init()
    screen  = pygame.display.set_mode((GRID_PX, GRID_PX + HUD_H))
    pygame.display.set_caption("Paper.io  —  You (blue) vs PPO model (red)")
    font_sm = pygame.font.SysFont("monospace", 15)
    font_lg = pygame.font.SysFont("monospace", 40, bold=True)
    clock   = pygame.time.Clock()

    fps       = DEF_FPS
    human_dir = RIGHT

    # Restart loop
    while True:
        obs, _ = env.reset()
        human_dir      = RIGHT
        quit_requested = False

        # Episode loop
        while env.agents:
            # Input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY_DIR:
                        human_dir = KEY_DIR[event.key]
                    elif event.key in KEY_QUIT:
                        quit_requested = True
                    elif event.key in KEY_FASTER:
                        fps = min(fps + 2, MAX_FPS)
                    elif event.key in KEY_SLOWER:
                        fps = max(fps - 2, MIN_FPS)

            if quit_requested:
                break

            # Actions
            actions = {}
            if "player_0" in env.agents:
                actions["player_0"] = human_dir
            if "player_1" in env.agents:
                # The model was trained as player_0, so present a 180°-rotated
                # view and flip the chosen action back to real coordinates.
                obs_p1 = np.rot90(obs["player_1"], k=2, axes=(0, 1)).copy()
                for ch in (6, 7, 8, 9):
                    obs_p1[:, :, ch] = 1.0 - obs_p1[:, :, ch]
                obs_p1 = np.transpose(obs_p1, (2, 0, 1))
                action, _ = model.predict(obs_p1, deterministic=False)
                actions["player_1"] = _ACTION_FLIP[int(action)]

            obs, _, _, _, _ = env.step(actions)

            # Render grid (env draws to its own offscreen surface)
            frame     = env.render()                                    # (H, W, 3)
            grid_surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            screen.blit(grid_surf, (0, 0))

            draw_hud(screen, font_sm, env, fps, GRID_PX)
            pygame.display.flip()
            clock.tick(fps)

        if quit_requested:
            break

        # result screen
        decision = show_result(screen, font_lg, font_sm, clock, env)
        if decision == "quit":
            break

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
