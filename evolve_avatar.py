"""Evolve a pixel-art reconstruction of my profile photo with the CIFO GA.

Same evolutionary engine as our CIFO course project (genetic art with
triangles on Girl with a Pearl Earring); the representation here is a
24x24 grid of squares restricted to my CV colour palette.
Run:  python evolve_avatar.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

from src.ga import run_ga
from src.renderer import render
from src.config import GRID, POP_SIZE, N_GENERATIONS, MUTATION_RATE, MUTATION_SIGMA, SEED

HERE = Path(__file__).parent
TARGET_SRC = HERE / "data" / "target.jpg"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SNAPSHOTS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 450, 650, 1000]


def load_target() -> np.ndarray:
    img = Image.open(TARGET_SRC).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w + side) // 2, (h + side) // 2))
    img = img.resize((GRID, GRID), Image.LANCZOS)
    img.resize((800, 800), Image.NEAREST).save(OUT / "target_grid_preview.png")
    return np.array(img)


def to_800(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr).resize((800, 800), Image.NEAREST)


def save_outputs(result: dict) -> None:
    best = result["best_individual"]
    np.save(OUT / "best_individual.npy", best)
    to_800(render(best)).save(OUT / "avatar_800.png")
    np.savetxt(OUT / "fitness_history.csv",
               np.array(result["fitness_history"]), fmt="%.6f")

    frames = [to_800(result["snapshots"][g]).resize((320, 320), Image.NEAREST)
              for g in sorted(result["snapshots"])]
    if frames:
        # segura o ultimo frame mais tempo
        frames[0].save(OUT / "evolution.gif", save_all=True,
                       append_images=frames[1:] + [frames[-1]] * 4,
                       duration=350, loop=0)
    print(f"best_fitness={result['best_fitness']:.4f} "
          f"elapsed={result['elapsed_time']:.0f}s")


if __name__ == "__main__":
    target = load_target()
    result = run_ga(
        target_image=target,
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        mutation_rate=MUTATION_RATE,
        mutation_sigma=MUTATION_SIGMA,
        smart_ratio=0.0,          # smart init amostrava cores; nao se aplica a paleta discreta
        seed=SEED,
        snapshot_generations=SNAPSHOTS,
    )
    save_outputs(result)
