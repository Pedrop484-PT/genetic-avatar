"""Evolve a triangle-art reconstruction of my profile photo with the CIFO GA.

Reuses the GA core from our CIFO course project (genetic art on
Girl with a Pearl Earring / Mona Lisa) and points it at my own photo.
Run:  python evolve_avatar.py [--bench]
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.ga import run_ga
from src.renderer import render

HERE = Path(__file__).parent
TARGET_SRC = str(HERE / "data" / "target.jpg")
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# canvas do projeto CIFO: 300x400 (retrato)
W, H = 300, 400

SNAPSHOTS = [1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000, 2500]

CONFIG = dict(
    pop_size=200,
    n_generations=2500,
    mutation_rate=0.01,
    mutation_sigma=0.05,
    crossover_rate=0.8,
    elitism_ratio=0.05,
    tournament_k=5,
    smart_ratio=0.5,
    seed=42,
)


def load_target() -> np.ndarray:
    img = Image.open(TARGET_SRC).convert("RGB")
    w, h = img.size
    # crop central 3:4
    target_ratio = W / H
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        img = img.crop(((w - new_w) // 2, 0, (w + new_w) // 2, h))
    else:
        new_h = int(w / target_ratio)
        img = img.crop((0, (h - new_h) // 2, w, (h + new_h) // 2))
    img = img.resize((W, H), Image.LANCZOS)
    img.save(OUT / "target_300x400.png")
    return np.array(img)


def save_outputs(result: dict) -> None:
    best = result["best_individual"]
    np.save(OUT / "best_individual.npy", best)

    # imagem final em alta resolucao + avatar quadrado
    hi = Image.fromarray(render(best, 600, 800))
    hi.save(OUT / "best_600x800.png")
    side = 600
    top = 60  # foco na cara, ligeiramente acima do centro
    avatar = hi.crop((0, top, side, top + side)).resize((800, 800), Image.LANCZOS)
    avatar.save(OUT / "avatar_800.png")

    # curva de fitness
    np.savetxt(OUT / "fitness_history.csv",
               np.array(result["fitness_history"]), fmt="%.6f")

    # GIF da evolucao
    frames = []
    for gen in sorted(result["snapshots"]):
        arr = result["snapshots"][gen]
        frames.append(Image.fromarray(arr).resize((240, 320), Image.NEAREST))
    if frames:
        frames[0].save(OUT / "evolution.gif", save_all=True,
                       append_images=frames[1:], duration=420, loop=0)
    print(f"best_fitness={result['best_fitness']:.4f} "
          f"elapsed={result['elapsed_time']:.0f}s")


if __name__ == "__main__":
    target = load_target()
    if "--bench" in sys.argv:
        cfg = dict(CONFIG, n_generations=10)
        t0 = time.time()
        run_ga(target_image=target, verbose=False, **cfg)
        per_gen = (time.time() - t0) / 10
        print(f"~{per_gen:.2f}s/geracao -> "
              f"{CONFIG['n_generations'] * per_gen / 60:.0f} min para a run completa")
    else:
        result = run_ga(target_image=target,
                        snapshot_generations=SNAPSHOTS, **CONFIG)
        save_outputs(result)
