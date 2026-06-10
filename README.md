# genetic-avatar

My GitHub avatar, evolved. A genetic algorithm reconstructs my profile photo with 100 semi-transparent triangles. The GA core comes from our Computational Intelligence for Optimization course project at NOVA IMS, where the same algorithm reconstructed Girl with a Pearl Earring and transferred to the Mona Lisa; here it gets pointed at my own face.

![Evolution](outputs/evolution.gif)

## How it works

- Representation: a (100, 10) NumPy array, one row per triangle: three vertices, RGB and alpha, all genes in [0, 1]
- Fitness: RMSE between the rendered canvas (300x400) and the target photo
- Selection: tournament of 5 with coin-flip tie-breaking
- Variation: whole-triangle uniform crossover (rate 0.8), Gaussian mutation N(0, 0.05) at rate 0.01, clamped to [0, 1]
- Elitism 5%, smart initialisation (half the population samples its colours from the target at each triangle's centroid)
- Budget: population 200, 2500 generations, seed 42. The mutation and population settings are the winning configuration from the grid search in the course project

## Run

```bash
pip install -r requirements.txt
python evolve_avatar.py            # full run
python evolve_avatar.py --bench    # estimate cost per generation first
```

Outputs land in `outputs/`: the evolved avatar (`avatar_800.png`), a high-resolution render, the fitness history and an evolution GIF.

## Credit

GA core written with Gonçalo Bento, Filipe Carmo and Marta La Feria for the CIFO course project (NOVA IMS, MSc Data Science and Advanced Analytics, 2025/26).
