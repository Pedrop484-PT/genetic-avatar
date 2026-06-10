from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Grelha e paleta (cores do meu CV)
GRID = 24
PALETTE = [
    (26, 35, 50),     # tinta  #1a2332
    (201, 165, 87),   # dourado #c9a557
    (243, 239, 230),  # pergaminho #f3efe6
]

# Representacao: 1 gene por celula da grelha
NUM_TRIANGLES = GRID * GRID   # nome herdado do projeto CIFO; aqui sao celulas
GENES_PER_TRIANGLE = 1

# Canvas de fitness = a propria grelha
CANVAS_WIDTH = GRID
CANVAS_HEIGHT = GRID

# GA (herdado do CIFO; sigma maior porque os genes saltam entre bins discretos)
POP_SIZE = 200
N_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.02
MUTATION_SIGMA = 0.4
ELITISM_RATIO = 0.05
N_ELITE = max(1, int(POP_SIZE * ELITISM_RATIO))
SEED = 42
