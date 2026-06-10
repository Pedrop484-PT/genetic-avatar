from pathlib import Path

# Core Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data & Input Paths
DATA_DIR = PROJECT_ROOT / "data"
TARGET_IMAGE1_PATH = DATA_DIR / "girl_pearl_earing.png"
TARGET_IMAGE2_PATH = DATA_DIR / "mona_lisa.png"

# Output & Results Paths
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_IMAGE1_DIR = RESULTS_DIR / "girl_pearl_earing"
RESULTS_IMAGE2_DIR = RESULTS_DIR / "mona_lisa"


FITNESS_LOG_FILE = "fitness_log.csv"
BEST_INDIVIDUAL_FILE = "best_individual.npy"
CONVERGENCE_FILE = "convergence.png"
DIVERSITY_FILE = "diversity.png"
SNAPSHOTS_FILE = "evolution_snapshots.png"
CONVERGENCE_MULTI_RUN_FILE = "convergence_multi_run.png"
BEST_IMAGE_FILE = "best_image.png"

# Canvas & representation
CANVAS_WIDTH = 300
CANVAS_HEIGHT = 400
NUM_TRIANGLES = 100
GENES_PER_TRIANGLE = 10
FIGURE_DPI = 150

# GA hyperparameters
POP_SIZE = 100
N_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MUTATION_SIGMA = 0.1
ELITISM_RATIO = 0.05
N_ELITE = max(1, int(POP_SIZE * ELITISM_RATIO))
SEED = 42

# SA hyperparameters
SIGMA = 0.05
N_SAMPLES = 100
ACCEPTANCE_PROB = 0.8
TMIN = 1e-8
L_INNER_LOOP = 1
NUMBER_OF_ITERATIONS_SA = (POP_SIZE + N_GENERATIONS * (POP_SIZE - N_ELITE)) - (N_SAMPLES + 1)
