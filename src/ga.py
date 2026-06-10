import time
import numpy as np
from src.renderer import render
from src.fitness import rmse_fitness
from src.config import (
    NUM_TRIANGLES, GENES_PER_TRIANGLE,
    POP_SIZE, N_GENERATIONS, CROSSOVER_RATE,
    MUTATION_RATE, MUTATION_SIGMA, ELITISM_RATIO, SEED,
)


def random_individual(rng: np.random.Generator) -> np.ndarray:
    """Return a (NUM_TRIANGLES, 10) array of random genes in [0, 1]."""
    return rng.random((NUM_TRIANGLES, GENES_PER_TRIANGLE))


def random_individual_smart(rng: np.random.Generator, target: np.ndarray) -> np.ndarray:
    """Random geometry, but RGB sampled from the target image at each triangle's centroid.

    Warm-starts the colour palette and typically speeds up early convergence.
    """
    genome = random_individual(rng)
    h, w = target.shape[:2]

    # Vectorised centroid → pixel-index lookup
    cx = genome[:, [0, 2, 4]].mean(axis=1)
    cy = genome[:, [1, 3, 5]].mean(axis=1)
    px = np.clip((cx * w).astype(int), 0, w - 1)
    py = np.clip((cy * h).astype(int), 0, h - 1)
    genome[:, 6:9] = target[py, px, :3] / 255.0

    return genome


def initialize_population(
    pop_size: int,
    target: np.ndarray,
    rng: np.random.Generator,
    smart_ratio: float = 0.5
) -> list[np.ndarray]:
    """Build the initial population mixing smart and random chromosomes."""
    n_smart = int(pop_size * smart_ratio)
    population = []
    for i in range(pop_size):
        if i < n_smart:
            population.append(random_individual_smart(rng, target))
        else:
            population.append(random_individual(rng))
    return population


def tournament_selection(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
    replace: bool = True
) -> np.ndarray:
    """Draw k individuals at random and return the best (lowest fitness)."""
    if not replace and k > len(population):
        raise ValueError("Tournament size k cannot exceed population size when sampling without replacement.")

    # Select k random indices
    indices = rng.choice(len(population), size=k, replace=replace)
    
    # Find the index with the best (lowest) fitness among the selected
    best_idx = indices[0]
    best_fitness = fitnesses[best_idx]
    
    for idx in indices[1:]:
        # Active tie-breaking: if fitness is equal, flip a coin (50% chance)
        if fitnesses[idx] < best_fitness or (fitnesses[idx] == best_fitness and rng.random() < 0.5):
            best_fitness = fitnesses[idx]
            best_idx = idx

    return population[best_idx]


def select_parents(population, fitnesses, n_parents, tournament_k, rng):
    """Creates a mating pool of parents using tournament selection."""
    return [tournament_selection(population, fitnesses, tournament_k, rng) for _ in range(n_parents)]


def gaussian_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    sigma: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Add N(0, sigma) noise to each gene with probability mutation_rate, clamped to [0, 1]."""
    mutated = np.copy(individual)
    mutation_mask = rng.random(mutated.shape) < mutation_rate
    
    # Only generate noise if there are mutations to apply
    if np.any(mutation_mask):
        noise = rng.normal(loc=0.0, scale=sigma, size=mutated.shape)
        mutated[mutation_mask] += noise[mutation_mask]
        np.clip(mutated, 0.0, 1.0, out=mutated)
        
    return mutated


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Swap whole triangles (rows) between parents."""
    n_triangles = parent1.shape[0]
    mask = rng.random(n_triangles) < 0.5

    child1 = np.copy(parent1)
    child2 = np.copy(parent2)

    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]

    return child1, child2


def evaluate_population(population, target, fitness_fn=rmse_fitness):
    """Render every individual and score it against the target image."""
    return np.array([fitness_fn(render(ind), target) for ind in population])


def run_ga(
    target_image: np.ndarray,
    pop_size: int = POP_SIZE,
    n_generations: int = N_GENERATIONS,
    fitness_fn=rmse_fitness,
    crossover_rate: float = CROSSOVER_RATE,
    mutation_rate: float = MUTATION_RATE,
    mutation_sigma: float = MUTATION_SIGMA,
    elitism_ratio: float = ELITISM_RATIO,
    tournament_k: int = 5,
    smart_ratio: float = 0.5,
    seed: int = SEED,
    verbose: bool = True,
    adaptive_mutation: bool = False,
    mutation_sigma_min: float = 0.01,
    snapshot_generations: list = None
) -> dict:
    """Run the full GA loop with elitism, crossover, and mutation.
    
    Supports adaptive mutation (sigma decay), diversity tracking, and snapshots.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    snapshot_set = set(snapshot_generations) if snapshot_generations else set()
    population = initialize_population(pop_size, target_image, rng, smart_ratio)
    fitnesses = evaluate_population(population, target_image, fitness_fn)

    n_elite = max(1, int(pop_size * elitism_ratio))
    fitness_history: list[float] = []
    diversity_history: list[float] = []
    sigma_history: list[float] = []
    snapshots: dict[int, np.ndarray] = {}

    best_idx = int(np.argmin(fitnesses))
    best_ever_fitness = float(fitnesses[best_idx])
    best_ever_individual = population[best_idx].copy()

    log_interval = max(1, n_generations // 10)

    try:
        for gen in range(n_generations):
            # Adaptive mutation: linear decay of sigma
            if adaptive_mutation:
                progress = gen / max(n_generations - 1, 1)
                current_sigma = mutation_sigma + (mutation_sigma_min - mutation_sigma) * progress
            else:
                current_sigma = mutation_sigma
            
            sigma_history.append(current_sigma)

            # Elitism: carry over the best individuals and their fitnesses
            elite_indices = np.argpartition(fitnesses, n_elite)[:n_elite]
            next_population = [population[i].copy() for i in elite_indices]
            next_fitnesses = [fitnesses[i] for i in elite_indices]

            # Fill the rest with offspring
            offspring = []
            while len(next_population) + len(offspring) < pop_size:
                p1 = tournament_selection(population, fitnesses, tournament_k, rng)
                p2 = tournament_selection(population, fitnesses, tournament_k, rng)

                if rng.random() < crossover_rate:
                    c1, c2 = uniform_crossover(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = gaussian_mutation(c1, mutation_rate, current_sigma, rng)
                c2 = gaussian_mutation(c2, mutation_rate, current_sigma, rng)

                offspring.append(c1)
                offspring.append(c2)

            # Ensure population size remains exactly at pop_size by discarding the last if it exceeds
            if len(next_population) + len(offspring) > pop_size:
                offspring = offspring[:pop_size - len(next_population)]

            # Evaluate only the new offspring
            offspring_fitnesses = evaluate_population(offspring, target_image, fitness_fn)

            # Combine into new generation
            population = next_population + offspring
            fitnesses = np.concatenate([next_fitnesses, offspring_fitnesses])

            # Track global best
            best_idx = int(np.argmin(fitnesses))
            gen_best_fitness = float(fitnesses[best_idx])
            fitness_history.append(gen_best_fitness)

            if gen_best_fitness < best_ever_fitness:
                best_ever_fitness = gen_best_fitness
                best_ever_individual = population[best_idx].copy()

            # Diversity: average standard deviation of genes across population
            diversity_history.append(float(np.stack(population).std(axis=0).mean()))

            # Snapshot of the best individual
            if (gen + 1) in snapshot_set:
                snapshots[gen + 1] = render(best_ever_individual)

            if verbose and (gen + 1) % log_interval == 0:
                print(
                    f"  Gen {gen + 1:>5d}/{n_generations}  "
                    f"|  Best fitness: {best_ever_fitness:.4f}  "
                    f"|  sigma: {current_sigma:.4f}  "
                    f"|  Diversity: {diversity_history[-1]:.4f}"
                )
    except KeyboardInterrupt:
        print("\nGA interrupted by user. Returning current best.")

    return {
        "best_individual": best_ever_individual,
        "best_fitness": best_ever_fitness,
        "fitness_history": fitness_history,
        "diversity_history": diversity_history,
        "sigma_history": sigma_history,
        "snapshots": snapshots,
        "final_population": population,
        "final_fitnesses": fitnesses,
        "elapsed_time": time.time() - start_time,
    }
