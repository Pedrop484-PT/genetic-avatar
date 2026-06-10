import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000


def rmse_fitness(generated: np.ndarray, target: np.ndarray) -> float:
    """Pixel-wise RMSE between two RGB images. Lower is better (0 = identical)."""
    gen_f = generated.astype(np.float32)
    tgt_f = target if target.dtype == np.float32 else target.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(gen_f - tgt_f))))


def perceptual_fitness(generated: np.ndarray, target: np.ndarray) -> float:
    """Mean CIEDE2000 perceptual colour difference. Lower is better (0 = identical)."""
    return float(np.mean(deltaE_ciede2000(rgb2lab(generated), rgb2lab(target))))


def fitness(generated: np.ndarray, target: np.ndarray, method: str = "rmse") -> float:
    """Dispatch to rmse_fitness or perceptual_fitness. method: 'rmse' or 'perceptual'."""
    if method == "rmse":
        return rmse_fitness(generated, target)
    if method == "perceptual":
        return perceptual_fitness(generated, target)
    raise ValueError(f"Unknown fitness method: {method!r}. Use 'rmse' or 'perceptual'.")
