import numpy as np

from src.config import GRID, PALETTE

_PALETTE_ARR = np.array(PALETTE, dtype=np.uint8)
_K = len(PALETTE)


def render(chromosome, width: int = GRID, height: int = GRID) -> np.ndarray:
    """Render a chromosome as a grid of palette-coloured squares.

    Parameters
    ----------
    chromosome : np.ndarray
        (GRID*GRID, 1) array of genes in [0, 1]. Each gene is quantised
        to one of the palette colours.
    width, height : int
        Output size in pixels (nearest-neighbour upscale of the grid).

    Returns
    -------
    np.ndarray
        (height, width, 3) uint8 RGB image.
    """
    idx = np.minimum((chromosome[:, 0] * _K).astype(int), _K - 1)
    img = _PALETTE_ARR[idx].reshape(GRID, GRID, 3)
    if width == GRID and height == GRID:
        return img
    reps_y, reps_x = height // GRID, width // GRID
    return np.repeat(np.repeat(img, reps_y, axis=0), reps_x, axis=1)
