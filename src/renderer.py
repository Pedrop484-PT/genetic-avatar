import numpy as np
from PIL import Image, ImageDraw
from src.config import CANVAS_WIDTH, CANVAS_HEIGHT


def _triangle_to_pixels(triangle, width: int, height: int) -> tuple:
    """Convert normalised [x1,y1,...,R,G,B,A] genes to pixel vertices and RGBA colour.

    Parameters
    ----------
    triangle : np.ndarray
        1-D array of 10 genes: x1,y1,x2,y2,x3,y3,R,G,B,A all in [0, 1].
    width : int
    height : int

    Returns
    -------
    tuple
        (vertices, color) where vertices is a list of (x, y) pixel pairs
        and color is a 4-tuple of uint8 RGBA values.
    """
    vertices = [(int(triangle[i] * width), int(triangle[i+1] * height)) for i in range(0, 6, 2)]
    color = tuple(int(min(max(triangle[i], 0.0), 1.0) * 255) for i in range(6, 10))
    return vertices, color


def render(chromosome, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT) -> np.ndarray:
    """Render a chromosome onto a black canvas and return an (H, W, 3) uint8 RGB array.

    Parameters
    ----------
    chromosome : np.ndarray
        (NUM_TRIANGLES, GENES_PER_TRIANGLE) array of genes in [0, 1].
    width : int
    height : int

    Returns
    -------
    np.ndarray
        (H, W, 3) uint8 RGB image.
    """
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    clear_box = (0, 0, width, height)

    for triangle in chromosome:
        vertices, color = _triangle_to_pixels(triangle, width, height)
        overlay.paste((0, 0, 0, 0), clear_box)
        draw.polygon(vertices, fill=color)
        canvas.alpha_composite(overlay)

    return np.array(canvas.convert("RGB"))
