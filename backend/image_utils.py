"""
Image processing utilities for pattern watermarking.
"""

import math
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple


def compute_grid_size(n_tokens: int) -> Tuple[int, int]:
    r = max(1, int(math.floor(math.sqrt(n_tokens))))
    c = max(1, int(math.ceil(n_tokens / r)))
    return r, c


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, w_bg, max_var, threshold = 0.0, 0, 0.0, 0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold


def binarize_image(img: Image.Image, rows: int, cols: int) -> np.ndarray:
    resized = img.resize((cols, rows), Image.BICUBIC)
    gray = np.array(resized.convert("L"), dtype=np.uint8)
    thresh = otsu_threshold(gray)
    binary = (gray > thresh).astype(np.uint8)
    return binary


def pattern_to_bits(pattern: np.ndarray) -> List[int]:
    return pattern.flatten().tolist()


def bits_to_pattern(bits: List[int], rows: int, cols: int) -> np.ndarray:
    total = rows * cols
    padded = list(bits) + [-1] * max(0, total - len(bits))
    return np.array(padded[:total], dtype=np.int8).reshape(rows, cols)


def reconstruct_grid(
    pattern_bits: List[int],
    recovered_bits: List[int],
    rows: int,
    cols: int,
) -> List[List[str]]:
    """
    Return a 2D grid of cell states for frontend visualisation:
      "hit"  -> recovered bit matches pattern AND is part of LCS
      "miss" -> recovered bit is part of LCS but bit value is 0
      "noise" -> position not in LCS (noise / modification)
    We encode as: "1" = green (bit 1 in LCS), "0" = red (bit 0 in LCS), "-" = noise.
    """
    from watermark_core import PatternWatermark

    _, lcs_pairs = PatternWatermark.compute_lcs(pattern_bits, recovered_bits)
    grid = [["-" for _ in range(cols)] for _ in range(rows)]
    for (bit, p_idx, _r_idx) in lcs_pairs:
        r = p_idx // cols
        c = p_idx % cols
        if r < rows and c < cols:
            grid[r][c] = "1" if bit == 1 else "0"
    return grid


def pattern_to_grid(pattern_bits: List[int], rows: int, cols: int) -> List[List[str]]:
    """Return the target pattern as a 2D grid of '1'/'0' strings for the frontend."""
    grid = [["0" for _ in range(cols)] for _ in range(rows)]
    for i, b in enumerate(pattern_bits):
        r = i // cols
        c = i % cols
        if r < rows and c < cols:
            grid[r][c] = "1" if b == 1 else "0"
    return grid


def make_builtin_pattern(choice: str, sz: int = 64) -> Image.Image:
    img = Image.new("RGB", (sz, sz), "white")
    d = ImageDraw.Draw(img)
    if choice == "square":
        d.rectangle([8, 8, sz - 9, sz - 9], fill="black")
    elif choice == "checkerboard":
        for r in range(0, sz, 8):
            for c in range(0, sz, 8):
                if (r // 8 + c // 8) % 2 == 0:
                    d.rectangle([c, r, c + 7, r + 7], fill="black")
    elif choice == "circle":
        d.ellipse([8, 8, sz - 9, sz - 9], outline="black", width=5)
    elif choice == "diamond":
        cx = sz // 2
        d.polygon([(cx, 4), (sz - 5, cx), (cx, sz - 5), (5, cx)], outline="black", fill="white")
    elif choice == "cross":
        t = sz // 4
        d.rectangle([t, 5, sz - t - 1, sz - 6], fill="black")
        d.rectangle([5, t, sz - 6, sz - t - 1], fill="black")
    else:
        d.rectangle([8, 8, sz - 9, sz - 9], fill="black")
    return img
