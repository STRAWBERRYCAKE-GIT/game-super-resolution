from __future__ import annotations

import numpy as np


def load_lut(path, scale: int, sampling_bits: int = 4) -> np.ndarray:
    if scale not in (2, 4):
        raise ValueError("This adapter supports x2 and x4.")
    if sampling_bits not in range(1, 9):
        raise ValueError("sampling_bits must be between 1 and 8.")
    table = np.load(path, allow_pickle=False)
    levels = 2 ** (8 - sampling_bits) + 1
    expected = levels ** 4 * scale ** 2
    if table.size != expected:
        raise ValueError(
            f"LUT size mismatch: got {table.shape}, expected "
            f"{levels ** 4} entries with {scale ** 2} outputs each. "
            "Check scale and sampling bits."
        )
    if table.dtype != np.int8:
        raise ValueError(f"Expected the official int8 LUT format, got {table.dtype}.")
    return table.reshape(levels ** 4, scale ** 2).astype(np.int32)


def simplex_direction(
    lut: np.ndarray, image: np.ndarray, scale: int,
    sampling_bits: int = 4, tile_rows: int = 32,
) -> np.ndarray:
    """One Ours-S 4D simplex pass, before rotation restoration.

    The lattice endpoints and q=2**bits follow the supplied official code,
    including its handling of the final sample at input value 255.
    """
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive.")
    h, w, channels = image.shape
    q = 2 ** sampling_bits
    levels = 2 ** (8 - sampling_bits) + 1
    strides = np.array([levels ** 3, levels ** 2, levels, 1], dtype=np.int64)
    padded = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode="reflect")
    output = np.empty((h * scale, w * scale, channels), dtype=np.float64)

    for top in range(0, h, tile_rows):
        bottom = min(top + tile_rows, h)
        # Last dimension indexes the four neighboring input pixels a,b,c,d.
        patch = np.stack(
            [padded[top:bottom, :w], padded[top:bottom, 1:w + 1],
             padded[top + 1:bottom + 1, :w], padded[top + 1:bottom + 1, 1:w + 1]],
            axis=-1,
        ).astype(np.int64)
        coarse, residual = patch // q, patch % q
        order = np.argsort(-residual, axis=-1, kind="stable")
        fractions = np.take_along_axis(residual, order, axis=-1)
        weights = np.concatenate(
            [q - fractions[..., :1],
             fractions[..., :-1] - fractions[..., 1:],
             fractions[..., -1:]],
            axis=-1,
        )
        vertex = (coarse * strides).sum(axis=-1)
        values = lut[vertex] * weights[..., 0, None]
        for dimension in range(4):
            vertex = vertex + strides[order[..., dimension]]
            values += lut[vertex] * weights[..., dimension + 1, None]
        # Preserve the supplied reference implementation's branch at line 134:
        # for fc > fd > fa > fb, it uses the c,a,d,b path (with one negative
        # weight) instead of c,d,a,b. Correcting it here would silently change
        # the official baseline. This compatibility behavior is intentional.
        fa, fb, fc, fd = (residual[..., index] for index in range(4))
        legacy = (fc > fd) & (fd > fa) & (fa > fb)
        if np.any(legacy):
            base = (coarse[legacy] * strides).sum(axis=-1)
            a, b, c, d = (v[legacy, None] for v in (fa, fb, fc, fd))
            values[legacy] = (
                (q - c) * lut[base]
                + (c - a) * lut[base + strides[2]]
                + (a - d) * lut[base + strides[2] + strides[0]]
                + (d - b) * lut[base + strides[2] + strides[0] + strides[3]]
                + b * lut[base + strides.sum()]
            )
        values = (values / q).reshape(bottom - top, w, channels, scale, scale)
        output[top * scale:bottom * scale] = values.transpose(
            0, 3, 1, 4, 2
        ).reshape((bottom - top) * scale, w * scale, channels)
    return output


def upscale(
    image: np.ndarray, lut: np.ndarray, scale: int,
    sampling_bits: int = 4, tile_rows: int = 32,
) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an HxWx3 uint8 RGB image.")
    if min(image.shape[:2]) < 2:
        raise ValueError("Input must be at least 2x2 for reflection padding.")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive.")
    expected = ((2 ** (8 - sampling_bits) + 1) ** 4, scale ** 2)
    if lut.shape != expected:
        raise ValueError(f"Expected LUT shape {expected}, got {lut.shape}.")
    total = np.zeros(
        (image.shape[0] * scale, image.shape[1] * scale, 3), dtype=np.float64
    )
    for rotation in range(4):
        rotated = np.rot90(image, rotation, axes=(0, 1))
        direction = simplex_direction(
            lut, rotated, scale, sampling_bits, tile_rows
        )
        total += np.rot90(direction, -rotation, axes=(0, 1))
    # Preserve the official normalization, sum of four directions, clipping,
    # and round-to-nearest conversion. Do not average the four outputs again.
    normalized = total / 255.0
    return np.round(np.clip(normalized, 0, 1) * 255).astype(np.uint8)

