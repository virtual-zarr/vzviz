"""Utility functions for visualization."""

from __future__ import annotations

import colorsys
from typing import Sequence


def generate_colors(n: int, saturation: float = 0.7, lightness: float = 0.5) -> list[str]:
    """
    Generate n visually distinct colors.

    Uses golden ratio to distribute hues evenly around the color wheel.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    saturation : float
        Color saturation (0-1).
    lightness : float
        Color lightness (0-1).

    Returns
    -------
    list[str]
        List of hex color strings.
    """
    colors = []
    golden_ratio = 0.618033988749895
    hue = 0.1  # Start hue

    for _ in range(n):
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        colors.append(hex_color)
        hue = (hue + golden_ratio) % 1.0

    return colors


def get_colormap(n: int, cmap_name: str | None = None) -> list[str]:
    """
    Get a list of colors for categorical data.

    Parameters
    ----------
    n : int
        Number of colors needed.
    cmap_name : str, optional
        Name of colormap to use. If None, uses generated distinct colors.
        Supported: "category10", "category20", or generates distinct colors.

    Returns
    -------
    list[str]
        List of hex color strings.
    """
    # Predefined categorical colormaps
    category10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    category20 = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
    ]

    if cmap_name == "category10":
        if n <= 10:
            return category10[:n]
        # Repeat if needed
        return (category10 * ((n // 10) + 1))[:n]

    if cmap_name == "category20":
        if n <= 20:
            return category20[:n]
        return (category20 * ((n // 20) + 1))[:n]

    # Default: generate distinct colors
    if n <= 10:
        return category10[:n]
    elif n <= 20:
        return category20[:n]
    else:
        return generate_colors(n)


def format_bytes(n_bytes: int) -> str:
    """
    Format byte count as human-readable string.

    Parameters
    ----------
    n_bytes : int
        Number of bytes.

    Returns
    -------
    str
        Human-readable string (e.g., "1.5 MB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} EB"


def truncate_path(path: str, max_length: int = 50) -> str:
    """
    Truncate a path for display, keeping the filename visible.

    Parameters
    ----------
    path : str
        Full path or URI.
    max_length : int
        Maximum length of output string.

    Returns
    -------
    str
        Truncated path with ellipsis if needed.
    """
    if len(path) <= max_length:
        return path

    # Keep the filename and as much of the path as possible
    parts = path.rsplit("/", 1)
    if len(parts) == 2:
        prefix, filename = parts
        if len(filename) >= max_length - 3:
            return "..." + filename[-(max_length - 3):]
        remaining = max_length - len(filename) - 4  # 4 for "/..."
        if remaining > 0:
            return prefix[:remaining] + "/..." + filename
        return ".../" + filename
    return "..." + path[-(max_length - 3):]
