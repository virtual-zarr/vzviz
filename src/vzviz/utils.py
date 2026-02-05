"""Utility functions for visualization."""

from __future__ import annotations

import colorsys


def generate_colors(
    n: int, saturation: float = 0.7, lightness: float = 0.5
) -> list[str]:
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
        Name of colormap to use. If None, uses glasbey_cool from colorcet.
        Supported: "category10", "category20", "glasbey_cool", or generates distinct colors.

    Returns
    -------
    list[str]
        List of hex color strings.
    """
    # Try to use colorcet's glasbey_cool for better categorical colors
    try:
        import colorcet as cc

        glasbey_cool = cc.glasbey_cool
    except ImportError:
        glasbey_cool = None

    # Predefined categorical colormaps
    category10 = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    category20 = [
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]

    if cmap_name == "category10":
        if n <= 10:
            return category10[:n]
        return (category10 * ((n // 10) + 1))[:n]

    if cmap_name == "category20":
        if n <= 20:
            return category20[:n]
        return (category20 * ((n // 20) + 1))[:n]

    # Default: use glasbey_cool if available, otherwise fallback
    if glasbey_cool is not None:
        if n <= len(glasbey_cool):
            return glasbey_cool[:n]
        # Repeat if needed
        return (glasbey_cool * ((n // len(glasbey_cool)) + 1))[:n]

    # Fallback to category colormaps
    if n <= 10:
        return category10[:n]
    elif n <= 20:
        return category20[:n]
    else:
        return generate_colors(n)


def format_bytes(n_bytes: int) -> str:
    """
    Format byte count as MB.

    Parameters
    ----------
    n_bytes : int
        Number of bytes.

    Returns
    -------
    str
        String formatted in MB (e.g., "1.5 MB").
    """
    mb = n_bytes / (1024 * 1024)
    if mb >= 100:
        return f"{mb:.0f} MB"
    elif mb >= 10:
        return f"{mb:.1f} MB"
    elif mb >= 1:
        return f"{mb:.2f} MB"
    else:
        return f"{mb:.3f} MB"


def get_variable_color_map(variables: list[str]) -> dict[str, str]:
    """
    Create a consistent color mapping for variables.

    Parameters
    ----------
    variables : list[str]
        List of variable names/paths.

    Returns
    -------
    dict[str, str]
        Mapping from variable name to hex color.
    """
    unique_vars = sorted(set(variables))
    colors = get_colormap(len(unique_vars))
    return {var: colors[i] for i, var in enumerate(unique_vars)}


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
            return "..." + filename[-(max_length - 3) :]
        remaining = max_length - len(filename) - 4  # 4 for "/..."
        if remaining > 0:
            return prefix[:remaining] + "/..." + filename
        return ".../" + filename
    return "..." + path[-(max_length - 3) :]
