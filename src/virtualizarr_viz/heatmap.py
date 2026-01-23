"""Chunk-to-file heatmap visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from virtualizarr_viz.core import ManifestLike, extract_manifest, manifest_to_dataframe
from virtualizarr_viz.utils import get_colormap

if TYPE_CHECKING:
    pass


def chunk_file_heatmap(
    data: ManifestLike,
    variable: str | None = None,
    dim_x: int | str = -1,
    dim_y: int | str = -2,
    slice_indices: dict[int | str, int] | None = None,
    color_by: Literal["file", "offset", "length"] = "file",
    cmap: str | None = None,
    width: int = 600,
    height: int = 400,
    backend: Literal["holoviews", "matplotlib"] = "holoviews",
    title: str | None = None,
) -> Any:
    """
    Create a heatmap showing which files contain which chunks.

    Each cell represents a chunk in the chunk grid, colored by the source file,
    byte offset, or chunk length.

    Parameters
    ----------
    data : ManifestLike
        Input manifest data (ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray).
    variable : str, optional
        Variable name if data is an xr.Dataset.
    dim_x : int or str
        Dimension to display on x-axis. Can be integer index (negative indices
        count from end) or dimension name.
    dim_y : int or str
        Dimension to display on y-axis.
    slice_indices : dict, optional
        For N-D arrays (N>2), specify fixed indices for dimensions not displayed.
        E.g., {0: 5, 2: 0} fixes dimension 0 to chunk index 5 and dimension 2 to 0.
    color_by : {"file", "offset", "length"}
        What to color cells by:
        - "file": categorical color by source file
        - "offset": continuous color by byte offset
        - "length": continuous color by chunk length
    cmap : str, optional
        Colormap name. Default uses categorical for "file", viridis for others.
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.
    backend : {"holoviews", "matplotlib"}
        Visualization backend to use.
    title : str, optional
        Plot title. Auto-generated if None.

    Returns
    -------
    holoviews.HeatMap or matplotlib.Figure
        The visualization object from the selected backend.

    Examples
    --------
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import chunk_file_heatmap
    >>> vds = open_virtual_dataset("data.nc")
    >>> chunk_file_heatmap(vds, "temperature")

    For a 4D array, show lat-lon slice at time=0, level=5:

    >>> chunk_file_heatmap(
    ...     vds, "temperature",
    ...     dim_x=-1, dim_y=-2,
    ...     slice_indices={0: 0, 1: 5}
    ... )
    """
    manifest = extract_manifest(data, variable)
    df = manifest_to_dataframe(data, variable)

    if df.empty:
        raise ValueError("Manifest contains no chunks")

    # Get chunk grid shape
    grid_shape = manifest.shape_chunk_grid
    ndim = len(grid_shape)

    if ndim < 1:
        raise ValueError("Cannot create heatmap for scalar array")

    # Resolve dimension indices
    dim_x_idx = _resolve_dim_index(dim_x, ndim)
    dim_y_idx = _resolve_dim_index(dim_y, ndim) if ndim > 1 else None

    # For 1D arrays, create a simple bar-like visualization
    if ndim == 1:
        dim_y_idx = None

    # Apply slicing for N-D arrays
    plot_df = _apply_slicing(df, ndim, dim_x_idx, dim_y_idx, slice_indices)

    if plot_df.empty:
        raise ValueError("No data after applying slice_indices")

    # Generate title
    if title is None:
        title = f"Chunk-to-File Map (Grid shape: {grid_shape})"

    # Dispatch to backend
    if backend == "holoviews":
        return _heatmap_holoviews(plot_df, dim_x_idx, dim_y_idx, color_by, cmap, width, height, title)
    elif backend == "matplotlib":
        return _heatmap_matplotlib(plot_df, dim_x_idx, dim_y_idx, color_by, cmap, width, height, title)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'holoviews' or 'matplotlib'.")


def _resolve_dim_index(dim: int | str, ndim: int) -> int:
    """Resolve dimension to a positive index."""
    if isinstance(dim, str):
        raise NotImplementedError("Dimension names not yet supported, use integer indices")

    if dim < 0:
        dim = ndim + dim

    if dim < 0 or dim >= ndim:
        raise ValueError(f"Dimension {dim} out of range for {ndim}-dimensional array")

    return dim


def _apply_slicing(
    df: pd.DataFrame,
    ndim: int,
    dim_x: int,
    dim_y: int | None,
    slice_indices: dict[int | str, int] | None,
) -> pd.DataFrame:
    """Filter DataFrame to a 2D slice for visualization."""
    if slice_indices is None:
        slice_indices = {}

    # Determine which dimensions to slice
    display_dims = {dim_x}
    if dim_y is not None:
        display_dims.add(dim_y)

    # Apply slicing for non-display dimensions
    plot_df = df.copy()
    for dim in range(ndim):
        if dim not in display_dims:
            col = f"dim_{dim}"
            if col in plot_df.columns:
                if dim in slice_indices:
                    slice_val = slice_indices[dim]
                    plot_df = plot_df[plot_df[col] == slice_val]
                else:
                    # Default to first index
                    first_val = plot_df[col].min()
                    plot_df = plot_df[plot_df[col] == first_val]

    return plot_df


def _heatmap_holoviews(
    df: pd.DataFrame,
    dim_x: int,
    dim_y: int | None,
    color_by: str,
    cmap: str | None,
    width: int,
    height: int,
    title: str,
) -> Any:
    """Create heatmap using holoviews."""
    from virtualizarr_viz._compat import import_holoviews

    hv = import_holoviews()

    x_col = f"dim_{dim_x}"
    y_col = f"dim_{dim_y}" if dim_y is not None else None

    # Prepare color data
    if color_by == "file":
        # Create categorical file index
        unique_files = df["path"].unique()
        file_to_idx = {f: i for i, f in enumerate(unique_files)}
        df = df.copy()
        df["color_val"] = df["path"].map(file_to_idx)
        color_col = "color_val"
        if cmap is None:
            cmap = "Category20"
    else:
        color_col = color_by if color_by in df.columns else "offset"
        if cmap is None:
            cmap = "viridis"

    if y_col is not None and y_col in df.columns:
        # 2D heatmap
        heatmap = hv.HeatMap(
            df,
            kdims=[x_col, y_col],
            vdims=[color_col, "chunk_key", "filename", "offset", "length"],
        ).opts(
            cmap=cmap,
            colorbar=True,
            width=width,
            height=height,
            title=title,
            tools=["hover"],
            xlabel=f"Dimension {dim_x}",
            ylabel=f"Dimension {dim_y}",
        )
    else:
        # 1D case - use bars
        heatmap = hv.Bars(
            df,
            kdims=[x_col],
            vdims=[color_col, "chunk_key", "filename", "offset", "length"],
        ).opts(
            color=color_col,
            cmap=cmap,
            width=width,
            height=height,
            title=title,
            tools=["hover"],
            xlabel=f"Dimension {dim_x}",
        )

    return heatmap


def _heatmap_matplotlib(
    df: pd.DataFrame,
    dim_x: int,
    dim_y: int | None,
    color_by: str,
    cmap: str | None,
    width: int,
    height: int,
    title: str,
) -> Any:
    """Create heatmap using matplotlib."""
    from virtualizarr_viz._compat import import_matplotlib

    plt = import_matplotlib()
    import matplotlib.colors as mcolors

    fig_width = width / 100
    fig_height = height / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_col = f"dim_{dim_x}"
    y_col = f"dim_{dim_y}" if dim_y is not None else None

    if y_col is not None and y_col in df.columns:
        # Create 2D grid
        x_vals = sorted(df[x_col].unique())
        y_vals = sorted(df[y_col].unique())

        if color_by == "file":
            unique_files = list(df["path"].unique())
            file_to_idx = {f: i for i, f in enumerate(unique_files)}
            n_files = len(unique_files)

            grid = np.full((len(y_vals), len(x_vals)), np.nan)
            for _, row in df.iterrows():
                xi = x_vals.index(row[x_col])
                yi = y_vals.index(row[y_col])
                grid[yi, xi] = file_to_idx[row["path"]]

            colors = get_colormap(n_files)
            cmap_obj = mcolors.ListedColormap(colors)
            im = ax.imshow(grid, cmap=cmap_obj, aspect="auto", origin="lower")
        else:
            color_col = color_by if color_by in df.columns else "offset"
            grid = np.full((len(y_vals), len(x_vals)), np.nan)
            for _, row in df.iterrows():
                xi = x_vals.index(row[x_col])
                yi = y_vals.index(row[y_col])
                grid[yi, xi] = row[color_col]

            im = ax.imshow(grid, cmap=cmap or "viridis", aspect="auto", origin="lower")
            plt.colorbar(im, ax=ax, label=color_by.capitalize())

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels(y_vals)
        ax.set_xlabel(f"Dimension {dim_x}")
        ax.set_ylabel(f"Dimension {dim_y}")
    else:
        # 1D bar chart
        x_vals = sorted(df[x_col].unique())
        if color_by == "file":
            unique_files = list(df["path"].unique())
            file_to_idx = {f: i for i, f in enumerate(unique_files)}
            colors = get_colormap(len(unique_files))
            bar_colors = [colors[file_to_idx[df[df[x_col] == x]["path"].iloc[0]]] for x in x_vals]
        else:
            bar_colors = "#1f77b4"

        heights = [1] * len(x_vals)  # All same height for simple display
        ax.bar(x_vals, heights, color=bar_colors)
        ax.set_xlabel(f"Dimension {dim_x}")

    ax.set_title(title)
    plt.tight_layout()
    return fig
