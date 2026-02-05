"""Chunk-to-file heatmap visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from vzviz.core import get_array, get_dimension_names, manifest_to_dataframe
from vzviz.utils import get_colormap

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestStore
    from vzviz.selection import SelectionState


def chunk_file_heatmap(
    store: "ManifestStore",
    variable: str,
    dim_x: int = -1,
    dim_y: int = -2,
    slice_indices: dict[int, int] | None = None,
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
    store : ManifestStore
        The ManifestStore to visualize.
    variable : str
        Variable path to visualize.
    dim_x : int
        Dimension to display on x-axis. Negative indices count from end.
    dim_y : int
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
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> chunk_file_heatmap(store, "temperature")
    """
    array = get_array(store, variable)
    df = manifest_to_dataframe(store, variable)

    if df.empty:
        raise ValueError("Manifest contains no chunks")

    grid_shape = array.manifest.shape_chunk_grid
    ndim = len(grid_shape)

    if ndim < 1:
        raise ValueError("Cannot create heatmap for scalar array")

    dim_x_idx = _resolve_dim_index(dim_x, ndim)
    dim_y_idx = _resolve_dim_index(dim_y, ndim) if ndim > 1 else None

    if ndim == 1:
        dim_y_idx = None

    plot_df = _apply_slicing(df, ndim, dim_x_idx, dim_y_idx, slice_indices)

    if plot_df.empty:
        raise ValueError("No data after applying slice_indices")

    if title is None:
        title = f"Chunk-to-File Map: {variable} (Grid: {grid_shape})"

    dim_names = get_dimension_names(store, variable)

    if backend == "holoviews":
        return _heatmap_holoviews(
            plot_df,
            dim_x_idx,
            dim_y_idx,
            color_by,
            cmap,
            width,
            height,
            title,
            dim_names,
        )
    elif backend == "matplotlib":
        return _heatmap_matplotlib(
            plot_df,
            dim_x_idx,
            dim_y_idx,
            color_by,
            cmap,
            width,
            height,
            title,
            dim_names,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'holoviews' or 'matplotlib'."
        )


def _resolve_dim_index(dim: int, ndim: int) -> int:
    """Resolve dimension to a positive index."""
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
    slice_indices: dict[int, int] | None,
) -> pd.DataFrame:
    """Filter DataFrame to a 2D slice for visualization."""
    if slice_indices is None:
        slice_indices = {}

    display_dims = {dim_x}
    if dim_y is not None:
        display_dims.add(dim_y)

    plot_df = df.copy()
    for dim in range(ndim):
        if dim not in display_dims:
            col = f"dim_{dim}"
            if col in plot_df.columns:
                if dim in slice_indices:
                    slice_val = slice_indices[dim]
                    plot_df = plot_df[plot_df[col] == slice_val]
                else:
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
    dim_names: list[str] | None = None,
) -> Any:
    """Create heatmap using holoviews."""
    from vzviz._compat import import_holoviews

    hv = import_holoviews()
    prepared_df, color_col, resolved_cmap = _prepare_heatmap_data(df, color_by, cmap)
    return _create_heatmap_plot(
        hv,
        prepared_df,
        dim_x,
        dim_y,
        color_col,
        resolved_cmap,
        width,
        height,
        title,
        dim_names=dim_names,
    )


def _prepare_heatmap_data(
    df: pd.DataFrame,
    color_by: str,
    cmap: str | None,
) -> tuple[pd.DataFrame, str, str]:
    """Prepare DataFrame for heatmap visualization."""
    if color_by == "file":
        unique_files = df["path"].unique()
        file_to_idx = {f: i for i, f in enumerate(unique_files)}
        df = df.copy()
        df["color_val"] = df["path"].map(file_to_idx)
        color_col = "color_val"
        resolved_cmap = cmap if cmap is not None else "Category20"
    else:
        color_col = color_by if color_by in df.columns else "offset"
        resolved_cmap = cmap if cmap is not None else "viridis"

    return df, color_col, resolved_cmap


def _get_dim_label(dim_idx: int, dim_names: list[str] | None) -> str:
    """Get axis label for a dimension, using name if available."""
    if dim_names is not None and dim_idx < len(dim_names):
        return dim_names[dim_idx]
    return f"Dimension {dim_idx}"


def _create_heatmap_plot(
    hv: Any,
    df: pd.DataFrame,
    dim_x: int,
    dim_y: int | None,
    color_col: str,
    cmap: str,
    width: int,
    height: int,
    title: str,
    dim_names: list[str] | None = None,
    selection_bounds: tuple[float, float, float, float] | None = None,
    interactive: bool = False,
) -> Any:
    """Create the heatmap plot with optional selection rectangle."""
    x_col = f"dim_{dim_x}"
    y_col = f"dim_{dim_y}" if dim_y is not None else None

    x_label = _get_dim_label(dim_x, dim_names)
    y_label = _get_dim_label(dim_y, dim_names) if dim_y is not None else None

    # Tools for interactive mode include box_select
    tools = ["hover", "box_select"] if interactive else ["hover"]

    if y_col is not None and y_col in df.columns:
        heatmap = hv.HeatMap(
            df,
            kdims=[x_col, y_col],
            vdims=[color_col, "chunk_key", "filename", "offset", "length"],
        ).opts(
            cmap=cmap,
            colorbar=False,
            width=width,
            height=height,
            title=title,
            tools=tools,
            xlabel=x_label,
            ylabel=y_label,
        )

        # Always create selection rectangle (invisible if no selection)
        # This ensures consistent return type for DynamicMap
        if selection_bounds is not None:
            x_min, y_min, x_max, y_max = selection_bounds
            rect_bounds = (x_min - 0.5, y_min - 0.5, x_max + 0.5, y_max + 0.5)
            selection_rect = hv.Rectangles([rect_bounds]).opts(
                fill_alpha=0.2,
                fill_color="#FFFF00",
                line_color="#FFFF00",
                line_width=3,
            )
        else:
            # Invisible placeholder rectangle to maintain consistent type
            selection_rect = hv.Rectangles([]).opts(fill_alpha=0, line_alpha=0)

        return heatmap * selection_rect
    else:
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
            tools=tools,
            xlabel=x_label,
        )

        # Always create selection span (invisible if no selection)
        if selection_bounds is not None:
            x_min, _, x_max, _ = selection_bounds
            selection_span = hv.VSpan(x_min - 0.5, x_max + 0.5).opts(
                fill_alpha=0.2,
                fill_color="#FFFF00",
                line_color="#FFFF00",
                line_width=2,
            )
        else:
            # Invisible placeholder
            selection_span = hv.VSpan(0, 0).opts(fill_alpha=0, line_alpha=0)

        return heatmap * selection_span


def chunk_file_heatmap_interactive(
    store: "ManifestStore",
    variable: str,
    selection_state: "SelectionState | None" = None,
    dim_x: int = -1,
    dim_y: int = -2,
    slice_indices: dict[int, int] | None = None,
    color_by: Literal["file", "offset", "length"] = "file",
    cmap: str | None = None,
    width: int = 600,
    height: int = 400,
    title: str | None = None,
) -> Any:
    """
    Create interactive heatmap with box selection for cross-panel synchronization.

    Use the box select tool to select a region of chunks. The selection will
    be synchronized with other panels (e.g., byte range chart) that share the
    same SelectionState.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to visualize.
    variable : str
        Variable path to visualize.
    selection_state : SelectionState, optional
        Shared selection state for cross-panel synchronization.
        If None, creates an isolated SelectionState.
    dim_x : int
        Dimension to display on x-axis. Negative indices count from end.
    dim_y : int
        Dimension to display on y-axis.
    slice_indices : dict, optional
        For N-D arrays (N>2), specify fixed indices for dimensions not displayed.
    color_by : {"file", "offset", "length"}
        What to color cells by.
    cmap : str, optional
        Colormap name.
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.
    title : str, optional
        Plot title. Auto-generated if None.

    Returns
    -------
    panel.pane.HoloViews
        Interactive Panel component with box selection support.
    """
    from vzviz._compat import import_holoviews, import_panel
    from vzviz.selection import SelectionState

    hv = import_holoviews()
    pn = import_panel()

    if selection_state is None:
        selection_state = SelectionState()

    array = get_array(store, variable)
    df = manifest_to_dataframe(store, variable)

    if df.empty:
        raise ValueError("Manifest contains no chunks")

    grid_shape = array.manifest.shape_chunk_grid
    ndim = len(grid_shape)

    if ndim < 1:
        raise ValueError("Cannot create heatmap for scalar array")

    dim_x_idx = _resolve_dim_index(dim_x, ndim)
    dim_y_idx = _resolve_dim_index(dim_y, ndim) if ndim > 1 else None

    if ndim == 1:
        dim_y_idx = None

    plot_df = _apply_slicing(df, ndim, dim_x_idx, dim_y_idx, slice_indices)

    if plot_df.empty:
        raise ValueError("No data after applying slice_indices")

    if title is None:
        title = f"Chunk-to-File Map: {variable} (Grid: {grid_shape})"

    prepared_df, color_col, resolved_cmap = _prepare_heatmap_data(
        plot_df, color_by, cmap
    )
    dim_names = get_dimension_names(store, variable)

    # Store dimension info in selection state
    selection_state.dim_x = dim_x_idx
    selection_state.dim_y = dim_y_idx if dim_y_idx is not None else 0

    # Track last bounds to avoid redundant updates
    last_bounds: dict[str, tuple | None] = {"value": None}

    def render_heatmap(bounds: tuple | None) -> Any:
        """Render heatmap with selection rectangle."""
        # Update selection state when bounds change from box select
        if bounds is not None and bounds != last_bounds["value"]:
            last_bounds["value"] = bounds
            # bounds is (x0, y0, x1, y1) from BoundsXY
            selection_state.set_bounds(bounds)

        # Use selection state bounds for rendering
        return _create_heatmap_plot(
            hv,
            prepared_df,
            dim_x_idx,
            dim_y_idx,
            color_col,
            resolved_cmap,
            width,
            height,
            title,
            dim_names=dim_names,
            selection_bounds=selection_state.bounds,
            interactive=True,
        )

    # Create stream for box selection
    bounds_stream = hv.streams.BoundsXY(bounds=None)

    # Create DynamicMap
    dmap = hv.DynamicMap(render_heatmap, streams=[bounds_stream])

    return pn.pane.HoloViews(dmap)


def _heatmap_matplotlib(
    df: pd.DataFrame,
    dim_x: int,
    dim_y: int | None,
    color_by: str,
    cmap: str | None,
    width: int,
    height: int,
    title: str,
    dim_names: list[str] | None = None,
) -> Any:
    """Create heatmap using matplotlib."""
    from vzviz._compat import import_matplotlib

    plt = import_matplotlib()
    import matplotlib.colors as mcolors

    fig_width = width / 100
    fig_height = height / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_col = f"dim_{dim_x}"
    y_col = f"dim_{dim_y}" if dim_y is not None else None

    x_label = _get_dim_label(dim_x, dim_names)
    y_label = _get_dim_label(dim_y, dim_names) if dim_y is not None else None

    if y_col is not None and y_col in df.columns:
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
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        x_vals = sorted(df[x_col].unique())
        if color_by == "file":
            unique_files = list(df["path"].unique())
            file_to_idx = {f: i for i, f in enumerate(unique_files)}
            colors = get_colormap(len(unique_files))
            bar_colors = [
                colors[file_to_idx[df[df[x_col] == x]["path"].iloc[0]]] for x in x_vals
            ]
        else:
            bar_colors = ["#1f77b4"] * len(x_vals)

        heights = [1] * len(x_vals)
        ax.bar(x_vals, heights, color=bar_colors)
        ax.set_xlabel(x_label)

    ax.set_title(title)
    plt.tight_layout()
    return fig
