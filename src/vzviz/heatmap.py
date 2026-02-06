"""Chunk-to-file heatmap visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from vzviz.core import get_array, get_dimension_names, manifest_to_dataframe

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestStore
    from vzviz.selection import SelectionState


def chunk_file_heatmap(
    store: "ManifestStore",
    variable: str,
    dim_x: int = -1,
    dim_y: int = -2,
    slice_indices: dict[int, int] | None = None,
    width: int = 600,
    height: int = 400,
    title: str | None = None,
) -> Any:
    """
    Create a heatmap showing chunk positions in array index space.

    Each cell represents a chunk, colored by the variable's color (consistent
    with the ByteMap). Cells span the array indices they contain.

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
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.
    title : str, optional
        Plot title. Auto-generated if None.

    Returns
    -------
    holoviews.Overlay
        The visualization object.

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
    array_shape = array.shape
    chunk_shape = array.chunks
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

    # Convert chunk indices to array index coordinates
    plot_df = _add_array_coordinates(
        plot_df, array_shape, chunk_shape, dim_x_idx, dim_y_idx
    )

    if title is None:
        title = f"ChunkMap: {variable} (Grid: {grid_shape})"

    dim_names = get_dimension_names(store, variable)

    return _heatmap_holoviews(
        plot_df,
        dim_x_idx,
        dim_y_idx,
        variable,
        width,
        height,
        title,
        dim_names,
    )


def _resolve_dim_index(dim: int, ndim: int) -> int:
    """Resolve dimension to a positive index."""
    if dim < 0:
        dim = ndim + dim

    if dim < 0 or dim >= ndim:
        raise ValueError(f"Dimension {dim} out of range for {ndim}-dimensional array")

    return dim


def _add_array_coordinates(
    df: pd.DataFrame,
    array_shape: tuple,
    chunk_shape: tuple,
    dim_x: int,
    dim_y: int | None,
) -> pd.DataFrame:
    """Add array index coordinates based on chunk indices.

    For each chunk, computes the center array index and the span (width/height)
    of that chunk in array index space.
    """
    df = df.copy()

    # X dimension - compute center and width
    x_col = f"dim_{dim_x}"
    if x_col in df.columns:
        chunk_size_x = chunk_shape[dim_x]
        array_size_x = array_shape[dim_x]

        # Compute start, end, center for each chunk
        df["x_start"] = df[x_col] * chunk_size_x
        df["x_end"] = ((df[x_col] + 1) * chunk_size_x).clip(upper=array_size_x)
        df["x_center"] = (df["x_start"] + df["x_end"]) / 2
        df["x_width"] = df["x_end"] - df["x_start"]

    # Y dimension - compute center and height
    if dim_y is not None:
        y_col = f"dim_{dim_y}"
        if y_col in df.columns:
            chunk_size_y = chunk_shape[dim_y]
            array_size_y = array_shape[dim_y]

            df["y_start"] = df[y_col] * chunk_size_y
            df["y_end"] = ((df[y_col] + 1) * chunk_size_y).clip(upper=array_size_y)
            df["y_center"] = (df["y_start"] + df["y_end"]) / 2
            df["y_height"] = df["y_end"] - df["y_start"]

    return df


def _compute_effective_slice_indices(
    df: pd.DataFrame,
    ndim: int,
    dim_x: int,
    dim_y: int | None,
    slice_indices: dict[int, int] | None,
) -> dict[int, int]:
    """Compute effective slice indices including defaults for non-displayed dims."""
    if slice_indices is None:
        slice_indices = {}

    display_dims = {dim_x}
    if dim_y is not None:
        display_dims.add(dim_y)

    effective = {}
    for dim in range(ndim):
        if dim not in display_dims:
            col = f"dim_{dim}"
            if col in df.columns:
                if dim in slice_indices:
                    effective[dim] = slice_indices[dim]
                else:
                    # Default to first value (same as _apply_slicing)
                    effective[dim] = int(df[col].min())

    return effective


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
    variable: str,
    width: int,
    height: int,
    title: str,
    dim_names: list[str] | None = None,
) -> Any:
    """Create heatmap using holoviews."""
    import holoviews as hv

    hv.extension("bokeh")
    return _create_heatmap_plot(
        hv,
        df,
        dim_x,
        dim_y,
        variable,
        width,
        height,
        title,
        dim_names=dim_names,
    )


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
    variable: str,
    width: int,
    height: int,
    title: str,
    dim_names: list[str] | None = None,
    selection_bounds: tuple[float, float, float, float] | None = None,
    interactive: bool = False,
) -> Any:
    """Create the heatmap plot with optional selection rectangle."""
    from vzviz.utils import get_variable_color_map

    x_label = _get_dim_label(dim_x, dim_names)
    y_label = _get_dim_label(dim_y, dim_names) if dim_y is not None else None

    # Get the variable's color (consistent with ByteMap)
    var_color = get_variable_color_map([variable])[variable]

    # Tools for interactive mode include box_select
    tools = ["hover", "box_select"] if interactive else ["hover"]

    # Check if we have array coordinates (x_start, x_end, etc.)
    has_array_coords = "x_start" in df.columns and "x_end" in df.columns

    if dim_y is not None and has_array_coords and "y_start" in df.columns:
        # Use Rectangles for proper array index display
        # Each rectangle spans the array indices contained in that chunk
        rects_data = df[
            [
                "x_start",
                "y_start",
                "x_end",
                "y_end",
                "variable",
                "chunk_key",
                "filename",
                "offset",
                "length",
            ]
        ].copy()

        heatmap = hv.Rectangles(
            rects_data,
            kdims=["x_start", "y_start", "x_end", "y_end"],
            vdims=["variable", "chunk_key", "filename", "offset", "length"],
        ).opts(
            fill_color=var_color,
            fill_alpha=1.0,
            width=width,
            height=height,
            title=title,
            tools=tools,
            xlabel=x_label,
            ylabel=y_label,
            line_color="white",
            line_width=1,
        )

        # Selection rectangle in array index space
        if selection_bounds is not None:
            x_min, y_min, x_max, y_max = selection_bounds
            selection_rect = hv.Rectangles([(x_min, y_min, x_max, y_max)]).opts(
                fill_alpha=0.2,
                fill_color="#FFFF00",
                line_color="#FFFF00",
                line_width=3,
            )
        else:
            selection_rect = hv.Rectangles([]).opts(fill_alpha=0, line_alpha=0)

        return heatmap * selection_rect

    elif dim_y is not None:
        # Fallback to HeatMap with chunk indices (no array coords available)
        x_col = f"dim_{dim_x}"
        y_col = f"dim_{dim_y}"

        # Create a constant color column for HeatMap
        df = df.copy()
        df["_color"] = 1

        heatmap = hv.HeatMap(
            df,
            kdims=[x_col, y_col],
            vdims=["_color", "variable", "chunk_key", "filename", "offset", "length"],
        ).opts(
            cmap=[var_color],
            colorbar=False,
            width=width,
            height=height,
            title=title,
            tools=tools,
            xlabel=x_label,
            ylabel=y_label,
            line_color="white",
            line_width=1,
        )

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
            selection_rect = hv.Rectangles([]).opts(fill_alpha=0, line_alpha=0)

        return heatmap * selection_rect
    else:
        # 1D case - use bars with array index coordinates if available
        x_col = f"dim_{dim_x}"

        if has_array_coords:
            # Use x_center for bar position
            bar_df = df[
                ["variable", "chunk_key", "filename", "offset", "length", "x_center"]
            ].copy()
            bar_df["_height"] = 1
            heatmap = hv.Bars(
                bar_df,
                kdims=["x_center"],
                vdims=[
                    "_height",
                    "variable",
                    "chunk_key",
                    "filename",
                    "offset",
                    "length",
                ],
            ).opts(
                color=var_color,
                width=width,
                height=height,
                title=title,
                tools=tools,
                xlabel=x_label,
            )
        else:
            df = df.copy()
            df["_height"] = 1
            heatmap = hv.Bars(
                df,
                kdims=[x_col],
                vdims=[
                    "_height",
                    "variable",
                    "chunk_key",
                    "filename",
                    "offset",
                    "length",
                ],
            ).opts(
                color=var_color,
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
    width: int = 600,
    height: int = 400,
    title: str | None = None,
) -> Any:
    """
    Create interactive heatmap with box selection for cross-panel synchronization.

    Use the box select tool to select a region of chunks. The selection will
    be synchronized with other panels (e.g., ByteMap) that share the
    same SelectionState. Cells are colored by variable (consistent with byte
    range chart) and span the array indices they contain.

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
    import holoviews as hv
    import panel as pn

    from vzviz.selection import SelectionState

    hv.extension("bokeh")

    if selection_state is None:
        selection_state = SelectionState()

    array = get_array(store, variable)
    df = manifest_to_dataframe(store, variable)

    if df.empty:
        raise ValueError("Manifest contains no chunks")

    grid_shape = array.manifest.shape_chunk_grid
    array_shape = array.shape
    chunk_shape = array.chunks
    ndim = len(grid_shape)

    if ndim < 1:
        raise ValueError("Cannot create heatmap for scalar array")

    dim_x_idx = _resolve_dim_index(dim_x, ndim)
    dim_y_idx = _resolve_dim_index(dim_y, ndim) if ndim > 1 else None

    if ndim == 1:
        dim_y_idx = None

    # Compute effective slice indices (including defaults for non-displayed dims)
    effective_slice_indices = _compute_effective_slice_indices(
        df, ndim, dim_x_idx, dim_y_idx, slice_indices
    )

    plot_df = _apply_slicing(df, ndim, dim_x_idx, dim_y_idx, slice_indices)

    if plot_df.empty:
        raise ValueError("No data after applying slice_indices")

    # Convert chunk indices to array index coordinates
    plot_df = _add_array_coordinates(
        plot_df, array_shape, chunk_shape, dim_x_idx, dim_y_idx
    )

    if title is None:
        title = f"ChunkMap: {variable} (Grid: {grid_shape})"

    dim_names = get_dimension_names(store, variable)

    # Store dimension and chunk info in selection state
    selection_state.dim_x = dim_x_idx
    selection_state.dim_y = dim_y_idx if dim_y_idx is not None else 0
    selection_state.chunk_shape = chunk_shape
    selection_state.bounds_in_array_space = True
    selection_state.bounds_variable = variable
    selection_state.slice_indices = effective_slice_indices

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
            plot_df,
            dim_x_idx,
            dim_y_idx,
            variable,
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
