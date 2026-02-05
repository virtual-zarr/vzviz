"""Byte range chart visualization showing chunk positions within files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from vzviz.core import manifest_to_dataframe
from vzviz.utils import (
    format_bytes,
    get_colormap,
    get_variable_color_map,
    truncate_path,
)

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestStore
    from vzviz.selection import SelectionState


def byte_range_chart(
    store: "ManifestStore",
    variable: str | None = None,
    max_files: int = 30,
    sort_by: Literal["name", "chunks", "bytes", "offset"] = "offset",
    color_by: Literal["variable", "chunk", "dim_0", "file", "none"] = "variable",
    show_gaps: bool = False,
    width: int = 900,
    height: int | None = None,
    backend: Literal["holoviews", "matplotlib"] = "holoviews",
    title: str | None = None,
) -> Any:
    """
    Visualize chunk byte ranges within each file.

    Creates a horizontal bar chart where each row represents a file and
    segments show chunk positions [offset, offset+length).

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to visualize.
    variable : str, optional
        If provided, only show chunks for this variable.
    max_files : int
        Maximum number of files to display. If exceeded, shows the files with
        the most chunks.
    sort_by : {"name", "chunks", "bytes", "offset"}
        How to sort files:
        - "name": alphabetically by filename
        - "chunks": by number of chunks (descending)
        - "bytes": by total bytes (descending)
        - "offset": by minimum offset
    color_by : {"variable", "chunk", "dim_0", "file", "none"}
        How to color segments:
        - "variable": color by variable name (default, consistent with heatmap)
        - "chunk": unique color per chunk (based on chunk key)
        - "dim_0": color by first dimension index
        - "file": same color for all chunks in a file
        - "none": uniform color
    show_gaps : bool
        If True, highlight gaps between chunks within a file.
    width : int
        Plot width in pixels.
    height : int, optional
        Plot height in pixels. If None, auto-calculated based on file count.
    backend : {"holoviews", "matplotlib"}
        Visualization backend to use.
    title : str, optional
        Plot title. Auto-generated if None.

    Returns
    -------
    holoviews.Layout or matplotlib.Figure
        The visualization object from the selected backend.

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> byte_range_chart(store, "temperature", sort_by="chunks")
    """
    df = manifest_to_dataframe(store, variable)

    if df.empty:
        raise ValueError("Manifest contains no chunks")

    plot_df = _prepare_byterange_data(df, max_files, sort_by, color_by)

    n_files = plot_df["filename"].nunique()
    if height is None:
        height = max(200, min(800, 50 + n_files * 25))

    if title is None:
        total_chunks = len(df)
        total_files = df["path"].nunique()
        title = f"ByteMap ({total_chunks} chunks across {total_files} files)"

    if backend == "holoviews":
        return _byte_range_holoviews(plot_df, width, height, title, color_by, show_gaps)
    elif backend == "matplotlib":
        return _byte_range_matplotlib(
            plot_df, width, height, title, color_by, show_gaps
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'holoviews' or 'matplotlib'."
        )


def _prepare_byterange_data(
    df: pd.DataFrame,
    max_files: int,
    sort_by: str,
    color_by: str,
) -> pd.DataFrame:
    """Prepare DataFrame for byte range visualization."""
    file_stats = (
        df.groupby("path")
        .agg(
            chunk_count=("chunk_key", "count"),
            total_bytes=("length", "sum"),
            min_offset=("offset", "min"),
            filename=("filename", "first"),
        )
        .reset_index()
    )

    if sort_by == "name":
        file_stats = file_stats.sort_values("filename")
    elif sort_by == "chunks":
        file_stats = file_stats.sort_values("chunk_count", ascending=False)
    elif sort_by == "bytes":
        file_stats = file_stats.sort_values("total_bytes", ascending=False)
    elif sort_by == "offset":
        file_stats = file_stats.sort_values("min_offset")

    if len(file_stats) > max_files:
        file_stats = file_stats.nlargest(max_files, "chunk_count")

    selected_paths = set(file_stats["path"])
    plot_df = df[df["path"].isin(selected_paths)].copy()

    path_to_label = {}
    for _, row in file_stats.iterrows():
        label = truncate_path(row["filename"], 40)
        label = f"{label} ({row['chunk_count']} chunks)"
        path_to_label[row["path"]] = label

    plot_df["y_label"] = plot_df["path"].map(path_to_label)

    sorted_paths = file_stats["path"].tolist()
    path_to_y = {path: i for i, path in enumerate(sorted_paths)}
    plot_df["y_pos"] = plot_df["path"].map(path_to_y)

    if color_by == "variable" and "variable" in plot_df.columns:
        var_color_map = get_variable_color_map(plot_df["variable"].tolist())
        plot_df["color"] = plot_df["variable"].map(var_color_map)
    elif color_by == "chunk":
        n_chunks = len(plot_df)
        colors = get_colormap(n_chunks)
        plot_df["color"] = colors[: len(plot_df)]
    elif color_by == "dim_0" and "dim_0" in plot_df.columns:
        n_dims = plot_df["dim_0"].nunique()
        colors = get_colormap(n_dims)
        dim_to_color = {
            dim: colors[i] for i, dim in enumerate(sorted(plot_df["dim_0"].unique()))
        }
        plot_df["color"] = plot_df["dim_0"].map(dim_to_color)
    elif color_by == "file":
        n_files = plot_df["path"].nunique()
        colors = get_colormap(n_files)
        path_to_color = {path: colors[i] for i, path in enumerate(sorted_paths)}
        plot_df["color"] = plot_df["path"].map(path_to_color)
    else:
        plot_df["color"] = "#1f77b4"

    return plot_df


def _byte_range_holoviews(
    df: pd.DataFrame,
    width: int,
    height: int,
    title: str,
    color_by: str,
    show_gaps: bool,
) -> Any:
    """Create byte range chart using holoviews."""
    from vzviz._compat import import_holoviews

    hv = import_holoviews()

    seg_df = _prepare_segments_df(df)
    y_labels = df[["y_pos", "y_label"]].drop_duplicates().sort_values("y_pos")
    yticks = list(zip(y_labels["y_pos"], y_labels["y_label"]))

    return _create_segments_plot(hv, seg_df, width, height, title, yticks)


def _bytes_to_mb(n_bytes: int | float) -> float:
    """Convert bytes to MB."""
    return n_bytes / (1024 * 1024)


def _prepare_segments_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for segments plot with values in MB."""
    segments = []
    for _, row in df.iterrows():
        segments.append(
            {
                "x0": _bytes_to_mb(row["offset"]),
                "x1": _bytes_to_mb(row["end_offset"]),
                "y": row["y_pos"],
                "y_label": row["y_label"],
                "color": row["color"],
                "chunk_key": row["chunk_key"],
                "path": row["path"],
                "offset": row["offset"],  # Keep original bytes for hover
                "length": row["length"],  # Keep original bytes for hover
                "filename": row["filename"],
            }
        )
    return pd.DataFrame(segments)


def _prepare_gaps_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for gaps (non-chunk regions) with values in MB.

    Gaps are computed as the inverse of chunks - regions from 0 to first chunk,
    between chunks, etc.
    """
    gaps = []

    # Group by file (y_pos)
    for y_pos in df["y_pos"].unique():
        file_df = df[df["y_pos"] == y_pos].copy()
        file_df = file_df.sort_values("offset")

        y_label = file_df["y_label"].iloc[0]
        path = file_df["path"].iloc[0]
        filename = file_df["filename"].iloc[0]

        # Gap from 0 to first chunk (leading metadata)
        first_offset = file_df["offset"].iloc[0]
        if first_offset > 0:
            gaps.append(
                {
                    "x0": 0,
                    "x1": _bytes_to_mb(first_offset),
                    "y": y_pos,
                    "y_label": y_label,
                    "color": "#888888",  # Gray for gaps
                    "gap_type": "header",
                    "path": path,
                    "offset": 0,
                    "length": first_offset,
                    "filename": filename,
                }
            )

        # Gaps between consecutive chunks
        prev_end = None
        for _, row in file_df.iterrows():
            if prev_end is not None and row["offset"] > prev_end:
                gap_size = row["offset"] - prev_end
                gaps.append(
                    {
                        "x0": _bytes_to_mb(prev_end),
                        "x1": _bytes_to_mb(row["offset"]),
                        "y": y_pos,
                        "y_label": y_label,
                        "color": "#aa4444",  # Reddish for inter-chunk gaps
                        "gap_type": "gap",
                        "path": path,
                        "offset": prev_end,
                        "length": gap_size,
                        "filename": filename,
                    }
                )
            prev_end = row["end_offset"]

    return pd.DataFrame(gaps) if gaps else pd.DataFrame()


def _create_segments_plot(
    hv: Any,
    seg_df: pd.DataFrame,
    width: int,
    height: int,
    title: str,
    yticks: list,
    selected_chunk_keys: set[str] | None = None,
    show_gaps: bool = False,
    gaps_df: pd.DataFrame | None = None,
) -> Any:
    """Create the segments plot with optional highlight for selected chunks."""
    # Calculate x-axis range starting from 0
    max_x = seg_df["x1"].max() if not seg_df.empty else 1

    common_opts = dict(
        tools=["hover"],
        width=width,
        height=height,
        title=title,
        xlabel="Offset (MB)",
        ylabel="",
        show_legend=False,
        yticks=yticks,
        invert_yaxis=True,
        xlim=(0, max_x * 1.02),  # Start at 0, small padding on right
    )

    # If showing gaps, render gaps instead of chunks
    if show_gaps and gaps_df is not None and not gaps_df.empty:
        return _create_gaps_plot(hv, gaps_df, common_opts)

    if selected_chunk_keys:
        selected = seg_df[seg_df["chunk_key"].isin(selected_chunk_keys)]
        non_selected = seg_df[~seg_df["chunk_key"].isin(selected_chunk_keys)]

        if not selected.empty:
            # Dimmed non-selected segments
            dimmed_plot = hv.Segments(
                non_selected,
                kdims=["x0", "y", "x1", "y"],
                vdims=[
                    "color",
                    "chunk_key",
                    "path",
                    "offset",
                    "length",
                    "filename",
                    "y_label",
                ],
            ).opts(
                color="color",
                line_width=8,
                alpha=0.25,
                **common_opts,
            )

            # Bright border for selected chunks
            highlight = hv.Segments(
                selected,
                kdims=["x0", "y", "x1", "y"],
            ).opts(
                color="#FFFF00",
                line_width=16,
            )

            # Selected segments at full opacity
            selected_plot = hv.Segments(
                selected,
                kdims=["x0", "y", "x1", "y"],
                vdims=[
                    "color",
                    "chunk_key",
                    "path",
                    "offset",
                    "length",
                    "filename",
                    "y_label",
                ],
            ).opts(
                color="color",
                line_width=8,
            )

            return dimmed_plot * highlight * selected_plot

    # No selection - show all at full opacity
    # Always return an Overlay for consistent type with DynamicMap
    segments_plot = hv.Segments(
        seg_df,
        kdims=["x0", "y", "x1", "y"],
        vdims=["color", "chunk_key", "path", "offset", "length", "filename", "y_label"],
    ).opts(
        color="color",
        line_width=8,
        **common_opts,
    )

    # Create invisible placeholder to maintain consistent Overlay type
    placeholder = hv.Segments(
        [],
        kdims=["x0", "y", "x1", "y"],
    ).opts(alpha=0)

    return segments_plot * placeholder


def _create_gaps_plot(hv: Any, gaps_df: pd.DataFrame, common_opts: dict) -> Any:
    """Create a plot showing gaps (non-chunk regions) instead of chunks."""
    if gaps_df.empty:
        # Return empty overlay if no gaps
        placeholder = hv.Segments([], kdims=["x0", "y", "x1", "y"]).opts(alpha=0)
        return placeholder * placeholder

    # Separate header gaps from inter-chunk gaps for different colors
    header_gaps = gaps_df[gaps_df["gap_type"] == "header"]
    inter_gaps = gaps_df[gaps_df["gap_type"] == "gap"]

    plots = []

    if not header_gaps.empty:
        header_plot = hv.Segments(
            header_gaps,
            kdims=["x0", "y", "x1", "y"],
            vdims=[
                "color",
                "gap_type",
                "path",
                "offset",
                "length",
                "filename",
                "y_label",
            ],
        ).opts(
            color="#888888",  # Gray for header/metadata
            line_width=8,
            **common_opts,
        )
        plots.append(header_plot)

    if not inter_gaps.empty:
        gap_plot = hv.Segments(
            inter_gaps,
            kdims=["x0", "y", "x1", "y"],
            vdims=[
                "color",
                "gap_type",
                "path",
                "offset",
                "length",
                "filename",
                "y_label",
            ],
        ).opts(
            color="#aa4444",  # Reddish for inter-chunk gaps
            line_width=8,
            **(common_opts if not plots else {}),  # Only apply opts to first plot
        )
        plots.append(gap_plot)

    if not plots:
        placeholder = hv.Segments([], kdims=["x0", "y", "x1", "y"]).opts(alpha=0)
        return placeholder * placeholder

    # Combine plots
    result = plots[0]
    for p in plots[1:]:
        result = result * p

    # Add placeholder for consistent Overlay type
    placeholder = hv.Segments([], kdims=["x0", "y", "x1", "y"]).opts(alpha=0)
    return result * placeholder


def byte_range_chart_interactive(
    store: "ManifestStore",
    variable: str | None = None,
    selection_state: "SelectionState | None" = None,
    max_files: int = 30,
    sort_by: Literal["name", "chunks", "bytes", "offset"] = "offset",
    color_by: Literal["variable", "chunk", "dim_0", "file", "none"] = "variable",
    width: int = 900,
    height: int | None = None,
    title: str | None = None,
    include_toggle: bool = False,
) -> Any:
    """
    Create interactive byte range chart with cross-panel selection support.

    This chart highlights chunks that are selected via box selection in
    the heatmap panel. The selection is driven by the shared SelectionState.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to visualize.
    variable : str, optional
        If provided, only show chunks for this variable.
    selection_state : SelectionState, optional
        Shared selection state for cross-panel synchronization.
        If None, creates an isolated SelectionState.
    max_files : int
        Maximum number of files to display.
    sort_by : {"name", "chunks", "bytes", "offset"}
        How to sort files.
    color_by : {"chunk", "dim_0", "file", "none"}
        How to color segments.
    width : int
        Plot width in pixels.
    height : int, optional
        Plot height in pixels. Auto-calculated if None.
    title : str, optional
        Plot title. Auto-generated if None.
    include_toggle : bool
        If True, include a toggle to switch between chunks and gaps view.

    Returns
    -------
    panel.pane.HoloViews or panel.Column
        Interactive Panel component with selection highlighting.
        If include_toggle=True, returns a Column with toggle and chart.
    """
    from vzviz._compat import import_holoviews, import_panel
    from vzviz.selection import SelectionState

    hv = import_holoviews()
    pn = import_panel()

    if selection_state is None:
        selection_state = SelectionState()

    df = manifest_to_dataframe(store, variable)
    if df.empty:
        raise ValueError("Manifest contains no chunks")

    plot_df = _prepare_byterange_data(df, max_files, sort_by, color_by)

    n_files = plot_df["filename"].nunique()
    if height is None:
        height = max(200, min(800, 50 + n_files * 25))

    if title is None:
        total_chunks = len(df)
        total_files = df["path"].nunique()
        title = f"ByteMap ({total_chunks} chunks across {total_files} files)"

    seg_df = _prepare_segments_df(plot_df)
    gaps_df = _prepare_gaps_df(plot_df)
    y_labels = plot_df[["y_pos", "y_label"]].drop_duplicates().sort_values("y_pos")
    yticks = list(zip(y_labels["y_pos"], y_labels["y_label"]))

    if include_toggle:
        # Create toggle widget
        view_toggle = pn.widgets.RadioButtonGroup(
            name="View",
            options=["Chunks", "Gaps/Metadata"],
            value="Chunks",
            button_type="default",
        )

        # Container for the chart
        chart_container = pn.Column()

        def update_chart(event=None):
            """Update chart based on toggle and selection state."""
            chart_container.clear()
            show_gaps = view_toggle.value == "Gaps/Metadata"

            # Only highlight chunks from bounds selection, not variable selection
            selected_keys = selection_state.get_selected_chunk_keys(
                plot_df, for_highlighting=True
            )

            current_title = title
            if show_gaps:
                # Calculate gap statistics
                if not gaps_df.empty:
                    total_gap_bytes = gaps_df["length"].sum()
                    n_gaps = len(gaps_df)
                    current_title = f"Gaps/Metadata ({n_gaps} regions, {total_gap_bytes / (1024*1024):.1f} MB)"
                else:
                    current_title = "Gaps/Metadata (no gaps found)"

            plot = _create_segments_plot(
                hv,
                seg_df,
                width,
                height,
                current_title,
                yticks,
                selected_chunk_keys=selected_keys
                if selected_keys and not show_gaps
                else None,
                show_gaps=show_gaps,
                gaps_df=gaps_df,
            )
            chart_container.append(pn.pane.HoloViews(plot))

        # Watch toggle changes
        view_toggle.param.watch(update_chart, "value")

        # Watch selection state changes
        selection_state.param.watch(update_chart, ["bounds", "selected_variables"])

        # Initial render
        update_chart()

        return pn.Column(view_toggle, chart_container)

    # Simple version without toggle
    def render_with_selection(
        bounds: tuple | None, selected_variables: list | None
    ) -> Any:
        """Render segments with highlight for selected chunks."""
        # Only highlight chunks from bounds selection, not variable selection
        selected_keys = selection_state.get_selected_chunk_keys(
            plot_df, for_highlighting=True
        )

        return _create_segments_plot(
            hv,
            seg_df,
            width,
            height,
            title,
            yticks,
            selected_chunk_keys=selected_keys if selected_keys else None,
        )

    # Create stream for selection state changes
    param_stream = hv.streams.Params(selection_state, ["bounds", "selected_variables"])

    # Create DynamicMap that reacts to selection changes
    dmap = hv.DynamicMap(render_with_selection, streams=[param_stream])

    return pn.pane.HoloViews(dmap)


def _byte_range_matplotlib(
    df: pd.DataFrame,
    width: int,
    height: int,
    title: str,
    color_by: str,
    show_gaps: bool,
) -> Any:
    """Create byte range chart using matplotlib."""
    from vzviz._compat import import_matplotlib

    plt = import_matplotlib()

    fig_width = width / 100
    fig_height = height / 100

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_labels = df[["y_pos", "y_label"]].drop_duplicates().sort_values("y_pos")

    bar_height = 0.6
    for _, row in df.iterrows():
        ax.barh(
            y=row["y_pos"],
            width=row["length"],
            left=row["offset"],
            height=bar_height,
            color=row["color"],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_yticks(y_labels["y_pos"])
    ax.set_yticklabels(y_labels["y_label"])
    ax.invert_yaxis()

    ax.set_xlabel("Offset (MB)")
    ax.set_title(title)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_bytes(int(x))))

    plt.tight_layout()
    return fig
