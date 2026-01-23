"""Byte range chart visualization showing chunk positions within files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from virtualizarr_viz.core import ManifestLike, manifest_to_dataframe
from virtualizarr_viz.utils import format_bytes, get_colormap, truncate_path

if TYPE_CHECKING:
    pass


def byte_range_chart(
    data: ManifestLike,
    variable: str | None = None,
    max_files: int = 30,
    sort_by: Literal["name", "chunks", "bytes", "offset"] = "offset",
    color_by: Literal["chunk", "dim_0", "file", "none"] = "chunk",
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
    data : ManifestLike
        Input manifest data (ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray).
    variable : str, optional
        Variable name if data is an xr.Dataset.
    max_files : int
        Maximum number of files to display. If exceeded, shows the files with
        the most chunks.
    sort_by : {"name", "chunks", "bytes", "offset"}
        How to sort files:
        - "name": alphabetically by filename
        - "chunks": by number of chunks (descending)
        - "bytes": by total bytes (descending)
        - "offset": by minimum offset
    color_by : {"chunk", "dim_0", "file", "none"}
        How to color segments:
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
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import byte_range_chart
    >>> vds = open_virtual_dataset("data.nc")
    >>> byte_range_chart(vds, "temperature", sort_by="chunks")
    """
    # Convert to DataFrame
    df = manifest_to_dataframe(data, variable)

    if df.empty:
        raise ValueError("Manifest contains no chunks")

    # Prepare data for plotting
    plot_df = _prepare_byterange_data(df, max_files, sort_by, color_by)

    # Auto-calculate height
    n_files = plot_df["filename"].nunique()
    if height is None:
        height = max(200, min(800, 50 + n_files * 25))

    # Generate title
    if title is None:
        total_chunks = len(df)
        total_files = df["path"].nunique()
        title = f"Chunk Byte Ranges ({total_chunks} chunks across {total_files} files)"

    # Dispatch to backend
    if backend == "holoviews":
        return _byte_range_holoviews(plot_df, width, height, title, color_by, show_gaps)
    elif backend == "matplotlib":
        return _byte_range_matplotlib(plot_df, width, height, title, color_by, show_gaps)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'holoviews' or 'matplotlib'.")


def _prepare_byterange_data(
    df: pd.DataFrame,
    max_files: int,
    sort_by: str,
    color_by: str,
) -> pd.DataFrame:
    """Prepare DataFrame for byte range visualization."""
    # Calculate per-file statistics
    file_stats = df.groupby("path").agg(
        chunk_count=("chunk_key", "count"),
        total_bytes=("length", "sum"),
        min_offset=("offset", "min"),
        filename=("filename", "first"),
    ).reset_index()

    # Sort files
    if sort_by == "name":
        file_stats = file_stats.sort_values("filename")
    elif sort_by == "chunks":
        file_stats = file_stats.sort_values("chunk_count", ascending=False)
    elif sort_by == "bytes":
        file_stats = file_stats.sort_values("total_bytes", ascending=False)
    elif sort_by == "offset":
        file_stats = file_stats.sort_values("min_offset")

    # Limit to max_files
    if len(file_stats) > max_files:
        # Keep files with most chunks
        file_stats = file_stats.nlargest(max_files, "chunk_count")

    # Filter original dataframe to selected files
    selected_paths = set(file_stats["path"])
    plot_df = df[df["path"].isin(selected_paths)].copy()

    # Create y-axis labels (file display names)
    path_to_label = {}
    for _, row in file_stats.iterrows():
        label = truncate_path(row["filename"], 40)
        # Add chunk count info
        label = f"{label} ({row['chunk_count']} chunks)"
        path_to_label[row["path"]] = label

    plot_df["y_label"] = plot_df["path"].map(path_to_label)

    # Assign y positions based on sort order
    sorted_paths = file_stats["path"].tolist()
    path_to_y = {path: i for i, path in enumerate(sorted_paths)}
    plot_df["y_pos"] = plot_df["path"].map(path_to_y)

    # Assign colors
    if color_by == "chunk":
        n_chunks = len(plot_df)
        colors = get_colormap(n_chunks)
        plot_df["color"] = colors[: len(plot_df)]
    elif color_by == "dim_0" and "dim_0" in plot_df.columns:
        n_dims = plot_df["dim_0"].nunique()
        colors = get_colormap(n_dims)
        dim_to_color = {dim: colors[i] for i, dim in enumerate(sorted(plot_df["dim_0"].unique()))}
        plot_df["color"] = plot_df["dim_0"].map(dim_to_color)
    elif color_by == "file":
        n_files = plot_df["path"].nunique()
        colors = get_colormap(n_files)
        path_to_color = {path: colors[i] for i, path in enumerate(sorted_paths)}
        plot_df["color"] = plot_df["path"].map(path_to_color)
    else:  # "none"
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
    from virtualizarr_viz._compat import import_holoviews

    hv = import_holoviews()

    # Create segments data
    segments = []
    for _, row in df.iterrows():
        segments.append({
            "x0": row["offset"],
            "x1": row["end_offset"],
            "y": row["y_pos"],
            "y_label": row["y_label"],
            "color": row["color"],
            "chunk_key": row["chunk_key"],
            "path": row["path"],
            "offset": row["offset"],
            "length": row["length"],
            "filename": row["filename"],
        })

    seg_df = pd.DataFrame(segments)

    # Create the segments plot
    segments_plot = hv.Segments(
        seg_df,
        kdims=["x0", "y", "x1", "y"],
        vdims=["color", "chunk_key", "path", "offset", "length", "filename", "y_label"],
    ).opts(
        color="color",
        line_width=8,
        tools=["hover"],
        width=width,
        height=height,
        title=title,
        xlabel="Byte Offset",
        ylabel="",
        show_legend=False,
    )

    # Create y-axis ticks
    y_labels = df[["y_pos", "y_label"]].drop_duplicates().sort_values("y_pos")
    yticks = list(zip(y_labels["y_pos"], y_labels["y_label"]))

    segments_plot = segments_plot.opts(
        yticks=yticks,
        invert_yaxis=True,  # Put first file at top
    )

    return segments_plot


def _byte_range_matplotlib(
    df: pd.DataFrame,
    width: int,
    height: int,
    title: str,
    color_by: str,
    show_gaps: bool,
) -> Any:
    """Create byte range chart using matplotlib."""
    from virtualizarr_viz._compat import import_matplotlib

    plt = import_matplotlib()

    # Convert pixels to inches (assuming 100 dpi)
    fig_width = width / 100
    fig_height = height / 100

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Get unique files in order
    y_labels = df[["y_pos", "y_label"]].drop_duplicates().sort_values("y_pos")

    # Plot each segment as a horizontal bar
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

    # Set y-axis ticks and labels
    ax.set_yticks(y_labels["y_pos"])
    ax.set_yticklabels(y_labels["y_label"])
    ax.invert_yaxis()  # Put first file at top

    # Labels and title
    ax.set_xlabel("Byte Offset")
    ax.set_title(title)

    # Format x-axis with byte units
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: format_bytes(int(x)))
    )

    plt.tight_layout()
    return fig
