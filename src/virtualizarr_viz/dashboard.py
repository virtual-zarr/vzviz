"""Interactive dashboard combining all visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from virtualizarr_viz.byterange import byte_range_chart
from virtualizarr_viz.core import ManifestLike, extract_manifest, get_manifest_info
from virtualizarr_viz.heatmap import chunk_file_heatmap
from virtualizarr_viz.summary import file_summary, manifest_summary

if TYPE_CHECKING:
    pass


def manifest_dashboard(
    data: ManifestLike,
    variable: str | None = None,
    show_byterange: bool = True,
    show_heatmap: bool = True,
    show_summary: bool = True,
) -> Any:
    """
    Create an interactive dashboard combining all visualizations.

    Parameters
    ----------
    data : ManifestLike
        Input manifest data (ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray).
    variable : str, optional
        Variable name if data is an xr.Dataset.
    show_byterange : bool
        Include byte range chart.
    show_heatmap : bool
        Include chunk-to-file heatmap.
    show_summary : bool
        Include summary statistics table.

    Returns
    -------
    panel.Column
        Interactive dashboard that can be displayed in Jupyter or served.

    Examples
    --------
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import manifest_dashboard
    >>> vds = open_virtual_dataset("data.nc")
    >>> dashboard = manifest_dashboard(vds, "temperature")
    >>> dashboard.show()  # Opens in browser
    >>> dashboard  # Displays inline in Jupyter
    """
    from virtualizarr_viz._compat import import_holoviews, import_panel

    pn = import_panel()
    hv = import_holoviews()

    # Get manifest info for title
    info = get_manifest_info(data, variable)

    # Build components
    components = []

    # Title
    title_text = f"## Chunk Manifest Visualization"
    subtitle_text = (
        f"**Grid shape:** {info['shape_chunk_grid']} | "
        f"**Total chunks:** {info['total_chunks']} | "
        f"**Files:** {info['unique_files']}"
    )
    components.append(pn.pane.Markdown(title_text))
    components.append(pn.pane.Markdown(subtitle_text))

    # Summary statistics
    if show_summary:
        components.append(pn.pane.Markdown("### Summary Statistics"))

        summary_df = manifest_summary(data, variable)
        files_df = file_summary(data, variable)

        # Display summary as formatted text
        summary_text = _format_summary(summary_df)
        components.append(pn.pane.Markdown(summary_text))

        # File summary table
        if len(files_df) > 0:
            components.append(pn.pane.Markdown("### Per-File Statistics"))
            # Limit display for large file counts
            if len(files_df) > 20:
                components.append(pn.pane.Markdown(f"*Showing top 20 of {len(files_df)} files*"))
                display_df = files_df.head(20)
            else:
                display_df = files_df

            # Select columns for display
            display_cols = ["filename", "chunk_count", "total_bytes_human", "byte_range", "is_contiguous"]
            display_df = display_df[[c for c in display_cols if c in display_df.columns]]
            components.append(pn.pane.DataFrame(display_df, width=800))

    # Byte range chart
    if show_byterange:
        components.append(pn.pane.Markdown("### Byte Range Chart"))
        try:
            byterange_plot = byte_range_chart(
                data, variable,
                backend="holoviews",
                width=800,
            )
            components.append(pn.pane.HoloViews(byterange_plot))
        except Exception as e:
            components.append(pn.pane.Markdown(f"*Error creating byte range chart: {e}*"))

    # Heatmap
    if show_heatmap and info['ndim'] >= 1:
        components.append(pn.pane.Markdown("### Chunk-to-File Heatmap"))
        try:
            heatmap_plot = chunk_file_heatmap(
                data, variable,
                backend="holoviews",
                width=600,
                height=400,
            )
            components.append(pn.pane.HoloViews(heatmap_plot))
        except Exception as e:
            components.append(pn.pane.Markdown(f"*Error creating heatmap: {e}*"))

    return pn.Column(*components)


def _format_summary(summary_df) -> str:
    """Format summary DataFrame as markdown text."""
    if summary_df.empty:
        return "*No data*"

    row = summary_df.iloc[0]

    lines = [
        f"- **Total chunks:** {row['total_chunks']:,}",
        f"- **Chunk grid shape:** {row['chunk_grid_shape']}",
        f"- **Unique files:** {row['unique_files']}",
        f"- **Chunks per file:** {row['chunks_per_file_min']:.0f} - {row['chunks_per_file_max']:.0f} (mean: {row['chunks_per_file_mean']:.1f})",
        f"- **Chunk sizes:** {row['chunk_bytes_min']:,} - {row['chunk_bytes_max']:,} bytes (mean: {row['chunk_bytes_mean']:,.0f})",
        f"- **Total data:** {row['total_bytes_human']}",
    ]

    return "\n".join(lines)
