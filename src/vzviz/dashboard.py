"""Interactive dashboard combining all visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vzviz.byterange import byte_range_chart, byte_range_chart_interactive
from vzviz.core import get_store_info, manifest_to_dataframe
from vzviz.heatmap import chunk_file_heatmap, chunk_file_heatmap_interactive
from vzviz.summary import file_summary, manifest_summary
from vzviz.variables import variables_overview

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestStore


def manifest_dashboard(
    store: "ManifestStore",
    variable: str | None = None,
    interactive: bool = True,
    show_overview: bool = True,
    show_byterange: bool = True,
    show_heatmap: bool = True,
    show_summary: bool = True,
) -> Any:
    """
    Create an interactive dashboard combining all visualizations.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to visualize.
    variable : str, optional
        If provided, focus visualizations on this variable.
        If None, shows an overview of all variables.
    interactive : bool
        If True (default), enables cross-panel selection synchronization.
        Clicking a chunk in one panel highlights it in all panels.
    show_overview : bool
        Include variables overview table.
    show_byterange : bool
        Include byte range chart.
    show_heatmap : bool
        Include chunk-to-file heatmap (requires variable to be specified).
    show_summary : bool
        Include summary statistics table.

    Returns
    -------
    panel.Column
        Interactive dashboard that can be displayed in Jupyter or served.

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> dashboard = manifest_dashboard(store)
    >>> dashboard.show()  # Opens in browser
    """
    from vzviz._compat import import_holoviews, import_panel

    pn = import_panel()
    hv = import_holoviews()  # noqa

    # Create shared selection state if interactive mode is enabled
    selection_state = None
    if interactive:
        from vzviz.selection import SelectionState

        selection_state = SelectionState()

    info = get_store_info(store)
    components = []

    # Title
    title_text = "## ManifestStore Visualization"
    subtitle_text = (
        f"**Variables:** {info['n_variables']} | "
        f"**Groups:** {info['n_groups']} | "
        f"**Total chunks:** {info['total_chunks']} | "
        f"**Files:** {info['unique_files']}"
    )
    components.append(pn.pane.Markdown(title_text))
    components.append(pn.pane.Markdown(subtitle_text))

    # Variables overview
    variables_table = None
    if show_overview:
        components.append(pn.pane.Markdown("### Variables Overview"))
        components.append(
            pn.pane.Markdown("*Click rows to highlight in byte range chart*")
        )
        overview_df = variables_overview(store)
        if len(overview_df) > 0:
            display_cols = [
                "variable",
                "shape",
                "chunks",
                "dtype",
                "total_chunks",
                "chunk_bytes_human",
                "total_bytes_human",
            ]
            display_df = overview_df[
                [c for c in display_cols if c in overview_df.columns]
            ]

            if interactive and selection_state is not None:
                # Use Tabulator for interactive row selection
                variables_table = pn.widgets.Tabulator(
                    display_df,
                    width=900,
                    height=min(400, 50 + len(display_df) * 30),
                    selectable="checkbox",
                    show_index=False,
                    configuration={"columnDefaults": {"headerSort": True}},
                )

                # Connect row selection to selection state
                def on_selection_change(event):
                    selected_indices = variables_table.selection
                    if selected_indices:
                        selected_vars = display_df.iloc[selected_indices][
                            "variable"
                        ].tolist()
                        selection_state.set_selected_variables(selected_vars)
                    else:
                        selection_state.set_selected_variables([])

                variables_table.param.watch(on_selection_change, "selection")
                components.append(variables_table)
            else:
                components.append(pn.pane.DataFrame(display_df, width=900))
        else:
            components.append(pn.pane.Markdown("*No variables found*"))

    # Summary statistics
    if show_summary:
        components.append(pn.pane.Markdown("### Summary Statistics"))
        summary_df = manifest_summary(store, variable)
        summary_text = _format_summary(summary_df)
        components.append(pn.pane.Markdown(summary_text))

        files_df = file_summary(store, variable)
        if len(files_df) > 0:
            components.append(pn.pane.Markdown("### Per-File Statistics"))
            if len(files_df) > 20:
                components.append(
                    pn.pane.Markdown(f"*Showing top 20 of {len(files_df)} files*")
                )
                display_df = files_df.head(20)
            else:
                display_df = files_df

            display_cols = [
                "filename",
                "chunk_count",
                "total_bytes_human",
                "byte_range",
                "is_contiguous",
            ]
            display_df = display_df[
                [c for c in display_cols if c in display_df.columns]
            ]
            components.append(pn.pane.DataFrame(display_df, width=800))

    # Byte range chart
    if show_byterange:
        components.append(pn.pane.Markdown("### Byte Range Chart"))
        try:
            if interactive and selection_state is not None:
                byterange_component = byte_range_chart_interactive(
                    store,
                    variable,
                    selection_state=selection_state,
                    width=800,
                )
                components.append(byterange_component)
            else:
                byterange_plot = byte_range_chart(
                    store,
                    variable,
                    backend="holoviews",
                    width=800,
                )
                components.append(pn.pane.HoloViews(byterange_plot))
        except Exception as e:
            components.append(
                pn.pane.Markdown(f"*Error creating byte range chart: {e}*")
            )

    # Heatmap (requires specific variable)
    if show_heatmap and variable is not None:
        from vzviz.core import get_array

        try:
            array = get_array(store, variable)
            ndim = len(array.shape)
            if ndim >= 1:
                components.append(
                    pn.pane.Markdown(f"### Chunk-to-File Heatmap: {variable}")
                )
                if interactive and selection_state is not None:
                    heatmap_component = chunk_file_heatmap_interactive(
                        store,
                        variable,
                        selection_state=selection_state,
                        width=600,
                        height=400,
                    )
                    components.append(heatmap_component)
                else:
                    heatmap_plot = chunk_file_heatmap(
                        store,
                        variable,
                        backend="holoviews",
                        width=600,
                        height=400,
                    )
                    components.append(pn.pane.HoloViews(heatmap_plot))
        except Exception as e:
            components.append(pn.pane.Markdown(f"*Error creating heatmap: {e}*"))

    # Selection info panel (only in interactive mode)
    if interactive and selection_state is not None:
        components.append(pn.pane.Markdown("### Selection"))
        df = manifest_to_dataframe(store, variable)

        def get_selection_info(bounds, selected_variables):
            return _format_selection_info(selection_state, df)

        selection_info = pn.bind(
            get_selection_info,
            bounds=selection_state.param.bounds,
            selected_variables=selection_state.param.selected_variables,
        )
        components.append(pn.pane.Markdown(selection_info))

    return pn.Column(*components)


def _format_selection_info(selection_state: Any, df: Any) -> str:
    """Format selection information for display."""
    from vzviz.utils import format_bytes

    if not selection_state.has_selection:
        return "*Click rows in the table or use box select in the heatmap to select chunks.*"

    selected_keys = selection_state.get_selected_chunk_keys(df)

    if not selected_keys:
        return "*No chunks in selection.*"

    # Calculate stats for selected chunks
    selected_df = df[df["chunk_key"].isin(selected_keys)]
    n_chunks = len(selected_df)
    total_bytes = selected_df["length"].sum()
    n_files = selected_df["path"].nunique()

    lines = []

    # Show selected variables
    if selection_state.selected_variables:
        var_list = ", ".join(selection_state.selected_variables[:5])
        if len(selection_state.selected_variables) > 5:
            var_list += f" (+{len(selection_state.selected_variables) - 5} more)"
        lines.append(f"**Selected variables:** {var_list}")

    # Show bounds if set
    if selection_state.bounds is not None:
        x_min, y_min, x_max, y_max = selection_state.bounds
        lines.append(
            f"**Selected region:** ({int(round(x_min))}, {int(round(y_min))}) to ({int(round(x_max))}, {int(round(y_max))})"
        )

    lines.extend(
        [
            f"**Chunks selected:** {n_chunks:,}",
            f"**Total size:** {format_bytes(int(total_bytes))}",
            f"**Files spanned:** {n_files}",
        ]
    )

    return "  \n".join(lines)


def _format_summary(summary_df) -> str:
    """Format summary DataFrame as markdown text."""
    from vzviz.utils import format_bytes

    if summary_df.empty:
        return "*No data*"

    row = summary_df.iloc[0]

    lines = [
        f"- **Total chunks:** {row['total_chunks']:,}",
        f"- **Unique files:** {row['unique_files']}",
        f"- **Chunks per file:** {row['chunks_per_file_min']:.0f} - {row['chunks_per_file_max']:.0f} (mean: {row['chunks_per_file_mean']:.1f})",
        f"- **Chunk sizes:** {format_bytes(int(row['chunk_bytes_min']))} - {format_bytes(int(row['chunk_bytes_max']))} (mean: {format_bytes(int(row['chunk_bytes_mean']))})",
        f"- **Total data:** {row['total_bytes_human']}",
    ]

    if "chunk_grid_shape" in row:
        lines.insert(1, f"- **Chunk grid shape:** {row['chunk_grid_shape']}")

    return "\n".join(lines)
