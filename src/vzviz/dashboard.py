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
        Include ByteMap.
    show_heatmap : bool
        Include chunk-to-file heatmap. If variable is specified, shows that variable.
        Otherwise, shows a reactive heatmap that updates based on table selection.
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
    import holoviews as hv
    import panel as pn

    hv.extension("bokeh")

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
            pn.pane.Markdown(
                "*Select rows to highlight in ByteMap. "
                "Select one variable to view its heatmap.*"
            )
        )
        overview_df = variables_overview(store)
        if len(overview_df) > 0:
            display_cols = [
                "variable",
                "shape",
                "chunks",
                "dtype",
                "codecs",
                "total_chunks",
                "chunk_bytes_human",
                "total_bytes_human",
            ]
            display_df = overview_df[
                [c for c in display_cols if c in overview_df.columns]
            ]

            if interactive and selection_state is not None:
                # Get variable colors matching the ByteMap
                from vzviz.utils import get_variable_color_map

                var_colors = get_variable_color_map(display_df["variable"].tolist())

                # Create row background color style function
                def row_style(row):
                    color = var_colors.get(row["variable"], "#ffffff")
                    # Use lighter version for background (add alpha)
                    return [f"background-color: {color}40"] * len(row)

                styled_df = display_df.style.apply(row_style, axis=1)

                # Use Tabulator for interactive row selection
                variables_table = pn.widgets.Tabulator(
                    styled_df,
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
                        # Use selected_dataframe to handle sorted/filtered views
                        selected_df = variables_table.selected_dataframe
                        if not selected_df.empty and "variable" in selected_df.columns:
                            selected_vars = selected_df["variable"].tolist()
                            selection_state.set_selected_variables(selected_vars)
                        else:
                            selection_state.set_selected_variables([])
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
        components.append(pn.pane.Markdown("### ByteMap"))
        try:
            if interactive and selection_state is not None:
                byterange_component = byte_range_chart_interactive(
                    store,
                    variable,
                    selection_state=selection_state,
                    width=800,
                    include_toggle=True,  # Enable chunks/gaps toggle
                )
                components.append(byterange_component)
            else:
                byterange_plot = byte_range_chart(
                    store,
                    variable,
                    width=800,
                )
                components.append(pn.pane.HoloViews(byterange_plot))
        except Exception as e:
            components.append(pn.pane.Markdown(f"*Error creating ByteMap: {e}*"))

    # Heatmap (reactive to variable selection or fixed variable)
    if show_heatmap:
        from vzviz.core import get_array

        if variable is not None:
            # Fixed variable mode - show specific variable
            try:
                array = get_array(store, variable)
                ndim = len(array.shape)
                if ndim >= 1:
                    components.append(pn.pane.Markdown(f"### ChunkMap: {variable}"))
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
                            width=600,
                            height=400,
                        )
                        components.append(pn.pane.HoloViews(heatmap_plot))
            except Exception as e:
                components.append(pn.pane.Markdown(f"*Error creating heatmap: {e}*"))
        elif interactive and selection_state is not None:
            # Reactive mode - heatmap updates based on selected variable
            components.append(pn.pane.Markdown("### ChunkMap"))
            components.append(
                pn.pane.Markdown(
                    "*Select a single variable above to view its chunk heatmap*"
                )
            )

            # Container for reactive heatmap
            heatmap_container = pn.Column()

            def update_heatmap(selected_variables):
                """Update heatmap when variable selection changes."""
                heatmap_container.clear()

                if not selected_variables:
                    heatmap_container.append(pn.pane.Markdown("*No variable selected*"))
                    return

                if len(selected_variables) > 1:
                    heatmap_container.append(
                        pn.pane.Markdown(
                            f"*{len(selected_variables)} variables selected - "
                            "select exactly one to view heatmap*"
                        )
                    )
                    return

                selected_var = selected_variables[0]
                try:
                    array = get_array(store, selected_var)
                    ndim = len(array.shape)
                    if ndim >= 1:
                        heatmap_component = chunk_file_heatmap_interactive(
                            store,
                            selected_var,
                            selection_state=selection_state,
                            width=600,
                            height=400,
                        )
                        heatmap_container.append(
                            pn.pane.Markdown(f"**Variable:** {selected_var}")
                        )
                        heatmap_container.append(heatmap_component)
                    else:
                        heatmap_container.append(
                            pn.pane.Markdown(
                                f"*Cannot create heatmap for scalar variable {selected_var}*"
                            )
                        )
                except Exception as e:
                    heatmap_container.append(
                        pn.pane.Markdown(
                            f"*Error creating heatmap for {selected_var}: {e}*"
                        )
                    )

            # Watch for variable selection changes
            selection_state.param.watch(
                lambda event: update_heatmap(event.new), "selected_variables"
            )

            # Initial render
            update_heatmap(selection_state.selected_variables)

            components.append(heatmap_container)

    # Selection info panel (only in interactive mode)
    if interactive and selection_state is not None:
        components.append(pn.pane.Markdown("### Selection"))

        # Clear selection button
        clear_btn = pn.widgets.Button(
            name="Clear Selection", button_type="light", width=150
        )

        def on_clear_selection(event):
            selection_state.clear_selection()
            if variables_table is not None:
                variables_table.selection = []

        clear_btn.on_click(on_clear_selection)
        components.append(clear_btn)

        # Get all chunks for selection calculations
        all_chunks_df = manifest_to_dataframe(store, None)

        # Use ParamFunction for reliable reactive updates
        @pn.depends(
            selection_state.param.selected_variables,
            selection_state.param.bounds,
        )
        def selection_info_panel(selected_vars, bounds):
            info_text = _format_selection_info(selection_state, all_chunks_df, store)
            return pn.pane.Markdown(info_text)

        components.append(pn.panel(selection_info_panel))

    return pn.Column(*components)


def _format_selection_info(selection_state: Any, df: Any, store: Any = None) -> str:
    """Format selection information for display."""
    import pandas as pd

    from vzviz.utils import format_bytes

    if not selection_state.has_selection:
        return "*Click rows in the table or use box select in the ChunkMap to select chunks.*"

    lines = []

    # Section 1: Variable selection from table
    if selection_state.selected_variables:
        var_list = ", ".join(selection_state.selected_variables[:5])
        if len(selection_state.selected_variables) > 5:
            var_list += f" (+{len(selection_state.selected_variables) - 5} more)"

        # Get stats for variable selection
        var_mask = df["variable"].isin(selection_state.selected_variables)
        var_df = df[var_mask]
        var_chunks = len(var_df)
        var_bytes = var_df["length"].sum()

        # Check contiguity per variable
        contiguity = {}
        for var_name in selection_state.selected_variables:
            vdf = var_df[var_df["variable"] == var_name].sort_values("offset")
            if len(vdf) <= 1:
                contiguity[var_name] = True
            else:
                ends = vdf["end_offset"].values[:-1]
                starts = vdf["offset"].values[1:]
                contiguity[var_name] = bool((ends == starts).all())

        lines.append("**From Table Selection:**")
        lines.append(f"- Variables: {var_list}")
        lines.append(f"- Total Chunks: {var_chunks:,}")
        lines.append(f"- Total Size: {format_bytes(int(var_bytes))}")
        for var_name in selection_state.selected_variables[:5]:
            is_contig = contiguity.get(var_name, False)
            label = "contiguous" if is_contig else "non-contiguous"
            lines.append(f"  - {var_name}: {label}")

    # Section 2: Region selection from ChunkMap
    if (
        selection_state.bounds is not None
        and selection_state.bounds_variable is not None
    ):
        x_min, y_min, x_max, y_max = selection_state.bounds

        # Get stats for region selection (highlighted chunks)
        region_keys = selection_state.get_selected_chunk_keys(df, for_highlighting=True)
        region_mask = pd.Series(
            [
                (var, ck) in region_keys
                for var, ck in zip(df["variable"], df["chunk_key"])
            ],
            index=df.index,
        )
        region_df = df[region_mask]
        region_chunks = len(region_df)
        region_bytes = region_df["length"].sum()
        region_files = region_df["path"].nunique()

        if lines:
            lines.append("")

        lines.append(
            f"**From ChunkMap Selection** ({selection_state.bounds_variable}):"
        )
        lines.append(
            f"- Region: ({int(round(x_min))}, {int(round(y_min))}) to ({int(round(x_max))}, {int(round(y_max))})"
        )
        lines.append(f"- Chunks: {region_chunks:,}")
        lines.append(f"- Size: {format_bytes(int(region_bytes))}")
        lines.append(f"- Files: {region_files}")

        # Add performance metrics
        if store is not None:
            try:
                from vzviz.query import metrics_from_selection

                metrics = metrics_from_selection(
                    store,
                    selection_state.bounds_variable,
                    selection_state.bounds,
                    selection_state.dim_x,
                    selection_state.dim_y,
                    selection_state.chunk_shape,
                    selection_state.bounds_in_array_space,
                )

                lines.append("")
                lines.append("**Performance Metrics:**")
                lines.append(f"- Array Elements Requested: {metrics.requested_cells:,}")
                lines.append(f"- Array Elements Read: {metrics.cells_read:,}")
                lines.append(f"- Read Amplification: {metrics.read_amplification:.2f}x")
                lines.append(f"- Read Efficiency: {metrics.read_efficiency:.1f}%")
                lines.append(
                    f"- Chunks Touched: {metrics.chunks_touched:,} / {metrics.total_chunks:,}"
                )
                lines.append(f"- Range Reads: {metrics.range_reads:,}")
                lines.append(f"- Coalescing Factor: {metrics.coalescing_factor:.2f}x")
                lines.append(f"- Storage Alignment: {metrics.storage_alignment:.2f}")
            except Exception:
                # Silently skip metrics if they can't be computed
                pass

    if not lines:
        return "*No chunks in selection.*"

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
