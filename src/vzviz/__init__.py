"""Visualization tools for VirtualiZarr ManifestStore."""

from __future__ import annotations

from vzviz.byterange import byte_range_chart, byte_range_chart_interactive
from vzviz.core import (
    get_array,
    get_store_info,
    list_variables,
    manifest_to_dataframe,
)
from vzviz.dashboard import manifest_dashboard
from vzviz.diagnostics import (
    chunk_size_distribution,
    chunk_size_histogram_plot,
    gap_analysis,
)
from vzviz.heatmap import chunk_file_heatmap, chunk_file_heatmap_interactive
from vzviz.io import (
    load_manifest_from_json,
    manifeststore_to_kerchunk_refs,
    save_manifest_to_json,
)
from vzviz.query import (
    QueryMetrics,
    compare_queries,
    metrics_from_selection,
    simulate_query,
)
from vzviz.selection import SelectionState
from vzviz.summary import file_summary, manifest_summary
from vzviz.variables import (
    ChunkGridInfo,
    DimensionInfo,
    chunk_grid_info,
    variables_overview,
)

try:
    from vzviz._version import __version__
except ImportError:
    __version__ = "unknown"


__all__ = [
    # Core utilities
    "manifest_to_dataframe",
    "get_array",
    "get_store_info",
    "list_variables",
    # Variable overview
    "variables_overview",
    "chunk_grid_info",
    "ChunkGridInfo",
    "DimensionInfo",
    # Query simulation (vischunk-like)
    "simulate_query",
    "compare_queries",
    "metrics_from_selection",
    "QueryMetrics",
    # Storage diagnostics
    "chunk_size_distribution",
    "chunk_size_histogram_plot",
    "gap_analysis",
    # Selection state for interactive visualizations
    "SelectionState",
    # Visualizations
    "byte_range_chart",
    "byte_range_chart_interactive",
    "chunk_file_heatmap",
    "chunk_file_heatmap_interactive",
    "manifest_summary",
    "file_summary",
    "manifest_dashboard",
    # IO (Kerchunk JSON round-trip)
    "load_manifest_from_json",
    "save_manifest_to_json",
    "manifeststore_to_kerchunk_refs",
    # Version
    "__version__",
]
