"""Visualization tools for VirtualiZarr chunk manifests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from virtualizarr_viz.byterange import byte_range_chart
from virtualizarr_viz.core import (
    ManifestLike,
    extract_manifest,
    get_manifest_info,
    manifest_to_dataframe,
)
from virtualizarr_viz.dashboard import manifest_dashboard
from virtualizarr_viz.heatmap import chunk_file_heatmap
from virtualizarr_viz.summary import file_summary, manifest_summary

if TYPE_CHECKING:
    import pandas as pd

try:
    from virtualizarr_viz._version import __version__
except ImportError:
    __version__ = "unknown"


def visualize_manifest(
    data: ManifestLike,
    variable: str | None = None,
    kind: Literal["byterange", "heatmap", "summary", "dashboard"] = "byterange",
    **kwargs: Any,
) -> Any:
    """
    Unified entry point for manifest visualization.

    Parameters
    ----------
    data : ManifestLike
        Input manifest data (ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray).
    variable : str, optional
        Variable name if data is an xr.Dataset.
    kind : {"byterange", "heatmap", "summary", "dashboard"}
        Type of visualization to create:
        - "byterange": Byte range chart showing chunk positions in files (default)
        - "heatmap": 2D heatmap of chunk-to-file mapping
        - "summary": Summary statistics as DataFrame
        - "dashboard": Interactive dashboard with all visualizations
    **kwargs
        Additional arguments passed to the specific visualization function.

    Returns
    -------
    Plot, DataFrame, or panel.Pane
        The visualization object (type depends on `kind` and `backend`).

    Examples
    --------
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import visualize_manifest
    >>> vds = open_virtual_dataset("data.nc")
    >>> visualize_manifest(vds, "temperature", kind="byterange")
    >>> visualize_manifest(vds, "temperature", kind="heatmap")
    >>> visualize_manifest(vds, "temperature", kind="summary")
    """
    funcs = {
        "byterange": byte_range_chart,
        "heatmap": chunk_file_heatmap,
        "summary": manifest_summary,
        "dashboard": manifest_dashboard,
    }

    if kind not in funcs:
        raise ValueError(
            f"Unknown visualization kind: {kind}. "
            f"Choose from: {list(funcs.keys())}"
        )

    return funcs[kind](data, variable=variable, **kwargs)


__all__ = [
    # Main entry point
    "visualize_manifest",
    # Individual visualizations
    "byte_range_chart",
    "chunk_file_heatmap",
    "manifest_summary",
    "file_summary",
    "manifest_dashboard",
    # Core utilities
    "extract_manifest",
    "manifest_to_dataframe",
    "get_manifest_info",
    # Type alias
    "ManifestLike",
    # Version
    "__version__",
]
