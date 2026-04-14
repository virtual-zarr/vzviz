"""Storage diagnostics for ManifestStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vzviz.core import manifest_to_dataframe
from vzviz.utils import format_bytes

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestArray, ManifestGroup, ManifestStore


def chunk_size_distribution(
    store: "ManifestStore",
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Get per-chunk size data for histogram visualization.

    Returns a DataFrame with one row per chunk, containing the stored
    (compressed) length and the theoretical uncompressed length.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to analyze.
    variable : str, optional
        If provided, only include chunks for this variable.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: variable, chunk_key, length, uncompressed_length.
    """
    records: list[dict] = []

    def process_group(group: "ManifestGroup", group_path: str = "") -> None:
        for array_name, array in group.arrays.items():
            var_path = f"{group_path}/{array_name}" if group_path else array_name
            if variable is not None and var_path != variable:
                continue
            _collect_chunk_sizes(array, var_path, records)

        for group_name, subgroup in group.groups.items():
            sub_path = f"{group_path}/{group_name}" if group_path else group_name
            process_group(subgroup, sub_path)

    process_group(store._group)

    if not records:
        return pd.DataFrame(
            columns=["variable", "chunk_key", "length", "uncompressed_length"]
        )

    return pd.DataFrame(records)


def _collect_chunk_sizes(
    array: "ManifestArray", var_path: str, records: list[dict]
) -> None:
    """Collect chunk sizes for a single array."""
    chunks = array.chunks
    shape = array.shape
    dtype = array.dtype
    itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 8

    for chunk_key, entry in array.manifest.dict().items():
        # Calculate uncompressed size for this specific chunk
        # (edge chunks may be smaller)
        if chunk_key and chunk_key != "c":
            indices = [int(i) for i in chunk_key.split(".") if i]
            chunk_cells = 1
            for dim_idx, idx in enumerate(indices):
                if dim_idx < len(chunks) and dim_idx < len(shape):
                    chunk_start = idx * chunks[dim_idx]
                    chunk_end = min(chunk_start + chunks[dim_idx], shape[dim_idx])
                    chunk_cells *= chunk_end - chunk_start
        else:
            chunk_cells = int(np.prod(chunks)) if chunks else 1

        records.append(
            {
                "variable": var_path,
                "chunk_key": chunk_key,
                "length": entry["length"],
                "uncompressed_length": chunk_cells * itemsize,
            }
        )


def gap_analysis(
    store: "ManifestStore",
    variable: str | None = None,
    per_variable: bool = False,
) -> pd.DataFrame:
    """
    Analyze gaps between chunks within each file.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to analyze.
    variable : str, optional
        If provided, only analyze chunks for this variable.
    per_variable : bool
        If True, compute gaps per (file, variable) pair instead of per file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: path, filename, n_chunks, data_bytes,
        gap_bytes, gap_bytes_human, n_gaps, largest_gap.
        If per_variable=True, also includes: variable.
    """
    df = manifest_to_dataframe(store, variable)

    if df.empty:
        cols = [
            "path",
            "filename",
            "n_chunks",
            "data_bytes",
            "gap_bytes",
            "gap_bytes_human",
            "n_gaps",
            "largest_gap",
        ]
        if per_variable:
            cols.insert(0, "variable")
        return pd.DataFrame(columns=cols)

    group_cols = ["path", "variable"] if per_variable else ["path"]
    records = []

    for group_key, group_df in df.groupby(group_cols):
        sorted_df = group_df.sort_values("offset")
        offsets = sorted_df["offset"].values
        end_offsets = sorted_df["end_offset"].values

        gaps = []
        for i in range(len(offsets) - 1):
            gap = int(offsets[i + 1] - end_offsets[i])
            if gap > 0:
                gaps.append(gap)

        total_gap = sum(gaps)
        data_bytes = int(sorted_df["length"].sum())

        record = {
            "path": group_key[0] if per_variable else group_key,
            "filename": sorted_df.iloc[0]["filename"],
            "n_chunks": len(sorted_df),
            "data_bytes": data_bytes,
            "gap_bytes": total_gap,
            "gap_bytes_human": format_bytes(total_gap),
            "n_gaps": len(gaps),
            "largest_gap": max(gaps) if gaps else 0,
        }
        if per_variable:
            record["variable"] = group_key[1]

        records.append(record)

    return pd.DataFrame(records)


def chunk_size_histogram_plot(
    store: "ManifestStore",
    variable: str | None = None,
    width: int = 800,
    height: int = 300,
) -> Any:
    """
    Create a histogram of chunk sizes (stored bytes).

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to analyze.
    variable : str, optional
        If provided, only include chunks for this variable.
    width : int
        Plot width in pixels.
    height : int
        Plot height in pixels.

    Returns
    -------
    holoviews.Element
        Histogram plot.
    """
    import holoviews as hv

    df = chunk_size_distribution(store, variable)
    if df.empty:
        return hv.Div("<p>No chunk data available</p>")

    lengths_mb = df["length"].values / (1024 * 1024)
    frequencies, edges = np.histogram(lengths_mb, bins=min(50, len(lengths_mb)))

    histogram = hv.Histogram((edges, frequencies)).opts(
        width=width,
        height=height,
        xlabel="Chunk Size (MB)",
        ylabel="Count",
        title="Chunk Size Distribution",
        tools=["hover"],
        color="#1f77b4",
    )

    return histogram
