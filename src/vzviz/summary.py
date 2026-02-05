"""Summary statistics for ManifestStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from vzviz.core import get_array, manifest_to_dataframe
from vzviz.utils import format_bytes

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestStore


def manifest_summary(
    store: "ManifestStore",
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Generate summary statistics for a ManifestStore or specific variable.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to summarize.
    variable : str, optional
        If provided, summarize only this variable.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        - total_chunks: int
        - chunk_grid_shape: str (if single variable)
        - ndim: int (if single variable)
        - unique_files: int
        - chunks_per_file_mean: float
        - chunks_per_file_min: int
        - chunks_per_file_max: int
        - chunk_bytes_mean: float
        - chunk_bytes_min: int
        - chunk_bytes_max: int
        - total_bytes: int
        - total_bytes_human: str

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> summary = manifest_summary(store)
    >>> print(summary.T)

    >>> # For a specific variable
    >>> summary = manifest_summary(store, "temperature")
    """
    df = manifest_to_dataframe(store, variable)

    if df.empty:
        return pd.DataFrame(
            [
                {
                    "total_chunks": 0,
                    "unique_files": 0,
                    "chunks_per_file_mean": 0,
                    "chunks_per_file_min": 0,
                    "chunks_per_file_max": 0,
                    "chunk_bytes_mean": 0,
                    "chunk_bytes_min": 0,
                    "chunk_bytes_max": 0,
                    "total_bytes": 0,
                    "total_bytes_human": "0 B",
                }
            ]
        )

    # Calculate per-file chunk counts
    chunks_per_file = df.groupby("path").size()
    total_bytes = df["length"].sum()

    summary = {
        "total_chunks": len(df),
        "unique_files": df["path"].nunique(),
        "chunks_per_file_mean": chunks_per_file.mean(),
        "chunks_per_file_min": chunks_per_file.min(),
        "chunks_per_file_max": chunks_per_file.max(),
        "chunk_bytes_mean": df["length"].mean(),
        "chunk_bytes_min": df["length"].min(),
        "chunk_bytes_max": df["length"].max(),
        "total_bytes": total_bytes,
        "total_bytes_human": format_bytes(total_bytes),
    }

    # Add array-specific info if single variable
    if variable is not None:
        array = get_array(store, variable)
        summary["chunk_grid_shape"] = str(array.manifest.shape_chunk_grid)
        summary["ndim"] = array.manifest.ndim_chunk_grid

    return pd.DataFrame([summary])


def file_summary(
    store: "ManifestStore",
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Generate per-file summary statistics.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to summarize.
    variable : str, optional
        If provided, summarize only chunks from this variable.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per file, columns:
        - filename: str (basename)
        - full_path: str (complete URI/path)
        - chunk_count: int
        - total_bytes: int
        - total_bytes_human: str
        - min_offset: int
        - max_end_offset: int
        - byte_range: str (formatted as "min-max")
        - is_contiguous: bool (True if no gaps between chunks)

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> files = file_summary(store)
    >>> print(files)
    """
    df = manifest_to_dataframe(store, variable)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "filename",
                "full_path",
                "chunk_count",
                "total_bytes",
                "total_bytes_human",
                "min_offset",
                "max_end_offset",
                "byte_range",
                "is_contiguous",
            ]
        )

    # Group by file
    file_stats = (
        df.groupby("path")
        .agg(
            filename=("filename", "first"),
            chunk_count=("chunk_key", "count"),
            total_bytes=("length", "sum"),
            min_offset=("offset", "min"),
            max_end_offset=("end_offset", "max"),
        )
        .reset_index()
    )

    file_stats = file_stats.rename(columns={"path": "full_path"})
    file_stats["total_bytes_human"] = file_stats["total_bytes"].apply(format_bytes)
    file_stats["byte_range"] = file_stats.apply(
        lambda row: f"{row['min_offset']:,} - {row['max_end_offset']:,}", axis=1
    )

    # Check contiguity
    def check_contiguous(path: str) -> bool:
        file_df = df[df["path"] == path].sort_values("offset")
        if len(file_df) <= 1:
            return True
        ends = file_df["end_offset"].values[:-1]
        starts = file_df["offset"].values[1:]
        return all(ends == starts)

    file_stats["is_contiguous"] = file_stats["full_path"].apply(check_contiguous)

    column_order = [
        "filename",
        "full_path",
        "chunk_count",
        "total_bytes",
        "total_bytes_human",
        "min_offset",
        "max_end_offset",
        "byte_range",
        "is_contiguous",
    ]
    file_stats = file_stats[column_order]

    return file_stats.sort_values("chunk_count", ascending=False).reset_index(drop=True)
