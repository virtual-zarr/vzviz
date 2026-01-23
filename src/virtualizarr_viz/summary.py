"""Summary statistics for chunk manifests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from virtualizarr_viz.core import ManifestLike, extract_manifest, manifest_to_dataframe
from virtualizarr_viz.utils import format_bytes

if TYPE_CHECKING:
    pass


def manifest_summary(
    data: ManifestLike,
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Generate overall summary statistics for a chunk manifest.

    Parameters
    ----------
    data : ManifestLike
        Input manifest data (ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray).
    variable : str, optional
        Variable name if data is an xr.Dataset.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        - total_chunks: int
        - chunk_grid_shape: str
        - ndim: int
        - unique_files: int
        - chunks_per_file_mean: float
        - chunks_per_file_min: int
        - chunks_per_file_max: int
        - chunk_bytes_mean: float
        - chunk_bytes_min: int
        - chunk_bytes_max: int
        - total_bytes: int
        - total_bytes_human: str
        - missing_chunks: int (chunks with empty path)

    Examples
    --------
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import manifest_summary
    >>> vds = open_virtual_dataset("data.nc")
    >>> summary = manifest_summary(vds, "temperature")
    >>> print(summary.T)  # Transpose for readability
    """
    manifest = extract_manifest(data, variable)
    df = manifest_to_dataframe(data, variable)

    if df.empty:
        return pd.DataFrame([{
            "total_chunks": 0,
            "chunk_grid_shape": str(manifest.shape_chunk_grid),
            "ndim": manifest.ndim_chunk_grid,
            "unique_files": 0,
            "chunks_per_file_mean": 0,
            "chunks_per_file_min": 0,
            "chunks_per_file_max": 0,
            "chunk_bytes_mean": 0,
            "chunk_bytes_min": 0,
            "chunk_bytes_max": 0,
            "total_bytes": 0,
            "total_bytes_human": "0 B",
            "missing_chunks": 0,
        }])

    # Calculate per-file chunk counts
    chunks_per_file = df.groupby("path").size()

    # Calculate statistics
    total_bytes = df["length"].sum()

    summary = {
        "total_chunks": len(df),
        "chunk_grid_shape": str(manifest.shape_chunk_grid),
        "ndim": manifest.ndim_chunk_grid,
        "unique_files": df["path"].nunique(),
        "chunks_per_file_mean": chunks_per_file.mean(),
        "chunks_per_file_min": chunks_per_file.min(),
        "chunks_per_file_max": chunks_per_file.max(),
        "chunk_bytes_mean": df["length"].mean(),
        "chunk_bytes_min": df["length"].min(),
        "chunk_bytes_max": df["length"].max(),
        "total_bytes": total_bytes,
        "total_bytes_human": format_bytes(total_bytes),
        "missing_chunks": 0,  # Empty paths are already filtered in manifest.dict()
    }

    return pd.DataFrame([summary])


def file_summary(
    data: ManifestLike,
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Generate per-file summary statistics.

    Parameters
    ----------
    data : ManifestLike
        Input manifest data (ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray).
    variable : str, optional
        Variable name if data is an xr.Dataset.

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
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import file_summary
    >>> vds = open_virtual_dataset("data.nc")
    >>> files = file_summary(vds, "temperature")
    >>> print(files)
    """
    df = manifest_to_dataframe(data, variable)

    if df.empty:
        return pd.DataFrame(columns=[
            "filename", "full_path", "chunk_count", "total_bytes",
            "total_bytes_human", "min_offset", "max_end_offset",
            "byte_range", "is_contiguous"
        ])

    # Group by file
    file_stats = df.groupby("path").agg(
        filename=("filename", "first"),
        chunk_count=("chunk_key", "count"),
        total_bytes=("length", "sum"),
        min_offset=("offset", "min"),
        max_end_offset=("end_offset", "max"),
    ).reset_index()

    # Rename path column
    file_stats = file_stats.rename(columns={"path": "full_path"})

    # Add human-readable bytes
    file_stats["total_bytes_human"] = file_stats["total_bytes"].apply(format_bytes)

    # Add byte range string
    file_stats["byte_range"] = file_stats.apply(
        lambda row: f"{row['min_offset']:,} - {row['max_end_offset']:,}",
        axis=1
    )

    # Check contiguity (no gaps between chunks)
    def check_contiguous(path: str) -> bool:
        file_df = df[df["path"] == path].sort_values("offset")
        if len(file_df) <= 1:
            return True
        # Check if each chunk starts where the previous one ends
        ends = file_df["end_offset"].values[:-1]
        starts = file_df["offset"].values[1:]
        return all(ends == starts)

    file_stats["is_contiguous"] = file_stats["full_path"].apply(check_contiguous)

    # Reorder columns
    column_order = [
        "filename", "full_path", "chunk_count", "total_bytes",
        "total_bytes_human", "min_offset", "max_end_offset",
        "byte_range", "is_contiguous"
    ]
    file_stats = file_stats[column_order]

    return file_stats.sort_values("chunk_count", ascending=False).reset_index(drop=True)
