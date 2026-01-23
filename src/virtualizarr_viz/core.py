"""Core data extraction and transformation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

    from virtualizarr.manifests import ChunkManifest, ManifestArray

# Type alias for inputs that can be converted to a ChunkManifest
ManifestLike = Union[
    "ChunkManifest",
    "ManifestArray",
    "xr.Dataset",
    "xr.DataArray",
]


def extract_manifest(
    data: ManifestLike,
    variable: str | None = None,
) -> "ChunkManifest":
    """
    Extract ChunkManifest from various input types.

    Parameters
    ----------
    data : ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray
        Input data containing manifest information.
    variable : str, optional
        Variable name. Required if data is an xr.Dataset with multiple
        data variables.

    Returns
    -------
    ChunkManifest
        The extracted chunk manifest.

    Raises
    ------
    ValueError
        If data is a Dataset and variable is not specified when there are
        multiple data variables.
    TypeError
        If data is not a supported type or doesn't contain a ManifestArray.

    Examples
    --------
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import extract_manifest
    >>> vds = open_virtual_dataset("data.nc")
    >>> manifest = extract_manifest(vds, "temperature")
    """
    # Import here to avoid circular imports and allow lazy loading
    from virtualizarr.manifests import ChunkManifest, ManifestArray

    # Already a ChunkManifest
    if isinstance(data, ChunkManifest):
        return data

    # ManifestArray - extract manifest directly
    if isinstance(data, ManifestArray):
        return data.manifest

    # Try xarray types
    try:
        import xarray as xr

        if isinstance(data, xr.DataArray):
            arr = data.data
            if isinstance(arr, ManifestArray):
                return arr.manifest
            raise TypeError(
                f"DataArray does not contain a ManifestArray, got {type(arr).__name__}"
            )

        if isinstance(data, xr.Dataset):
            # Get list of data variables (exclude coordinates)
            data_vars = list(data.data_vars)

            if variable is None:
                if len(data_vars) == 1:
                    variable = data_vars[0]
                else:
                    raise ValueError(
                        f"Dataset has multiple data variables ({data_vars}). "
                        "Please specify the 'variable' parameter."
                    )

            if variable not in data_vars:
                raise ValueError(
                    f"Variable '{variable}' not found in dataset. "
                    f"Available variables: {data_vars}"
                )

            arr = data[variable].data
            if isinstance(arr, ManifestArray):
                return arr.manifest
            raise TypeError(
                f"Variable '{variable}' does not contain a ManifestArray, "
                f"got {type(arr).__name__}"
            )
    except ImportError:
        pass

    raise TypeError(
        f"Cannot extract manifest from {type(data).__name__}. "
        "Expected ChunkManifest, ManifestArray, xr.Dataset, or xr.DataArray."
    )


def manifest_to_dataframe(
    data: ManifestLike,
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Convert a ChunkManifest to a pandas DataFrame.

    Parameters
    ----------
    data : ManifestLike
        Input data containing manifest information.
    variable : str, optional
        Variable name if data is an xr.Dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - chunk_key: str (e.g., "0.1.2")
        - dim_0, dim_1, ...: int (dimension indices for each dimension)
        - path: str (full URI/path)
        - filename: str (basename extracted from path)
        - offset: int (byte offset in file)
        - length: int (byte length of chunk)
        - end_offset: int (offset + length)
        - file_id: int (categorical index for coloring)

    Examples
    --------
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr_viz import manifest_to_dataframe
    >>> vds = open_virtual_dataset("data.nc")
    >>> df = manifest_to_dataframe(vds, "temperature")
    >>> df.head()
    """
    manifest = extract_manifest(data, variable)

    # Get chunk data as dictionary
    chunk_dict = manifest.dict()

    if not chunk_dict:
        # Empty manifest
        return pd.DataFrame(
            columns=["chunk_key", "path", "filename", "offset", "length", "end_offset", "file_id"]
        )

    # Build records from chunk dictionary
    records = []
    for chunk_key, entry in chunk_dict.items():
        record = {
            "chunk_key": chunk_key,
            "path": entry["path"],
            "offset": entry["offset"],
            "length": entry["length"],
        }

        # Parse chunk key into dimension indices
        if chunk_key == "c":
            # Scalar array
            pass
        else:
            indices = [int(i) for i in chunk_key.split(".")]
            for dim_idx, idx in enumerate(indices):
                record[f"dim_{dim_idx}"] = idx

        records.append(record)

    df = pd.DataFrame(records)

    # Add derived columns
    df["end_offset"] = df["offset"] + df["length"]

    # Extract filename from path
    df["filename"] = df["path"].apply(_extract_filename)

    # Assign file_id based on unique paths
    unique_paths = df["path"].unique()
    path_to_id = {path: idx for idx, path in enumerate(unique_paths)}
    df["file_id"] = df["path"].map(path_to_id)

    # Reorder columns for readability
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    other_cols = ["chunk_key"] + dim_cols + ["path", "filename", "offset", "length", "end_offset", "file_id"]
    df = df[[c for c in other_cols if c in df.columns]]

    return df


def _extract_filename(path: str) -> str:
    """Extract filename from a path or URI."""
    if not path:
        return ""

    # Handle URIs (s3://, file://, http://, etc.)
    parsed = urlparse(path)
    if parsed.scheme:
        # It's a URI, extract the path portion
        path_part = parsed.path
    else:
        path_part = path

    # Get the basename
    parts = path_part.rstrip("/").rsplit("/", 1)
    return parts[-1] if parts else path


def get_manifest_info(
    data: ManifestLike,
    variable: str | None = None,
) -> dict:
    """
    Get basic information about a manifest.

    Parameters
    ----------
    data : ManifestLike
        Input data containing manifest information.
    variable : str, optional
        Variable name if data is an xr.Dataset.

    Returns
    -------
    dict
        Dictionary with manifest information including:
        - shape_chunk_grid: tuple of chunk grid dimensions
        - ndim: number of dimensions
        - total_chunks: total number of chunks
        - unique_files: number of unique files
    """
    manifest = extract_manifest(data, variable)

    df = manifest_to_dataframe(manifest)

    return {
        "shape_chunk_grid": manifest.shape_chunk_grid,
        "ndim": manifest.ndim_chunk_grid,
        "total_chunks": len(df),
        "unique_files": df["path"].nunique() if len(df) > 0 else 0,
    }
