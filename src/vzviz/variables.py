"""Variable overview and chunk analysis for ManifestStore."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vzviz.utils import format_bytes

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestArray, ManifestGroup, ManifestStore


def variables_overview(store: "ManifestStore") -> pd.DataFrame:
    """
    Generate an overview of all variables in a ManifestStore.

    Shows per-variable information similar to vischunk's interface:
    - Shape and chunk shape
    - Number of cells and chunks
    - Chunk size in bytes
    - Data type

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per variable, columns:
        - variable: str (full path to variable)
        - shape: tuple (array shape)
        - chunks: tuple (chunk shape)
        - dtype: str (data type)
        - ndim: int (number of dimensions)
        - total_cells: int (product of shape)
        - total_chunks: int (number of chunks in manifest)
        - cells_per_chunk: int (product of chunk shape)
        - chunk_bytes_median: int (median chunk size in bytes)
        - chunk_bytes_human: str (human-readable chunk size)
        - total_bytes: int (total data size from manifest)
        - total_bytes_human: str (human-readable total size)

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> overview = variables_overview(store)
    >>> print(overview)
    """
    records = []

    def process_group(group: "ManifestGroup", group_path: str = "") -> None:
        for array_name, array in group.arrays.items():
            var_path = f"{group_path}/{array_name}" if group_path else array_name
            record = _analyze_array(array, var_path)
            records.append(record)

        for group_name, subgroup in group.groups.items():
            sub_path = f"{group_path}/{group_name}" if group_path else group_name
            process_group(subgroup, sub_path)

    process_group(store._group)

    if not records:
        return pd.DataFrame(
            columns=[
                "variable",
                "shape",
                "chunks",
                "dtype",
                "codecs",
                "ndim",
                "total_cells",
                "total_chunks",
                "cells_per_chunk",
                "chunk_bytes_median",
                "chunk_bytes_human",
                "total_bytes",
                "total_bytes_human",
                "dimension_names",
                "fill_value",
                "fill_value_attr",
                "cf_attrs",
                "compression_ratio",
            ]
        )

    df = pd.DataFrame(records)
    df = df.sort_values("total_bytes", ascending=False).reset_index(drop=True)
    return df


def _extract_codec_names(array: "ManifestArray") -> str:
    """Extract a human-readable codec summary from a ManifestArray."""
    try:
        codecs = array.metadata.codecs
        names = [codec.to_dict()["name"] for codec in codecs]
        # Filter out the default "bytes" codec for cleaner display
        names = [n for n in names if n != "bytes"]
        return " | ".join(names) if names else "none"
    except Exception:
        return "unknown"


def _analyze_array(array: "ManifestArray", var_path: str) -> dict:
    """Analyze a single ManifestArray and return its statistics."""
    shape = array.shape
    chunks = array.chunks
    dtype = array.dtype
    manifest = array.manifest

    total_cells = int(np.prod(shape)) if shape else 1
    cells_per_chunk = int(np.prod(chunks)) if chunks else 1

    chunk_dict = manifest.dict()
    total_chunks = len(chunk_dict)

    if chunk_dict:
        chunk_lengths = [entry["length"] for entry in chunk_dict.values()]
        chunk_bytes_median = int(np.median(chunk_lengths))
        total_bytes = sum(chunk_lengths)
    else:
        itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 8
        chunk_bytes_median = cells_per_chunk * itemsize
        total_bytes = 0

    # Extract codec names from metadata
    codecs_str = _extract_codec_names(array)

    # Dimension names
    metadata = array.metadata
    dim_names = metadata.dimension_names
    dim_names_str = ", ".join(str(d) for d in dim_names) if dim_names else ""

    # Fill value from zarr metadata
    fill_value = metadata.fill_value
    fill_value_str = str(fill_value) if fill_value is not None else ""

    # _FillValue from attributes (CF convention)
    attrs = metadata.attributes or {}
    fill_value_attr = attrs.get("_FillValue")
    fill_value_attr_str = str(fill_value_attr) if fill_value_attr is not None else ""

    # CF attribute summary
    cf_attr_keys = ["scale_factor", "add_offset", "units", "long_name", "coordinates"]
    present_cf_attrs = [k for k in cf_attr_keys if k in attrs]
    cf_attrs_str = ", ".join(present_cf_attrs)

    # Compression ratio
    itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 8
    uncompressed_bytes = total_cells * itemsize
    if uncompressed_bytes > 0 and total_bytes > 0:
        ratio = total_bytes / uncompressed_bytes
        compression_ratio_str = f"{ratio:.2f}x"
    else:
        compression_ratio_str = ""

    return {
        "variable": var_path,
        "shape": shape,
        "chunks": chunks,
        "dtype": str(dtype),
        "codecs": codecs_str,
        "ndim": len(shape) if shape else 0,
        "total_cells": total_cells,
        "total_chunks": total_chunks,
        "cells_per_chunk": cells_per_chunk,
        "chunk_bytes_median": chunk_bytes_median,
        "chunk_bytes_human": format_bytes(chunk_bytes_median),
        "total_bytes": total_bytes,
        "total_bytes_human": format_bytes(total_bytes),
        "dimension_names": dim_names_str,
        "fill_value": fill_value_str,
        "fill_value_attr": fill_value_attr_str,
        "cf_attrs": cf_attrs_str,
        "compression_ratio": compression_ratio_str,
    }


@dataclass
class DimensionInfo:
    """Information about a single dimension in the chunk grid."""

    dim_index: int
    size: int
    chunk_size: int
    n_chunks: int
    last_chunk_size: int
    is_regular: bool


@dataclass
class ChunkGridInfo:
    """Information about the chunk grid for a variable."""

    variable: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    chunk_grid_shape: tuple[int, ...]
    dimensions: list[DimensionInfo]
    dtype: str
    total_chunks: int
    chunk_bytes: int
    total_bytes: int

    @property
    def chunk_bytes_human(self) -> str:
        """Human-readable chunk size."""
        return format_bytes(self.chunk_bytes)

    @property
    def total_bytes_human(self) -> str:
        """Human-readable total size."""
        return format_bytes(self.total_bytes)

    def __repr__(self) -> str:
        lines = [
            f"ChunkGridInfo(variable='{self.variable}')",
            f"  shape:            {self.shape}",
            f"  chunks:           {self.chunks}",
            f"  chunk_grid_shape: {self.chunk_grid_shape}",
            f"  total_chunks:     {self.total_chunks}",
            f"  dtype:            {self.dtype}",
            f"  chunk_bytes:      {self.chunk_bytes_human}",
            f"  total_bytes:      {self.total_bytes_human}",
        ]
        for dim in self.dimensions:
            lines.append(
                f"  dim {dim.dim_index}: size={dim.size}, "
                f"chunk_size={dim.chunk_size}, n_chunks={dim.n_chunks}"
            )
        return "\n".join(lines)


def chunk_grid_info(
    store: "ManifestStore",
    variable: str,
) -> ChunkGridInfo:
    """
    Get detailed chunk grid information for a variable.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore containing the variable.
    variable : str
        Variable path (e.g., "temperature" or "science/data").

    Returns
    -------
    ChunkGridInfo
        Dataclass with:
        - variable: str
        - shape: tuple (array shape)
        - chunks: tuple (chunk shape)
        - chunk_grid_shape: tuple (number of chunks along each dimension)
        - dimensions: list[DimensionInfo] (per-dimension details)
        - dtype: str (data type)
        - total_chunks: int (total number of chunks)
        - chunk_bytes: int (median chunk size in bytes)
        - total_bytes: int (total data size)

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> info = chunk_grid_info(store, "temperature")
    >>> print(info)
    """
    from vzviz.core import get_array

    array = get_array(store, variable)
    shape = array.shape
    chunks = array.chunks
    dtype = str(array.dtype)
    manifest = array.manifest

    chunk_grid_shape = tuple(math.ceil(s / c) for s, c in zip(shape, chunks))

    dimensions = []
    for i, (size, chunk_size) in enumerate(zip(shape, chunks)):
        n_chunks = math.ceil(size / chunk_size)
        last_chunk_size = size % chunk_size or chunk_size
        dimensions.append(
            DimensionInfo(
                dim_index=i,
                size=size,
                chunk_size=chunk_size,
                n_chunks=n_chunks,
                last_chunk_size=last_chunk_size,
                is_regular=last_chunk_size == chunk_size,
            )
        )

    # Get chunk statistics from manifest
    chunk_dict = manifest.dict()
    total_chunks = len(chunk_dict)

    if chunk_dict:
        chunk_lengths = [entry["length"] for entry in chunk_dict.values()]
        chunk_bytes = int(np.median(chunk_lengths))
        total_bytes = sum(chunk_lengths)
    else:
        # Estimate from dtype if no chunks
        itemsize = array.dtype.itemsize if hasattr(array.dtype, "itemsize") else 8
        cells_per_chunk = int(np.prod(chunks)) if chunks else 1
        chunk_bytes = cells_per_chunk * itemsize
        total_bytes = 0

    return ChunkGridInfo(
        variable=variable,
        shape=shape,
        chunks=chunks,
        chunk_grid_shape=chunk_grid_shape,
        dimensions=dimensions,
        dtype=dtype,
        total_chunks=total_chunks,
        chunk_bytes=chunk_bytes,
        total_bytes=total_bytes,
    )
