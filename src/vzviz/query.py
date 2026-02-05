"""Query simulation with vischunk-like performance metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vzviz.utils import format_bytes

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestStore
    import pandas as pd


@dataclass
class QueryMetrics:
    """
    Performance metrics for a simulated query, inspired by vischunk.

    These metrics help understand the efficiency implications of chunking
    for a given data access pattern.
    """

    # Query definition
    variable: str
    query: dict[int, slice]  # {dim_index: slice}
    shape: tuple[int, ...]
    chunks: tuple[int, ...]

    # Cell counts
    requested_cells: int
    cells_read: int
    total_cells: int

    # Chunk counts
    chunks_touched: int
    total_chunks: int

    # Byte counts
    bytes_requested: int  # Estimated uncompressed bytes for requested cells
    bytes_read: int  # Actual bytes that would be read (from manifest)

    # Range read info
    range_reads: int  # Number of separate byte range requests
    ranges: list[tuple[int, int]] = field(default_factory=list)  # (start, end) tuples

    @property
    def read_amplification(self) -> float:
        """Ratio of cells read to cells requested. Lower is better. 1.0 is optimal."""
        if self.requested_cells == 0:
            return float("inf")
        return self.cells_read / self.requested_cells

    @property
    def read_efficiency(self) -> float:
        """Percentage of read data that is actually needed. Higher is better."""
        if self.cells_read == 0:
            return 0.0
        return (self.requested_cells / self.cells_read) * 100

    @property
    def coalescing_factor(self) -> float:
        """
        How much read coalescing improves I/O vs worst case.

        Computed as chunks_touched / range_reads. Higher means more chunks
        are combined into fewer reads due to contiguous storage.

        Note: This assumes a sharded format where chunks can be coalesced.
        For unsharded Zarr (one file per chunk), this would always be 1.0.
        """
        if self.range_reads == 0:
            return 0.0
        return self.chunks_touched / self.range_reads

    @property
    def chunk_coverage(self) -> float:
        """Percentage of chunks that are touched by this query."""
        if self.total_chunks == 0:
            return 0.0
        return (self.chunks_touched / self.total_chunks) * 100

    def __repr__(self) -> str:
        lines = [
            f"QueryMetrics(variable='{self.variable}')",
            f"  Query: {self._format_query()}",
            f"  Shape: {self.shape}, Chunks: {self.chunks}",
            "",
            "  Cell Metrics:",
            f"    Requested:        {self.requested_cells:,}",
            f"    Read:             {self.cells_read:,}",
            f"    Read Amplification: {self.read_amplification:.2f}x",
            f"    Read Efficiency:  {self.read_efficiency:.1f}%",
            "",
            "  Chunk Metrics:",
            f"    Chunks Touched:   {self.chunks_touched:,} / {self.total_chunks:,} ({self.chunk_coverage:.1f}%)",
            f"    Range Reads:      {self.range_reads:,}",
            f"    Coalescing Factor: {self.coalescing_factor:.2f}x",
            "",
            "  Byte Metrics:",
            f"    Bytes Read:       {format_bytes(self.bytes_read)}",
        ]
        return "\n".join(lines)

    def _format_query(self) -> str:
        parts = []
        for dim, slc in sorted(self.query.items()):
            if slc.stop is not None:
                parts.append(f"dim{dim}=[{slc.start}:{slc.stop}]")
            else:
                parts.append(f"dim{dim}=[{slc.start}:]")
        return ", ".join(parts) if parts else "full array"


def simulate_query(
    store: "ManifestStore",
    variable: str,
    query: dict[int, slice] | None = None,
) -> QueryMetrics:
    """
    Simulate a query and compute vischunk-like performance metrics.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore containing the variable.
    variable : str
        Variable path (e.g., "temperature" or "science/LSAR/data").
    query : dict[int, slice], optional
        Query specification as {dimension_index: slice}.
        E.g., {0: slice(0, 10), 2: slice(100, 200)} queries the first
        10 elements of dimension 0 and elements 100-199 of dimension 2.
        If None, simulates reading the entire array.

    Returns
    -------
    QueryMetrics
        Dataclass containing all computed metrics.

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)

    # Query first 100 time steps, all lat/lon
    >>> metrics = simulate_query(store, "temperature", {0: slice(0, 100)})
    >>> print(metrics)

    # Full array read
    >>> metrics = simulate_query(store, "temperature")
    >>> print(f"Read amplification: {metrics.read_amplification:.2f}x")
    """
    from vzviz.core import get_array

    array = get_array(store, variable)
    shape = array.shape
    chunks = array.chunks
    manifest = array.manifest

    # Default query: entire array
    if query is None:
        query = {i: slice(0, s) for i, s in enumerate(shape)}

    # Normalize query - fill in missing dimensions
    full_query = {}
    for i, s in enumerate(shape):
        if i in query:
            slc = query[i]
            start = slc.start or 0
            stop = slc.stop if slc.stop is not None else s
            full_query[i] = slice(start, min(stop, s))
        else:
            full_query[i] = slice(0, s)

    # Calculate requested cells
    requested_cells = 1
    for i, s in enumerate(shape):
        slc = full_query[i]
        requested_cells *= slc.stop - slc.start

    total_cells = int(np.prod(shape)) if shape else 1

    # Determine which chunks are touched
    chunk_grid = tuple(math.ceil(s / c) for s, c in zip(shape, chunks))
    total_chunks = int(np.prod(chunk_grid)) if chunk_grid else 1

    touched_chunks = _get_touched_chunks(full_query, shape, chunks)
    chunks_touched = len(touched_chunks)

    # Calculate cells read (all cells in touched chunks)
    cells_read = 0
    for chunk_idx in touched_chunks:
        chunk_cells = 1
        for i, (ci, cs, s) in enumerate(zip(chunk_idx, chunks, shape)):
            # Handle edge chunks that may be smaller
            chunk_start = ci * cs
            chunk_end = min(chunk_start + cs, s)
            chunk_cells *= chunk_end - chunk_start
        cells_read += chunk_cells

    # Get actual byte ranges from manifest
    chunk_dict = manifest.dict()
    touched_entries = []

    for chunk_idx in touched_chunks:
        chunk_key = ".".join(str(i) for i in chunk_idx)
        if chunk_key in chunk_dict:
            entry = chunk_dict[chunk_key]
            touched_entries.append(
                {
                    "offset": entry["offset"],
                    "length": entry["length"],
                    "end": entry["offset"] + entry["length"],
                }
            )

    # Calculate bytes read
    bytes_read = sum(e["length"] for e in touched_entries)

    # Estimate bytes requested (uncompressed)
    dtype = array.dtype
    itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 8
    bytes_requested = requested_cells * itemsize

    # Calculate coalesced byte ranges
    ranges = _coalesce_ranges(touched_entries)
    range_reads = len(ranges)

    return QueryMetrics(
        variable=variable,
        query=full_query,
        shape=shape,
        chunks=chunks,
        requested_cells=requested_cells,
        cells_read=cells_read,
        total_cells=total_cells,
        chunks_touched=chunks_touched,
        total_chunks=total_chunks,
        bytes_requested=bytes_requested,
        bytes_read=bytes_read,
        range_reads=range_reads,
        ranges=ranges,
    )


def _get_touched_chunks(
    query: dict[int, slice],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """
    Determine which chunks are touched by a query.

    Returns list of chunk indices (tuples).
    """
    # Calculate chunk ranges for each dimension
    chunk_ranges = []
    for i, (s, c) in enumerate(zip(shape, chunks)):
        slc = query[i]
        start_chunk = slc.start // c
        # End chunk is inclusive of the last element
        end_chunk = (slc.stop - 1) // c if slc.stop > slc.start else start_chunk
        n_chunks = math.ceil(s / c)
        end_chunk = min(end_chunk, n_chunks - 1)
        chunk_ranges.append(range(start_chunk, end_chunk + 1))

    # Generate all chunk index combinations
    touched: list[tuple[int, ...]] = []
    _enumerate_chunks(chunk_ranges, 0, [], touched)
    return touched


def _enumerate_chunks(
    ranges: list[range],
    dim: int,
    current: list[int],
    result: list[tuple[int, ...]],
) -> None:
    """Recursively enumerate all chunk combinations."""
    if dim == len(ranges):
        result.append(tuple(current))
        return

    for idx in ranges[dim]:
        current.append(idx)
        _enumerate_chunks(ranges, dim + 1, current, result)
        current.pop()


def _coalesce_ranges(entries: list[dict]) -> list[tuple[int, int]]:
    """
    Coalesce adjacent byte ranges into larger contiguous ranges.

    Parameters
    ----------
    entries : list[dict]
        List of {"offset": int, "length": int, "end": int} dicts.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) tuples for coalesced ranges.
    """
    if not entries:
        return []

    # Sort by offset
    sorted_entries = sorted(entries, key=lambda e: e["offset"])

    ranges = []
    current_start = sorted_entries[0]["offset"]
    current_end = sorted_entries[0]["end"]

    for entry in sorted_entries[1:]:
        if entry["offset"] <= current_end:
            # Overlapping or adjacent - extend current range
            current_end = max(current_end, entry["end"])
        else:
            # Gap - save current range and start new one
            ranges.append((current_start, current_end))
            current_start = entry["offset"]
            current_end = entry["end"]

    # Don't forget the last range
    ranges.append((current_start, current_end))

    return ranges


def compare_queries(
    store: "ManifestStore",
    variable: str,
    queries: list[dict[int, slice]],
    names: list[str] | None = None,
) -> "pd.DataFrame":
    """
    Compare multiple queries and their metrics side by side.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore containing the variable.
    variable : str
        Variable path.
    queries : list[dict[int, slice]]
        List of query specifications.
    names : list[str], optional
        Names for each query (for display).

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per query.

    Examples
    --------
    >>> queries = [
    ...     {0: slice(0, 10)},    # First 10 time steps
    ...     {0: slice(0, 100)},   # First 100 time steps
    ...     {1: slice(0, 50), 2: slice(0, 50)},  # 50x50 spatial subset
    ... ]
    >>> comparison = compare_queries(store, "temperature", queries,
    ...                              names=["small_time", "large_time", "spatial"])
    >>> print(comparison)
    """
    import pandas as pd

    if names is None:
        names = [f"query_{i}" for i in range(len(queries))]

    records = []
    for name, query in zip(names, queries):
        metrics = simulate_query(store, variable, query)
        records.append(
            {
                "name": name,
                "requested_cells": metrics.requested_cells,
                "cells_read": metrics.cells_read,
                "read_amplification": round(metrics.read_amplification, 2),
                "read_efficiency_pct": round(metrics.read_efficiency, 1),
                "chunks_touched": metrics.chunks_touched,
                "range_reads": metrics.range_reads,
                "coalescing_factor": round(metrics.coalescing_factor, 2),
                "bytes_read": metrics.bytes_read,
                "bytes_read_human": format_bytes(metrics.bytes_read),
            }
        )

    return pd.DataFrame(records)
