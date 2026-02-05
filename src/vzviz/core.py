"""Core data extraction and transformation functions for ManifestStore."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pandas as pd

if TYPE_CHECKING:
    from virtualizarr.manifests import ManifestArray, ManifestGroup, ManifestStore


def manifest_to_dataframe(
    store: "ManifestStore",
    variable: str | None = None,
) -> pd.DataFrame:
    """
    Convert chunk manifests from a ManifestStore to a pandas DataFrame.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to extract chunk information from.
    variable : str, optional
        If provided, only extract chunks for this variable. The variable path
        can include group paths (e.g., "science/LSAR/data").

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - variable: str (full path to variable)
        - chunk_key: str (e.g., "0.1.2")
        - dim_0, dim_1, ...: int (dimension indices for each dimension)
        - path: str (full URI/path to file)
        - filename: str (basename extracted from path)
        - offset: int (byte offset in file)
        - length: int (byte length of chunk)
        - end_offset: int (offset + length)
        - file_id: int (categorical index for coloring)

    Examples
    --------
    >>> from virtualizarr.parsers import HDFParser
    >>> parser = HDFParser()
    >>> store = parser(url, registry)
    >>> df = manifest_to_dataframe(store)
    >>> df.head()
    """
    records = []

    def process_group(group: "ManifestGroup", group_path: str = "") -> None:
        """Recursively process all arrays in groups."""
        for array_name, array in group.arrays.items():
            var_path = f"{group_path}/{array_name}" if group_path else array_name

            # Filter by variable if specified
            if variable is not None and var_path != variable:
                continue

            manifest = array.manifest
            for chunk_key, entry in manifest.dict().items():
                record = {
                    "variable": var_path,
                    "chunk_key": chunk_key,
                    "path": entry["path"],
                    "offset": entry["offset"],
                    "length": entry["length"],
                }

                # Parse chunk key into dimension indices
                # Chunk keys can be:
                # - "c" for scalars
                # - "0" for 1D single chunk
                # - "0.1.2" for multi-dimensional
                # - "" (empty) for some edge cases
                if chunk_key and chunk_key != "c":
                    parts = chunk_key.split(".")
                    # Filter out empty strings from split result
                    indices = [int(i) for i in parts if i]
                    for dim_idx, idx in enumerate(indices):
                        record[f"dim_{dim_idx}"] = idx

                records.append(record)

        # Recursively process subgroups
        for group_name, subgroup in group.groups.items():
            sub_path = f"{group_path}/{group_name}" if group_path else group_name
            process_group(subgroup, group_path=sub_path)

    # Start from root group
    process_group(store._group)

    if not records:
        return pd.DataFrame(
            columns=[
                "variable",
                "chunk_key",
                "path",
                "filename",
                "offset",
                "length",
                "end_offset",
                "file_id",
            ]
        )

    df = pd.DataFrame(records)

    # Add derived columns
    df["end_offset"] = df["offset"] + df["length"]
    df["filename"] = df["path"].apply(_extract_filename)

    # Assign file_id based on unique paths
    unique_paths = df["path"].unique()
    path_to_id = {path: idx for idx, path in enumerate(unique_paths)}
    df["file_id"] = df["path"].map(path_to_id)

    # Reorder columns for readability
    dim_cols = sorted([c for c in df.columns if c.startswith("dim_")])
    base_cols = (
        ["variable", "chunk_key"]
        + dim_cols
        + ["path", "filename", "offset", "length", "end_offset", "file_id"]
    )
    df = df[[c for c in base_cols if c in df.columns]]

    return df


def get_array(store: "ManifestStore", variable: str) -> "ManifestArray":
    """
    Get a ManifestArray from the store by variable path.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to search.
    variable : str
        Variable path (e.g., "temperature" or "science/LSAR/data").

    Returns
    -------
    ManifestArray
        The array at the specified path.

    Raises
    ------
    KeyError
        If the variable is not found.
    """
    parts = variable.strip("/").split("/")
    current = store._group

    # Navigate to the array
    for i, part in enumerate(parts[:-1]):
        if part in current.groups:
            current = current.groups[part]
        else:
            raise KeyError(f"Group '{part}' not found in path '{variable}'")

    array_name = parts[-1]
    if array_name in current.arrays:
        return current.arrays[array_name]
    else:
        available = list(current.arrays.keys())
        raise KeyError(f"Array '{array_name}' not found. Available: {available}")


def list_variables(store: "ManifestStore") -> list[str]:
    """
    List all variable paths in a ManifestStore.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to list.

    Returns
    -------
    list[str]
        List of variable paths (e.g., ["temperature", "science/data"]).
    """
    variables = []

    def collect(group: "ManifestGroup", path: str = "") -> None:
        for name in group.arrays:
            var_path = f"{path}/{name}" if path else name
            variables.append(var_path)
        for name, subgroup in group.groups.items():
            sub_path = f"{path}/{name}" if path else name
            collect(subgroup, sub_path)

    collect(store._group)
    return variables


def get_store_info(store: "ManifestStore") -> dict:
    """
    Get basic information about a ManifestStore.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to analyze.

    Returns
    -------
    dict
        Dictionary with:
        - n_variables: int
        - n_groups: int
        - total_chunks: int
        - unique_files: int
        - variables: list[str]
    """
    variables = list_variables(store)
    df = manifest_to_dataframe(store)

    return {
        "n_variables": len(variables),
        "n_groups": _count_groups(store._group),
        "total_chunks": len(df),
        "unique_files": df["path"].nunique() if len(df) > 0 else 0,
        "variables": variables,
    }


def _count_groups(group: "ManifestGroup") -> int:
    """Count total number of groups recursively."""
    count = len(group.groups)
    for subgroup in group.groups.values():
        count += _count_groups(subgroup)
    return count


def get_dimension_names(store: "ManifestStore", variable: str) -> list[str] | None:
    """
    Get dimension names for a variable if available.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore containing the variable.
    variable : str
        Variable path.

    Returns
    -------
    list[str] or None
        List of dimension names, or None if not available.
    """
    try:
        array = get_array(store, variable)
        dim_names = array.metadata.dimension_names
        if dim_names is not None:
            return list(dim_names)
    except (KeyError, AttributeError):
        pass
    return None


def _extract_filename(path: str) -> str:
    """Extract filename from a path or URI."""
    if not path:
        return ""

    # Handle URIs (s3://, file://, http://, etc.)
    parsed = urlparse(path)
    if parsed.scheme:
        path_part = parsed.path
    else:
        path_part = path

    # Get the basename
    parts = path_part.rstrip("/").rsplit("/", 1)
    return parts[-1] if parts else path
