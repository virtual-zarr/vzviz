"""Kerchunk JSON serialization for ManifestStore.

Provides round-trip conversion between ManifestStore and Kerchunk JSON files,
bypassing xarray to avoid dimension conflict errors with complex HDF5 files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import ujson

from virtualizarr.codecs import zarr_codec_config_to_v3
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import ChunkEntry
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.utils import convert_v3_to_v2_metadata
from virtualizarr.writers.kerchunk import to_kerchunk_json

if TYPE_CHECKING:
    from zarr.core.metadata.v3 import ArrayV3Metadata

from obspec_utils.registry import ObjectStoreRegistry


def load_manifest_from_json(json_path: str | Path) -> ManifestStore:
    """
    Load a ManifestStore from a Kerchunk JSON file.

    Handles flattened hierarchies where variable names contain slashes
    (e.g., 'science/LSAR/data') and rebuilds the nested group structure.

    Parameters
    ----------
    json_path : str or Path
        Path to the Kerchunk JSON file.

    Returns
    -------
    ManifestStore
        The reconstructed ManifestStore.
    """
    with open(json_path) as f:
        kerchunk = ujson.load(f)

    refs = kerchunk["refs"]

    # Find all variable names (paths ending in /.zarray)
    var_names = _find_var_names(refs)

    # Build ManifestArrays for each variable
    arrays = {}
    for var_name in var_names:
        zarray_str = refs.get(f"{var_name}/.zarray")
        zattrs_str = refs.get(f"{var_name}/.zattrs", "{}")

        if not zarray_str:
            continue

        zattrs = ujson.loads(zattrs_str) if isinstance(zattrs_str, str) else zattrs_str
        metadata = _parse_zarray(zarray_str, zattrs)

        # Collect chunk entries
        chunk_entries = {}
        prefix = f"{var_name}/"
        for key, value in refs.items():
            if key.startswith(prefix) and not key.endswith(
                (".zarray", ".zattrs", ".zgroup")
            ):
                chunk_key = key[len(prefix) :]
                if not chunk_key or chunk_key.startswith("."):
                    continue
                if isinstance(value, list) and len(value) == 3:
                    path, offset, length = value
                    chunk_entries[chunk_key] = ChunkEntry.with_validation(
                        path=path, offset=offset, length=length
                    )

        if chunk_entries:
            manifest = ChunkManifest(entries=chunk_entries)
            arrays[var_name] = ManifestArray(metadata=metadata, chunkmanifest=manifest)

    # Get root attributes
    root_attrs_str = refs.get(".zattrs", "{}")
    root_attrs = (
        ujson.loads(root_attrs_str)
        if isinstance(root_attrs_str, str)
        else root_attrs_str
    )

    group = _build_nested_group(arrays, root_attrs)
    return ManifestStore(group=group, registry=ObjectStoreRegistry())


def save_manifest_to_json(
    store: ManifestStore,
    json_path: str | Path,
    metadata: dict | None = None,
) -> None:
    """
    Save a ManifestStore to a Kerchunk JSON file.

    Converts the ManifestStore directly to Kerchunk refs format,
    bypassing xarray entirely.

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to serialize.
    json_path : str or Path
        Path to write the Kerchunk JSON file.
    metadata : dict, optional
        If provided, written as a JSON sidecar file alongside the manifest
        (e.g., source URL, query parameters used to find the data).
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    kerchunk_refs = manifeststore_to_kerchunk_refs(store)

    with open(json_path, "w") as f:
        ujson.dump(kerchunk_refs, f)

    if metadata is not None:
        metadata_path = json_path.parent / "manifest_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def manifeststore_to_kerchunk_refs(store: ManifestStore) -> dict:
    """
    Convert a ManifestStore to Kerchunk refs format.

    The hierarchy is flattened so all arrays appear at the root level
    with their full paths as names (e.g., "science/LSAR/data").

    Parameters
    ----------
    store : ManifestStore
        The ManifestStore to convert.

    Returns
    -------
    dict
        Kerchunk-compatible refs dict with "version" and "refs" keys.
    """
    refs: dict[str, str | list] = {}

    refs[".zgroup"] = '{"zarr_format":2}'
    root_attrs = store._group.metadata.attributes or {}
    refs[".zattrs"] = ujson.dumps(root_attrs)

    def collect_arrays(group, path_prefix=""):
        arrays = []
        for array_name, array in group.arrays.items():
            full_path = f"{path_prefix}/{array_name}" if path_prefix else array_name
            arrays.append((full_path, array))
        for group_name, subgroup in group.groups.items():
            sub_path = f"{path_prefix}/{group_name}" if path_prefix else group_name
            arrays.extend(collect_arrays(subgroup, sub_path))
        return arrays

    for array_path, array in collect_arrays(store._group):
        v2_metadata = convert_v3_to_v2_metadata(array.metadata)
        refs[f"{array_path}/.zarray"] = to_kerchunk_json(v2_metadata)

        array_attrs = {}
        if array.metadata.dimension_names:
            array_attrs["_ARRAY_DIMENSIONS"] = list(array.metadata.dimension_names)
        refs[f"{array_path}/.zattrs"] = ujson.dumps(array_attrs)

        for chunk_key, entry in array.manifest.dict().items():
            path = entry["path"]
            if path.startswith("file://"):
                path = path[7:]
            if chunk_key == "":
                chunk_key = "c"
            refs[f"{array_path}/{chunk_key}"] = [
                path,
                entry["offset"],
                entry["length"],
            ]

    return {"version": 1, "refs": refs}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_var_names(refs: dict) -> list[str]:
    """Find variable names in refs by looking for keys ending in /.zarray."""
    return [key[:-8] for key in refs if key.endswith("/.zarray")]


def _parse_zarray(zarray_str: str, zattrs: dict) -> ArrayV3Metadata:
    """Parse a .zarray JSON string into ArrayV3Metadata."""
    zarray = ujson.loads(zarray_str) if isinstance(zarray_str, str) else zarray_str

    dtype = np.dtype(zarray["dtype"])
    fill_value = zarray.get("fill_value")

    if np.issubdtype(dtype, np.floating) and (
        fill_value is None or fill_value == "NaN" or fill_value == "nan"
    ):
        fill_value = np.nan
    elif np.issubdtype(dtype, np.complexfloating):
        if isinstance(fill_value, list) and len(fill_value) == 2:
            fill_value = complex(fill_value[0], fill_value[1])
        elif fill_value is None or fill_value == "NaN" or fill_value == "nan":
            fill_value = complex(np.nan, np.nan)

    filters = zarray.get("filters", []) or []
    compressor = zarray.get("compressor")

    codec_configs = [*filters, *(compressor if compressor is not None else [])]
    numcodec_configs = [
        zarr_codec_config_to_v3(config) for config in codec_configs if config
    ]

    dimension_names = zattrs.get("_ARRAY_DIMENSIONS")

    return create_v3_array_metadata(
        chunk_shape=tuple(zarray["chunks"]),
        data_type=dtype,
        codecs=numcodec_configs,
        fill_value=fill_value,
        shape=tuple(zarray["shape"]),
        dimension_names=dimension_names,
        attributes={k: v for k, v in zattrs.items() if k != "_ARRAY_DIMENSIONS"},
    )


def _build_nested_group(arrays_dict: dict, root_attrs: dict) -> ManifestGroup:
    """Build a nested ManifestGroup hierarchy from a flat dict of arrays."""
    groups_tree: dict[str, dict[str, ManifestArray]] = {}

    for full_path, array in arrays_dict.items():
        parts = full_path.split("/")
        if len(parts) == 1:
            group_path = ""
            array_name = parts[0]
        else:
            group_path = "/".join(parts[:-1])
            array_name = parts[-1]

        if group_path not in groups_tree:
            groups_tree[group_path] = {}
        groups_tree[group_path][array_name] = array

    def build_group(path: str) -> ManifestGroup:
        arrays_at_level = groups_tree.get(path, {})

        child_groups: dict[str, ManifestGroup] = {}
        prefix = f"{path}/" if path else ""
        for group_path in groups_tree:
            if group_path == path:
                continue
            if path == "":
                child_name = group_path.split("/")[0]
            else:
                if not group_path.startswith(prefix):
                    continue
                remainder = group_path[len(prefix) :]
                child_name = remainder.split("/")[0]

            if child_name and child_name not in child_groups:
                child_path = f"{prefix}{child_name}" if prefix else child_name
                child_groups[child_name] = build_group(child_path)

        attrs = root_attrs if path == "" else {}
        return ManifestGroup(
            arrays=arrays_at_level, groups=child_groups, attributes=attrs
        )

    return build_group("")
