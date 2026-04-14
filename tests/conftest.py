"""Pytest configuration and fixtures."""

from __future__ import annotations

import numpy as np
import pytest


def _create_manifest_store(manifest_array, var_name="data"):
    """Helper to create a ManifestStore from a ManifestArray."""
    from virtualizarr.manifests import ManifestGroup, ManifestStore

    group = ManifestGroup(arrays={var_name: manifest_array}, groups={})
    return ManifestStore(group)


@pytest.fixture
def sample_manifest():
    """Create a sample ChunkManifest for testing."""
    from virtualizarr.manifests import ChunkManifest

    entries = {
        "0.0": {"path": "file:///data/file1.nc", "offset": 100, "length": 1000},
        "0.1": {"path": "file:///data/file1.nc", "offset": 1100, "length": 1000},
        "1.0": {"path": "file:///data/file2.nc", "offset": 200, "length": 1000},
        "1.1": {"path": "file:///data/file2.nc", "offset": 1200, "length": 1000},
    }
    return ChunkManifest(entries=entries, shape=(2, 2))


@pytest.fixture
def sample_manifest_array(sample_manifest):
    """Create a sample ManifestArray for testing."""
    from virtualizarr.manifests import ManifestArray
    from virtualizarr.manifests.utils import create_v3_array_metadata

    metadata = create_v3_array_metadata(
        shape=(20, 20),
        chunk_shape=(10, 10),
        data_type=np.dtype("float32"),
        codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
        fill_value=0.0,
    )

    return ManifestArray(metadata=metadata, chunkmanifest=sample_manifest)


@pytest.fixture
def sample_store(sample_manifest_array):
    """Create a sample ManifestStore for testing."""
    return _create_manifest_store(sample_manifest_array, "data")


@pytest.fixture
def multi_file_manifest():
    """Create a manifest spanning multiple files for testing."""
    from virtualizarr.manifests import ChunkManifest

    entries = {}
    for i in range(5):
        for j in range(4):
            file_idx = (i + j) % 3
            entries[f"{i}.{j}"] = {
                "path": f"s3://bucket/data/file{file_idx}.nc",
                "offset": i * 10000 + j * 1000,
                "length": 900 + (i * j) % 100,
            }

    return ChunkManifest(entries=entries, shape=(5, 4))


@pytest.fixture
def multi_file_store(multi_file_manifest):
    """Create a ManifestStore spanning multiple files for testing."""
    from virtualizarr.manifests import ManifestArray
    from virtualizarr.manifests.utils import create_v3_array_metadata

    metadata = create_v3_array_metadata(
        shape=(50, 40),
        chunk_shape=(10, 10),
        data_type=np.dtype("float32"),
        codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
        fill_value=0.0,
    )

    array = ManifestArray(metadata=metadata, chunkmanifest=multi_file_manifest)
    return _create_manifest_store(array, "data")


@pytest.fixture
def manifest_array_with_metadata():
    """Create a ManifestArray with dimension names and CF attributes."""
    from virtualizarr.manifests import ChunkManifest, ManifestArray
    from virtualizarr.manifests.utils import create_v3_array_metadata

    entries = {
        "0.0": {"path": "file:///data/temp.nc", "offset": 0, "length": 800},
        "0.1": {"path": "file:///data/temp.nc", "offset": 800, "length": 1200},
        "1.0": {"path": "file:///data/temp.nc", "offset": 2000, "length": 800},
        "1.1": {"path": "file:///data/temp.nc", "offset": 2800, "length": 1200},
    }
    manifest = ChunkManifest(entries=entries, shape=(2, 2))

    metadata = create_v3_array_metadata(
        shape=(20, 20),
        chunk_shape=(10, 10),
        data_type=np.dtype("float32"),
        codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
        fill_value=-9999.0,
        dimension_names=("y", "x"),
        attributes={
            "_FillValue": -9999.0,
            "scale_factor": 0.01,
            "add_offset": 273.15,
            "units": "K",
            "long_name": "Surface Temperature",
        },
    )

    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


@pytest.fixture
def store_with_metadata(manifest_array_with_metadata):
    """Create a ManifestStore wrapping a metadata-rich ManifestArray."""
    return _create_manifest_store(manifest_array_with_metadata, "temperature")


@pytest.fixture
def multi_variable_store():
    """Create a ManifestStore with two variables spanning multiple files."""
    from virtualizarr.manifests import (
        ChunkManifest,
        ManifestArray,
        ManifestGroup,
        ManifestStore,
    )
    from virtualizarr.manifests.utils import create_v3_array_metadata

    # temperature: 2 chunks in file1.nc
    temp_entries = {
        "0": {"path": "file:///data/file1.nc", "offset": 100, "length": 1000},
        "1": {"path": "file:///data/file1.nc", "offset": 1100, "length": 1000},
    }
    temp_manifest = ChunkManifest(entries=temp_entries, shape=(2,))
    temp_metadata = create_v3_array_metadata(
        shape=(200,),
        chunk_shape=(100,),
        data_type=np.dtype("float32"),
        codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
        fill_value=0.0,
        dimension_names=("time",),
    )
    temp_array = ManifestArray(metadata=temp_metadata, chunkmanifest=temp_manifest)

    # pressure: 2 chunks across file1.nc and file2.nc
    pressure_entries = {
        "0": {"path": "file:///data/file1.nc", "offset": 5000, "length": 1000},
        "1": {"path": "file:///data/file2.nc", "offset": 100, "length": 1000},
    }
    pressure_manifest = ChunkManifest(entries=pressure_entries, shape=(2,))
    pressure_metadata = create_v3_array_metadata(
        shape=(200,),
        chunk_shape=(100,),
        data_type=np.dtype("float64"),
        codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
        fill_value=0.0,
        dimension_names=("time",),
    )
    pressure_array = ManifestArray(
        metadata=pressure_metadata, chunkmanifest=pressure_manifest
    )

    group = ManifestGroup(
        arrays={"temperature": temp_array, "pressure": pressure_array}, groups={}
    )
    return ManifestStore(group)
