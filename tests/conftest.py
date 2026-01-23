"""Pytest configuration and fixtures."""

from __future__ import annotations

import numpy as np
import pytest


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
    from zarr.core.metadata import ArrayV3Metadata
    from zarr.core.chunk_grids import RegularChunkGrid
    from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
    from zarr.abc.codec import Codec
    from zarr.codecs import BytesCodec

    metadata = ArrayV3Metadata(
        shape=(20, 20),
        data_type=np.dtype("float32"),
        chunk_grid=RegularChunkGrid(chunk_shape=(10, 10)),
        chunk_key_encoding=DefaultChunkKeyEncoding(separator="."),
        fill_value=0.0,
        codecs=[BytesCodec()],
    )

    return ManifestArray(metadata=metadata, chunkmanifest=sample_manifest)


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
