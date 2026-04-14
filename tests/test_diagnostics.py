"""Tests for diagnostics module."""

import numpy as np

from vzviz.diagnostics import chunk_size_distribution, gap_analysis


class TestChunkSizeDistribution:
    def test_returns_dataframe(self, sample_store):
        df = chunk_size_distribution(sample_store)
        assert "variable" in df.columns
        assert "length" in df.columns

    def test_single_variable(self, sample_store):
        df = chunk_size_distribution(sample_store, variable="data")
        assert len(df) == 4  # 4 chunks
        assert (df["variable"] == "data").all()

    def test_all_variables(self, multi_variable_store):
        df = chunk_size_distribution(multi_variable_store)
        assert set(df["variable"]) == {"temperature", "pressure"}

    def test_includes_uncompressed_size(self, sample_store):
        df = chunk_size_distribution(sample_store)
        assert "uncompressed_length" in df.columns
        # Each chunk is 10*10 * 4 bytes (float32) = 400
        assert (df["uncompressed_length"] == 400).all()


class TestGapAnalysis:
    def test_contiguous_chunks_no_gaps(self):
        from virtualizarr.manifests import (
            ChunkManifest,
            ManifestArray,
            ManifestGroup,
            ManifestStore,
        )
        from virtualizarr.manifests.utils import create_v3_array_metadata

        entries = {
            "0": {"path": "file:///data/f.nc", "offset": 100, "length": 500},
            "1": {"path": "file:///data/f.nc", "offset": 600, "length": 500},
        }
        manifest = ChunkManifest(entries=entries, shape=(2,))
        metadata = create_v3_array_metadata(
            shape=(20,),
            chunk_shape=(10,),
            data_type=np.dtype("float32"),
            codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
            fill_value=0.0,
        )
        array = ManifestArray(metadata=metadata, chunkmanifest=manifest)
        store = ManifestStore(ManifestGroup(arrays={"data": array}, groups={}))

        df = gap_analysis(store)
        assert len(df) == 1
        assert df.iloc[0]["gap_bytes"] == 0
        assert df.iloc[0]["n_gaps"] == 0

    def test_gaps_detected(self):
        from virtualizarr.manifests import (
            ChunkManifest,
            ManifestArray,
            ManifestGroup,
            ManifestStore,
        )
        from virtualizarr.manifests.utils import create_v3_array_metadata

        entries = {
            "0": {"path": "file:///data/f.nc", "offset": 100, "length": 500},
            "1": {"path": "file:///data/f.nc", "offset": 800, "length": 500},
        }
        manifest = ChunkManifest(entries=entries, shape=(2,))
        metadata = create_v3_array_metadata(
            shape=(20,),
            chunk_shape=(10,),
            data_type=np.dtype("float32"),
            codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
            fill_value=0.0,
        )
        array = ManifestArray(metadata=metadata, chunkmanifest=manifest)
        store = ManifestStore(ManifestGroup(arrays={"data": array}, groups={}))

        df = gap_analysis(store)
        assert df.iloc[0]["gap_bytes"] == 200
        assert df.iloc[0]["n_gaps"] == 1

    def test_gap_analysis_per_variable(self, multi_variable_store):
        df = gap_analysis(multi_variable_store, per_variable=True)
        assert "variable" in df.columns
