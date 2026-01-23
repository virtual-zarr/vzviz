"""Tests for core module."""

from __future__ import annotations

import pytest

from virtualizarr_viz.core import (
    _extract_filename,
    extract_manifest,
    manifest_to_dataframe,
)


class TestExtractFilename:
    """Tests for _extract_filename function."""

    def test_simple_path(self):
        assert _extract_filename("/path/to/file.nc") == "file.nc"

    def test_uri_file(self):
        assert _extract_filename("file:///data/file.nc") == "file.nc"

    def test_uri_s3(self):
        assert _extract_filename("s3://bucket/path/to/file.nc") == "file.nc"

    def test_uri_http(self):
        assert _extract_filename("https://example.com/data/file.nc") == "file.nc"

    def test_empty_string(self):
        assert _extract_filename("") == ""

    def test_trailing_slash(self):
        assert _extract_filename("/path/to/dir/") == "dir"


class TestExtractManifest:
    """Tests for extract_manifest function."""

    def test_from_chunk_manifest(self, sample_manifest):
        result = extract_manifest(sample_manifest)
        assert result is sample_manifest

    def test_from_manifest_array(self, sample_manifest_array):
        result = extract_manifest(sample_manifest_array)
        assert result is sample_manifest_array.manifest

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="Cannot extract manifest"):
            extract_manifest("invalid")


class TestManifestToDataframe:
    """Tests for manifest_to_dataframe function."""

    def test_basic_conversion(self, sample_manifest):
        df = manifest_to_dataframe(sample_manifest)

        assert len(df) == 4
        assert "chunk_key" in df.columns
        assert "path" in df.columns
        assert "offset" in df.columns
        assert "length" in df.columns
        assert "end_offset" in df.columns
        assert "filename" in df.columns
        assert "file_id" in df.columns

    def test_dimension_columns(self, sample_manifest):
        df = manifest_to_dataframe(sample_manifest)

        assert "dim_0" in df.columns
        assert "dim_1" in df.columns

    def test_end_offset_calculation(self, sample_manifest):
        df = manifest_to_dataframe(sample_manifest)

        for _, row in df.iterrows():
            assert row["end_offset"] == row["offset"] + row["length"]

    def test_file_id_assignment(self, sample_manifest):
        df = manifest_to_dataframe(sample_manifest)

        # Should have 2 unique file IDs (file1.nc and file2.nc)
        assert df["file_id"].nunique() == 2

    def test_multi_file_manifest(self, multi_file_manifest):
        df = manifest_to_dataframe(multi_file_manifest)

        assert len(df) == 20  # 5 * 4 chunks
        assert df["path"].nunique() == 3  # 3 different files
