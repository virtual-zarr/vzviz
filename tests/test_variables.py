"""Tests for variables module."""

from vzviz.variables import variables_overview


class TestVariablesOverview:
    def test_dimension_names_present(self, store_with_metadata):
        df = variables_overview(store_with_metadata)
        assert "dimension_names" in df.columns
        row = df[df["variable"] == "temperature"].iloc[0]
        assert row["dimension_names"] == "y, x"

    def test_dimension_names_missing(self, sample_store):
        df = variables_overview(sample_store)
        assert "dimension_names" in df.columns

    def test_fill_value_present(self, store_with_metadata):
        df = variables_overview(store_with_metadata)
        assert "fill_value" in df.columns
        row = df[df["variable"] == "temperature"].iloc[0]
        assert row["fill_value"] == "-9999.0"

    def test_fill_value_attr(self, store_with_metadata):
        df = variables_overview(store_with_metadata)
        assert "fill_value_attr" in df.columns
        row = df[df["variable"] == "temperature"].iloc[0]
        assert row["fill_value_attr"] == "-9999.0"

    def test_cf_attrs_present(self, store_with_metadata):
        df = variables_overview(store_with_metadata)
        assert "cf_attrs" in df.columns
        row = df[df["variable"] == "temperature"].iloc[0]
        assert "scale_factor" in row["cf_attrs"]
        assert "add_offset" in row["cf_attrs"]
        assert "units" in row["cf_attrs"]

    def test_cf_attrs_empty(self, sample_store):
        df = variables_overview(sample_store)
        row = df.iloc[0]
        assert row["cf_attrs"] == ""

    def test_compression_ratio(self, store_with_metadata):
        df = variables_overview(store_with_metadata)
        assert "compression_ratio" in df.columns
        row = df[df["variable"] == "temperature"].iloc[0]
        # uncompressed = 20*20*4 = 1600 bytes
        # compressed total = 800+1200+800+1200 = 4000 bytes
        # ratio = 4000/1600 = 2.50x
        assert row["compression_ratio"] == "2.50x"
