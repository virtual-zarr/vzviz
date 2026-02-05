"""Tests for selection module."""

from __future__ import annotations

import pandas as pd
import pytest

from vzviz.selection import (
    SelectionState,
    get_chunk_info,
)


class TestSelectionState:
    """Tests for SelectionState class."""

    def test_initial_state(self):
        state = SelectionState()
        assert state.bounds is None
        assert state.selected_variables == []
        assert not state.has_selection
        assert not state.has_variable_selection

    def test_set_bounds(self):
        state = SelectionState()
        state.set_bounds((0, 0, 5, 10))
        assert state.bounds == (0, 0, 5, 10)
        assert state.has_selection

    def test_set_bounds_normalizes(self):
        state = SelectionState()
        # Bounds with swapped min/max should be normalized
        state.set_bounds((10, 5, 2, 0))
        assert state.bounds == (2, 0, 10, 5)

    def test_set_bounds_none_clears(self):
        state = SelectionState()
        state.set_bounds((0, 0, 5, 10))
        state.set_bounds(None)
        assert state.bounds is None

    def test_set_selected_variables(self):
        state = SelectionState()
        state.set_selected_variables(["temp", "pressure"])
        assert state.selected_variables == ["temp", "pressure"]
        assert state.has_selection
        assert state.has_variable_selection

    def test_set_selected_variables_empty_clears(self):
        state = SelectionState()
        state.set_selected_variables(["temp"])
        state.set_selected_variables([])
        assert state.selected_variables == []
        assert not state.has_variable_selection

    def test_clear_selection(self):
        state = SelectionState()
        state.set_bounds((0, 0, 5, 10))
        state.set_selected_variables(["temp"])
        state.clear_selection()
        assert state.bounds is None
        assert state.selected_variables == []
        assert not state.has_selection

    @pytest.fixture
    def chunk_df(self):
        """Create a DataFrame with chunks at various dimension positions."""
        return pd.DataFrame(
            {
                "chunk_key": ["var1_0.0", "var1_0.1", "var1_1.0", "var2_0.0"],
                "path": ["/data/file1.nc"] * 4,
                "filename": ["file1.nc"] * 4,
                "offset": [100, 200, 300, 400],
                "length": [1000, 1000, 1000, 1000],
                "variable": ["var1", "var1", "var1", "var2"],
                "dim_0": [0, 0, 1, 0],
                "dim_1": [0, 1, 0, 0],
            }
        )

    def test_get_selected_chunk_keys_by_variables(self, chunk_df):
        state = SelectionState()
        state.set_selected_variables(["var1"])
        selected = state.get_selected_chunk_keys(chunk_df)
        assert selected == {"var1_0.0", "var1_0.1", "var1_1.0"}

    def test_get_selected_chunk_keys_by_bounds(self, chunk_df):
        state = SelectionState()
        state.dim_x = 0
        state.dim_y = 1
        state.set_bounds((0, 0, 0, 1))  # dim_0=0, dim_1=0-1
        state.bounds_in_array_space = False  # Bounds are in chunk indices
        selected = state.get_selected_chunk_keys(chunk_df)
        # Should match chunks where dim_0=0 AND dim_1 in [0,1]
        assert "var1_0.0" in selected
        assert "var1_0.1" in selected
        assert "var2_0.0" in selected
        assert "var1_1.0" not in selected

    def test_get_selected_chunk_keys_bounds_in_array_space(self, chunk_df):
        state = SelectionState()
        state.dim_x = 0
        state.dim_y = 1
        # Assume each chunk covers 10 array indices
        state.chunk_shape = (10, 10)
        state.bounds_in_array_space = True
        # Select array indices 0-5, 0-15 -> chunk indices 0, 0-1
        state.set_bounds((0, 0, 5, 15))
        selected = state.get_selected_chunk_keys(chunk_df)
        # Should convert to dim_0=0, dim_1=0-1
        assert "var1_0.0" in selected
        assert "var1_0.1" in selected
        assert "var2_0.0" in selected
        assert "var1_1.0" not in selected

    def test_get_selected_chunk_keys_for_highlighting(self, chunk_df):
        """for_highlighting=True only returns bounds-selected chunks."""
        state = SelectionState()
        state.dim_x = 0
        state.dim_y = 1
        # Select by variable and by bounds
        state.set_selected_variables(["var2"])
        state.set_bounds((1, 0, 1, 0))  # dim_0=1, dim_1=0
        state.bounds_in_array_space = False
        state.bounds_variable = "var1"  # Bounds are from var1's ChunkMap

        # for_highlighting=True: only bounds selection (for yellow highlighting)
        selected_highlight = state.get_selected_chunk_keys(
            chunk_df, for_highlighting=True
        )
        assert "var1_1.0" in selected_highlight  # From bounds selection
        assert "var2_0.0" not in selected_highlight  # Variable selection excluded

        # for_highlighting=False (default): both bounds and variable selection
        selected_stats = state.get_selected_chunk_keys(chunk_df, for_highlighting=False)
        assert "var1_1.0" in selected_stats  # From bounds selection
        assert "var2_0.0" in selected_stats  # From variable selection

    def test_get_selected_chunk_keys_no_selection(self, chunk_df):
        state = SelectionState()
        selected = state.get_selected_chunk_keys(chunk_df)
        assert selected == set()

    def test_get_selected_chunk_keys_bounds_variable_filter(self, chunk_df):
        """Bounds from ChunkMap should only match the displayed variable."""
        state = SelectionState()
        state.dim_x = 0
        state.dim_y = 1
        # Set bounds that would match both var1 and var2 chunks
        state.set_bounds((0, 0, 0, 0))  # dim_0=0, dim_1=0
        state.bounds_in_array_space = False
        # But specify this came from var1's ChunkMap
        state.bounds_variable = "var1"
        selected = state.get_selected_chunk_keys(chunk_df)
        # Should only match var1's chunk, not var2's
        assert "var1_0.0" in selected
        assert "var2_0.0" not in selected


class TestGetChunkInfo:
    """Tests for get_chunk_info function."""

    @pytest.fixture
    def chunk_df(self):
        return pd.DataFrame(
            {
                "chunk_key": ["0.0", "0.1"],
                "path": ["/data/file1.nc", "/data/file2.nc"],
                "filename": ["file1.nc", "file2.nc"],
                "offset": [100, 200],
                "length": [1000, 2000],
                "variable": ["temp", "temp"],
            }
        )

    def test_get_chunk_info_found(self, chunk_df):
        info = get_chunk_info("0.0", chunk_df)
        assert info is not None
        assert info["chunk_key"] == "0.0"
        assert info["filename"] == "file1.nc"
        assert info["offset"] == 100
        assert info["length"] == 1000

    def test_get_chunk_info_not_found(self, chunk_df):
        info = get_chunk_info("9.9", chunk_df)
        assert info is None
