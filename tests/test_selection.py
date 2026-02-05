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
        assert state.hovered_chunk_key is None
        assert state.selected_chunk_key is None
        assert state.selection_locked is False
        assert state.effective_chunk_key is None

    def test_set_hover(self):
        state = SelectionState()
        state.set_hover("0.1.2")
        assert state.hovered_chunk_key == "0.1.2"
        assert state.effective_chunk_key == "0.1.2"

    def test_toggle_selection_locks(self):
        state = SelectionState()
        state.toggle_selection("0.1.2")
        assert state.selection_locked is True
        assert state.selected_chunk_key == "0.1.2"
        assert state.effective_chunk_key == "0.1.2"

    def test_toggle_selection_unlocks_same_chunk(self):
        state = SelectionState()
        state.toggle_selection("0.1.2")
        state.toggle_selection("0.1.2")
        assert state.selection_locked is False
        assert state.selected_chunk_key is None

    def test_toggle_selection_switches_chunk(self):
        state = SelectionState()
        state.toggle_selection("0.1.2")
        state.toggle_selection("1.2.3")
        assert state.selection_locked is True
        assert state.selected_chunk_key == "1.2.3"

    def test_hover_ignored_when_locked(self):
        state = SelectionState()
        state.toggle_selection("0.1.2")
        state.set_hover("1.2.3")
        # Hover should be ignored when selection is locked
        assert state.effective_chunk_key == "0.1.2"

    def test_clear_selection(self):
        state = SelectionState()
        state.toggle_selection("0.1.2")
        state.clear_selection()
        assert state.selection_locked is False
        assert state.selected_chunk_key is None

    def test_toggle_selection_with_none_clears(self):
        state = SelectionState()
        state.toggle_selection("0.1.2")
        state.toggle_selection(None)
        assert state.selection_locked is False
        assert state.selected_chunk_key is None


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
