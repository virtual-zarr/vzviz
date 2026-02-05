"""Selection state management for cross-panel synchronization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import param

if TYPE_CHECKING:
    import pandas as pd


class SelectionState(param.Parameterized):
    """
    Centralized selection state for cross-panel synchronization.

    This class manages region selection state across multiple visualization
    panels, enabling linked brushing behavior via box selection in the heatmap
    and variable selection in the overview table.

    Parameters
    ----------
    bounds : tuple or None
        The selected region bounds as (x_min, y_min, x_max, y_max) in
        heatmap dimension coordinates.
    selected_variables : list or None
        List of selected variable names from the overview table.
    dim_x : int
        Which dimension is displayed on the x-axis of the heatmap.
    dim_y : int
        Which dimension is displayed on the y-axis of the heatmap.

    Examples
    --------
    >>> state = SelectionState()
    >>> state.set_bounds((0, 0, 5, 10))
    >>> state.has_selection
    True
    >>> state.clear_selection()
    >>> state.has_selection
    False
    """

    bounds = param.Tuple(
        default=None,
        allow_None=True,
        length=4,
        doc="Selected region bounds as (x_min, y_min, x_max, y_max).",
    )

    selected_variables = param.List(
        default=[],
        doc="List of selected variable names.",
    )

    dim_x = param.Integer(
        default=0,
        doc="Which dimension is displayed on x-axis.",
    )

    dim_y = param.Integer(
        default=1,
        doc="Which dimension is displayed on y-axis.",
    )

    @property
    def has_selection(self) -> bool:
        """Return True if any selection is active."""
        return self.bounds is not None or len(self.selected_variables) > 0

    @property
    def has_variable_selection(self) -> bool:
        """Return True if variables are selected."""
        return len(self.selected_variables) > 0

    def set_bounds(self, bounds: tuple[float, float, float, float] | None) -> None:
        """
        Set the selection bounds.

        Parameters
        ----------
        bounds : tuple or None
            (x_min, y_min, x_max, y_max) or None to clear.
        """
        if bounds is not None:
            # Normalize bounds so min < max
            x_min, y_min, x_max, y_max = bounds
            self.bounds = (
                min(x_min, x_max),
                min(y_min, y_max),
                max(x_min, x_max),
                max(y_min, y_max),
            )
        else:
            self.bounds = None

    def clear_selection(self) -> None:
        """Clear any selection."""
        self.bounds = None
        self.selected_variables = []

    def set_selected_variables(self, variables: list[str]) -> None:
        """Set the selected variables."""
        self.selected_variables = list(variables) if variables else []

    def get_selected_chunk_keys(self, df: "pd.DataFrame") -> set[str]:
        """
        Get all chunk keys matching the current selection.

        Selection can be based on:
        - Region bounds from heatmap box selection
        - Selected variables from overview table

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with chunk data including dim_X columns and variable column.

        Returns
        -------
        set[str]
            Set of chunk_key values matching the selection criteria.
        """
        if not self.has_selection:
            return set()

        selected_keys = set()

        # Filter by selected variables
        if self.selected_variables and "variable" in df.columns:
            var_mask = df["variable"].isin(self.selected_variables)
            selected_keys.update(df.loc[var_mask, "chunk_key"].astype(str))

        # Filter by bounds (only if bounds are set)
        if self.bounds is not None:
            x_min, y_min, x_max, y_max = self.bounds
            x_col = f"dim_{self.dim_x}"
            y_col = f"dim_{self.dim_y}"

            if x_col in df.columns and y_col in df.columns:
                bounds_mask = (
                    (df[x_col] >= int(round(x_min)))
                    & (df[x_col] <= int(round(x_max)))
                    & (df[y_col] >= int(round(y_min)))
                    & (df[y_col] <= int(round(y_max)))
                )
                selected_keys.update(df.loc[bounds_mask, "chunk_key"].astype(str))

        return selected_keys


def get_chunk_info(chunk_key: str, df: "pd.DataFrame") -> dict | None:
    """
    Get full chunk information for display.

    Parameters
    ----------
    chunk_key : str
        The chunk key to look up.
    df : pd.DataFrame
        DataFrame containing chunk data.

    Returns
    -------
    dict or None
        Dictionary with chunk details (path, offset, length, etc.),
        or None if chunk not found.
    """
    matches = df[df["chunk_key"] == chunk_key]
    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "chunk_key": chunk_key,
        "path": row.get("path"),
        "filename": row.get("filename"),
        "offset": row.get("offset"),
        "length": row.get("length"),
        "variable": row.get("variable"),
    }
