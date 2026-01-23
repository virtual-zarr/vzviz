"""Optional dependency handling for virtualizarr-viz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import holoviews as hv
    import matplotlib.pyplot as plt
    import panel as pn


def import_holoviews() -> Any:
    """Import holoviews and related packages, raising helpful error if missing."""
    try:
        import holoviews as hv

        hv.extension("bokeh")
        return hv
    except ImportError as e:
        raise ImportError(
            "holoviews is required for interactive visualization. "
            "Install with: pip install 'virtualizarr-viz[holoviews]' "
            "or: pip install holoviews hvplot bokeh"
        ) from e


def import_panel() -> Any:
    """Import panel, raising helpful error if missing."""
    try:
        import panel as pn

        return pn
    except ImportError as e:
        raise ImportError(
            "panel is required for dashboard functionality. "
            "Install with: pip install 'virtualizarr-viz[holoviews]' "
            "or: pip install panel"
        ) from e


def import_matplotlib() -> Any:
    """Import matplotlib, raising helpful error if missing."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for static visualization. "
            "Install with: pip install 'virtualizarr-viz[matplotlib]' "
            "or: pip install matplotlib"
        ) from e


def has_holoviews() -> bool:
    """Check if holoviews is available."""
    try:
        import holoviews  # noqa: F401

        return True
    except ImportError:
        return False


def has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def get_default_backend() -> str:
    """Get the default backend based on available packages."""
    if has_holoviews():
        return "holoviews"
    elif has_matplotlib():
        return "matplotlib"
    else:
        raise ImportError(
            "No visualization backend available. "
            "Install holoviews or matplotlib: "
            "pip install 'virtualizarr-viz[holoviews]' or "
            "pip install 'virtualizarr-viz[matplotlib]'"
        )
