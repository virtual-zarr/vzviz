"""Command-line interface for virtualizarr-viz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="virtualizarr-viz",
        description="Visualize chunk storage locations in VirtualiZarr manifests.",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to virtual dataset (Kerchunk JSON/Parquet, or file to virtualize)",
    )

    parser.add_argument(
        "-v", "--variable",
        type=str,
        default=None,
        help="Variable name to visualize",
    )

    parser.add_argument(
        "-k", "--kind",
        type=str,
        choices=["byterange", "heatmap", "summary", "dashboard"],
        default="byterange",
        help="Visualization type (default: byterange)",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (HTML for interactive, PNG for static)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=30,
        help="Maximum files to show in byte range chart (default: 30)",
    )

    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["name", "chunks", "bytes", "offset"],
        default="offset",
        help="Sort files by (default: offset)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["holoviews", "matplotlib"],
        default=None,
        help="Visualization backend (default: auto-detect)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=900,
        help="Plot width in pixels (default: 900)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Plot height in pixels (default: auto)",
    )

    args = parser.parse_args(argv)

    try:
        return _run(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    """Execute the visualization."""
    from virtualizarr_viz._compat import get_default_backend

    # Load the dataset
    data = _load_data(args.input)

    # Determine backend
    backend = args.backend or get_default_backend()

    # Run visualization
    if args.kind == "summary":
        _run_summary(data, args.variable)
        return 0

    if args.kind == "byterange":
        plot = _run_byterange(data, args, backend)
    elif args.kind == "heatmap":
        plot = _run_heatmap(data, args, backend)
    elif args.kind == "dashboard":
        plot = _run_dashboard(data, args)
    else:
        raise ValueError(f"Unknown visualization kind: {args.kind}")

    # Output
    if args.output:
        _save_output(plot, args.output, backend, args.kind)
        print(f"Saved to {args.output}")
    else:
        _show_output(plot, backend, args.kind)

    return 0


def _load_data(input_path: str):
    """Load data from various input formats."""
    from virtualizarr import open_virtual_dataset

    path = Path(input_path)

    if path.suffix == ".json":
        # Kerchunk JSON reference file
        return open_virtual_dataset(input_path, filetype="kerchunk")
    elif path.suffix == ".parquet" or path.is_dir():
        # Kerchunk Parquet reference
        return open_virtual_dataset(input_path, filetype="kerchunk")
    else:
        # Try to open as a regular file (NetCDF, HDF5, etc.)
        return open_virtual_dataset(input_path)


def _run_summary(data, variable: str | None) -> None:
    """Run summary visualization."""
    from virtualizarr_viz.summary import file_summary, manifest_summary

    print("\n=== Manifest Summary ===\n")
    summary = manifest_summary(data, variable)
    for col in summary.columns:
        print(f"{col}: {summary[col].iloc[0]}")

    print("\n=== Per-File Summary ===\n")
    files = file_summary(data, variable)
    print(files.to_string())


def _run_byterange(data, args: argparse.Namespace, backend: str):
    """Run byte range visualization."""
    from virtualizarr_viz.byterange import byte_range_chart

    return byte_range_chart(
        data,
        variable=args.variable,
        max_files=args.max_files,
        sort_by=args.sort_by,
        backend=backend,
        width=args.width,
        height=args.height,
    )


def _run_heatmap(data, args: argparse.Namespace, backend: str):
    """Run heatmap visualization."""
    from virtualizarr_viz.heatmap import chunk_file_heatmap

    return chunk_file_heatmap(
        data,
        variable=args.variable,
        backend=backend,
        width=args.width,
        height=args.height or 400,
    )


def _run_dashboard(data, args: argparse.Namespace):
    """Run dashboard visualization."""
    from virtualizarr_viz.dashboard import manifest_dashboard

    return manifest_dashboard(
        data,
        variable=args.variable,
    )


def _save_output(plot, output_path: str, backend: str, kind: str) -> None:
    """Save plot to file."""
    path = Path(output_path)

    if kind == "dashboard":
        # Dashboard is always saved as HTML
        plot.save(output_path)
        return

    if backend == "holoviews":
        import holoviews as hv

        if path.suffix.lower() in [".html", ".htm"]:
            hv.save(plot, output_path)
        elif path.suffix.lower() == ".png":
            hv.save(plot, output_path, fmt="png")
        else:
            hv.save(plot, output_path)
    elif backend == "matplotlib":
        plot.savefig(output_path, dpi=150, bbox_inches="tight")


def _show_output(plot, backend: str, kind: str) -> None:
    """Display plot interactively."""
    if kind == "dashboard":
        plot.show()
        return

    if backend == "holoviews":
        import holoviews as hv

        # Try to display in browser
        try:
            import panel as pn

            pn.extension()
            pn.serve(plot, show=True)
        except ImportError:
            # Fall back to saving temp file
            import tempfile
            import webbrowser

            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                hv.save(plot, f.name)
                webbrowser.open(f"file://{f.name}")
    elif backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    sys.exit(main())
