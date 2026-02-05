"""Shared utilities for virtualizarr visualization scripts.

Configuration is controlled via the VVIZ_CONFIG environment variable:
    export VVIZ_CONFIG=local_netcdf    # Use local_netcdf_config.py
    export VVIZ_CONFIG=goes16          # Use goes16_config.py
    (unset or "default")               # Use built-in NEX-GDDP-CMIP6 config
"""

from __future__ import annotations

import importlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def _load_config():
    """Load configuration based on VVIZ_CONFIG environment variable."""
    config_name = os.environ.get("VVIZ_CONFIG", "default")

    if config_name == "default":
        return None  # Use built-in defaults

    # Try to import the config module
    try:
        return importlib.import_module(f"{config_name}_config")
    except ImportError as e:
        raise ImportError(
            f"Could not load config '{config_name}_config'. "
            f"Make sure {config_name}_config.py exists in the examples directory. "
            f"Original error: {e}"
        ) from e


# Load config module (if any)
_config = _load_config()

# Dataset configuration - from config module or built-in defaults
if _config and hasattr(_config, "DEFAULT_BUCKET"):
    DEFAULT_BUCKET = _config.DEFAULT_BUCKET
    DEFAULT_URL = _config.DEFAULT_URL
    DEFAULT_REGION = getattr(_config, "DEFAULT_REGION", None)
    CONFIG_NAME = os.environ.get("VVIZ_CONFIG", "default")
else:
    # Built-in default: NEX-GDDP-CMIP6
    DEFAULT_BUCKET = "s3://nex-gddp-cmip6"
    DEFAULT_PATH = "NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"
    DEFAULT_URL = f"{DEFAULT_BUCKET}/{DEFAULT_PATH}"
    DEFAULT_REGION = "us-west-2"
    CONFIG_NAME = "default"


def create_traced_registry(bucket: str, trace, region: str | None = DEFAULT_REGION):
    """Create an ObjectStoreRegistry with request tracing."""
    import obstore as obs
    from obspec_utils.registry import ObjectStoreRegistry
    from obspec_utils.tracing import TracingReadableStore

    # Use config's custom registry creator if available
    if _config and hasattr(_config, "create_traced_registry"):
        return _config.create_traced_registry(trace)

    # Default behavior for S3/cloud storage
    store_kwargs = {}
    if region:
        store_kwargs["region"] = region
        store_kwargs["skip_signature"] = True

    store = obs.store.from_url(bucket, **store_kwargs)
    traced_store = TracingReadableStore(store, trace)
    return ObjectStoreRegistry({bucket: traced_store})


def get_parser_and_reader_name():
    """
    Create the HDF parser and return reader description.

    Returns
    -------
    tuple
        (parser, reader_name) where parser is the HDFParser and reader_name is
        a human-readable description of the reader configuration.
    """
    # Use config's custom parser if available
    if _config and hasattr(_config, "get_parser"):
        return _config.get_parser()

    import virtualizarr as vz

    reader_name = "Eager"
    parser = vz.parsers.HDFParser()
    return parser, reader_name


def get_loadable_variables() -> list[str]:
    """Get the list of variables to load eagerly."""
    if _config and hasattr(_config, "get_loadable_variables"):
        return _config.get_loadable_variables()
    return []


def get_dataset_title() -> str:
    """Get a human-readable title for the current dataset configuration."""
    if _config and hasattr(_config, "DATASET_TITLE"):
        return _config.DATASET_TITLE
    if CONFIG_NAME == "default":
        return "NEX-GDDP-CMIP6"
    return CONFIG_NAME.replace("_", " ").title()


def print_config_info() -> None:
    """Print information about the current configuration."""
    print(f"Config: {CONFIG_NAME}")
    print(f"URL: {DEFAULT_URL}")
    if DEFAULT_REGION:
        print(f"Region: {DEFAULT_REGION}")


def extract_chunks_dataframe(vds) -> pd.DataFrame:
    """Extract all chunks from a virtual dataset into a DataFrame."""
    from virtualizarr.manifests import ManifestArray

    records = []
    for var_name in vds.data_vars:
        var = vds[var_name]
        if isinstance(var.data, ManifestArray):
            manifest = var.data.manifest
            for chunk_key, entry in manifest.dict().items():
                records.append(
                    {
                        "variable": var_name,
                        "chunk_key": chunk_key,
                        "path": entry["path"],
                        "start": entry["offset"],
                        "length": entry["length"],
                        "end": entry["offset"] + entry["length"],
                    }
                )
    return pd.DataFrame(records)


def compute_gaps(chunks_df: pd.DataFrame, file_size: int) -> list[dict]:
    """Compute gaps between chunks (non-chunk portions of the file)."""
    if chunks_df.empty:
        return [
            {
                "x0": 0,
                "x1": file_size,
                "length": file_size,
                "description": f"Entire file: 0 - {file_size:,}",
            }
        ]

    # Sort chunks by start position
    sorted_chunks = chunks_df.sort_values("start").reset_index(drop=True)

    gaps = []
    current_pos = 0

    for _, row in sorted_chunks.iterrows():
        chunk_start = int(row["start"])
        chunk_end = int(row["end"])

        # Gap before this chunk
        if chunk_start > current_pos:
            gaps.append(
                {
                    "x0": current_pos,
                    "x1": chunk_start,
                    "length": chunk_start - current_pos,
                    "description": f"Non-chunk: {current_pos:,} - {chunk_start:,}",
                }
            )

        # Move position to end of this chunk (handle overlapping chunks)
        current_pos = max(current_pos, chunk_end)

    # Gap after last chunk
    if current_pos < file_size:
        gaps.append(
            {
                "x0": current_pos,
                "x1": file_size,
                "length": file_size - current_pos,
                "description": f"Non-chunk: {current_pos:,} - {file_size:,}",
            }
        )

    return gaps


def save_trace_data(
    requests_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    output_path: Path,
    metadata: dict | None = None,
) -> None:
    """Save trace data to a JSON file for later analysis."""
    # Convert timestamps to human-readable ISO format
    requests_copy = requests_df.copy()
    if "timestamp" in requests_copy.columns:
        requests_copy["timestamp_iso"] = pd.to_datetime(
            requests_copy["timestamp"], unit="s"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        # Keep original timestamp for calculations, but also store readable version

    trace_data = {
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            **(metadata or {}),
        },
        "requests": requests_copy.to_dict(orient="records"),
        "chunks": chunks_df.to_dict(orient="records"),
    }

    with open(output_path, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"      Trace data saved to: {output_path}")


def load_trace_data(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load trace data from a JSON file."""
    with open(input_path) as f:
        trace_data = json.load(f)

    requests_df = pd.DataFrame(trace_data["requests"])
    chunks_df = pd.DataFrame(trace_data["chunks"])
    metadata = trace_data.get("metadata", {})

    return requests_df, chunks_df, metadata


def get_virtualizarr_branch(script_path: Path) -> str:
    """
    Extract the VirtualiZarr branch name from script metadata.

    Parameters
    ----------
    script_path
        Path to the script file containing the dependency specification.

    Returns
    -------
    str
        The branch name, or "unknown" if not found.
    """
    content = script_path.read_text()
    match = re.search(r'VirtualiZarr\.git@([^"\']+)', content)
    return match.group(1) if match else "unknown"


def get_file_size(
    bucket: str, file_path: str, region: str | None = DEFAULT_REGION
) -> int:
    """Get the size of a file in an object store or local filesystem."""
    # Use config's custom file size getter if available
    if _config and hasattr(_config, "get_file_size"):
        return _config.get_file_size()

    import obstore as obs

    store_kwargs = {}
    if region:
        store_kwargs["region"] = region
        store_kwargs["skip_signature"] = True

    store = obs.store.from_url(bucket, **store_kwargs)
    file_meta = obs.head(store, file_path)
    return file_meta["size"]


def format_mb(bytes_value: int | float) -> str:
    """Format a byte value as MB with 2 decimal places."""
    return f"{bytes_value / (1024 * 1024):.2f} MB"


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use in filenames."""
    return name.replace("(", "_").replace(")", "").replace("=", "_")
