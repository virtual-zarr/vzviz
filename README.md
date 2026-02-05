# vzviz

Visualization and analysis tools for [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr) ManifestStore.

Inspired by [vischunk](https://github.com/jkeifer/vischunk) (MIT License, Copyright (c) 2025 Jarrett Keifer) - the query simulation metrics and cross-panel selection architecture are adapted from vischunk's approach.

## Installation

```bash
pip install vzviz @ git+https://github.com/virtual-zarr/vzviz
```

## Usage

All functions work directly with `ManifestStore` - the output of VirtualiZarr parsers:

```python
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
import vzviz

# Parse a file to get a ManifestStore
parser = HDFParser()
store = parser(url, registry)

# Overview of all variables
overview = vzviz.variables_overview(store)
print(overview)

# Detailed chunk grid info for a variable
info = vzviz.chunk_grid_info(store, "science/LSAR/data")
print(info)

# Simulate a query and get vischunk-like metrics
metrics = vzviz.simulate_query(
    store,
    "science/LSAR/data",
    query={0: slice(0, 100), 1: slice(0, 500)}
)
print(metrics)
# Output:
#   Requested Cells:   50,000
#   Cells Read:        125,000
#   Read Amplification: 2.50x
#   Read Efficiency:   40.0%
#   Chunks Touched:    25
#   Range Reads:       12
#   Coalescing Factor: 2.08x

# Compare different query patterns
comparison = vzviz.compare_queries(
    store,
    "temperature",
    queries=[
        {0: slice(0, 10)},      # Time slice
        {1: slice(0, 50), 2: slice(0, 50)},  # Spatial subset
    ],
    names=["time_slice", "spatial_subset"]
)
print(comparison)

# Byte range chart - shows chunk positions within files
vzviz.byte_range_chart(store, "temperature")

# Chunk-to-file heatmap - 2D grid colored by source file
vzviz.chunk_file_heatmap(store, "temperature")

# Interactive dashboard with cross-panel selection
# Click a chunk in one panel to highlight it in all panels
dashboard = vzviz.manifest_dashboard(store)
dashboard.show()
```

## Features

### Variables Overview

Shows all variables in a file with their shapes, chunk sizes, and storage statistics:

```python
overview = vzviz.variables_overview(store)
```

| variable | shape | chunks | dtype | total_chunks | chunk_bytes_human | total_bytes_human |
|----------|-------|--------|-------|--------------|-------------------|-------------------|
| science/LSAR/data | (1000, 2000, 500) | (100, 200, 50) | float32 | 500 | 4.0 MB | 2.0 GB |

### Query Simulation (vischunk-inspired)

Simulate data access patterns and get performance metrics:

```python
metrics = vzviz.simulate_query(store, "variable", {0: slice(0, 100)})
```

**Metrics returned:**
- **Requested Cells**: Cells in your query region
- **Cells Read**: Total cells that must be read due to chunking
- **Read Amplification**: Ratio of read to requested (lower is better)
- **Read Efficiency**: Percentage of useful data (higher is better)
- **Chunks Touched**: Number of chunks intersecting the query
- **Range Reads**: Number of separate I/O operations needed
- **Coalescing Factor**: How well chunks combine into fewer reads

### Byte Range Chart

Visualizes where chunks are located within each file:

```
file.nc  |====|  |====|      |====|
         0      500     1000    1500  (bytes)
```

### Chunk-to-File Heatmap

2D grid showing chunk-to-file mapping, useful for understanding data locality.

### Interactive Dashboard

The dashboard combines all visualizations with **cross-panel selection**:
- Click a chunk in the byte range chart or heatmap to select it
- The selected chunk is highlighted in all panels simultaneously
- A selection panel shows detailed information about the selected chunk
- Click again to deselect

```python
dashboard = vzviz.manifest_dashboard(store, variable="temperature")
dashboard.show()
```

## API Reference

### Core Functions

- `manifest_to_dataframe(store, variable=None)` - Convert to pandas DataFrame
- `get_array(store, variable)` - Get a specific ManifestArray
- `list_variables(store)` - List all variable paths
- `get_store_info(store)` - Get store statistics

### Variable Analysis

- `variables_overview(store)` - Overview table of all variables
- `chunk_grid_info(store, variable)` - Detailed chunk grid info

### Query Simulation

- `simulate_query(store, variable, query)` - Simulate a query and get metrics
- `compare_queries(store, variable, queries, names)` - Compare multiple queries

### Visualizations

- `byte_range_chart(store, variable, ...)` - Byte range visualization
- `byte_range_chart_interactive(store, variable, selection_state, ...)` - With selection support
- `chunk_file_heatmap(store, variable, ...)` - 2D chunk heatmap
- `chunk_file_heatmap_interactive(store, variable, selection_state, ...)` - With selection support
- `manifest_summary(store, variable)` - Overall statistics
- `file_summary(store, variable)` - Per-file statistics
- `manifest_dashboard(store, variable, interactive=True)` - Interactive dashboard

### Selection State

- `SelectionState` - Shared state for cross-panel selection synchronization

## License

Apache-2.0
