# virtualizarr-viz

Visualization tools for [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr) chunk manifests.

## Installation

```bash
pip install virtualizarr-viz
```

For interactive visualizations (recommended):

```bash
pip install 'virtualizarr-viz[holoviews]'
```

For static matplotlib plots:

```bash
pip install 'virtualizarr-viz[matplotlib]'
```

## Usage

### Python API

```python
from virtualizarr import open_virtual_dataset
from virtualizarr_viz import byte_range_chart, chunk_file_heatmap, manifest_summary

# Open a virtual dataset
vds = open_virtual_dataset("data.nc")

# Byte range chart - shows chunk positions within files
byte_range_chart(vds, "temperature")

# Chunk-to-file heatmap - 2D grid colored by source file
chunk_file_heatmap(vds, "temperature")

# Summary statistics
summary = manifest_summary(vds, "temperature")
print(summary.T)

# Interactive dashboard with all visualizations
from virtualizarr_viz import manifest_dashboard
dashboard = manifest_dashboard(vds, "temperature")
dashboard.show()
```

### Command Line

```bash
# Byte range chart (default)
virtualizarr-viz data.nc -v temperature

# Heatmap
virtualizarr-viz data.nc -v temperature -k heatmap

# Summary statistics
virtualizarr-viz data.nc -v temperature -k summary

# Save to file
virtualizarr-viz data.nc -v temperature -o chunks.html

# Interactive dashboard
virtualizarr-viz data.nc -v temperature -k dashboard
```

## Visualizations

### Byte Range Chart

Shows chunk positions within each file as horizontal segments:

```
file1.nc  |====|  |====|      |====|
file2.nc  |==|    |====|  |==|
file3.nc  |========|          |====|
          0       500      1000     1500  (bytes)
```

- Each file is a row
- Segments show `[offset, offset+length)` for each chunk
- Reveals fragmentation, data locality, and gaps between chunks

### Chunk-to-File Heatmap

2D grid where each cell represents a chunk, colored by:
- Source file (categorical)
- Byte offset (continuous)
- Chunk length (continuous)

### Summary Statistics

DataFrames with:
- Overall statistics: total chunks, unique files, size distribution
- Per-file breakdown: chunk counts, byte ranges, contiguity

## API Reference

### Core Functions

- `extract_manifest(data, variable)` - Extract ChunkManifest from various inputs
- `manifest_to_dataframe(data, variable)` - Convert manifest to pandas DataFrame

### Visualizations

- `byte_range_chart(data, variable, ...)` - Byte range visualization
- `chunk_file_heatmap(data, variable, ...)` - 2D heatmap
- `manifest_summary(data, variable)` - Overall statistics
- `file_summary(data, variable)` - Per-file statistics
- `manifest_dashboard(data, variable, ...)` - Interactive dashboard

### Unified Entry Point

```python
visualize_manifest(data, variable, kind="byterange", **kwargs)
```

## License

Apache-2.0
