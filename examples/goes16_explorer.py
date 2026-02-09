#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "virtualizarr[hdf]",
#     "vzviz @ git+https://github.com/virtual-zarr/vzviz",
#     "obstore",
#     "panel",
#     "holoviews",
#     "bokeh",
#     "colorcet",
# ]
#
# ///
"""
GOES-16 Manifest Explorer

Launch an interactive dashboard to explore the chunk manifest of a GOES-16
ABI Level 2 Multi-Band Cloud and Moisture Imagery (MCMIPF) file from the
NOAA open data bucket on S3.

Usage:
    uv run examples/goes16_explorer.py
"""

from urllib.parse import urlparse

from obspec_utils.registry import ObjectStoreRegistry

import obstore as obs
import virtualizarr as vz
import vzviz

GOES16_URL = "s3://noaa-goes16/ABI-L2-MCMIPF/2024/099/18/OR_ABI-L2-MCMIPF-M6_G16_s20240991800204_e20240991809524_c20240991810005.nc"


def _per_band_var_names(var_name: str) -> list[str]:
    """Generate per-band variable names for all 16 ABI channels."""
    return [f"{var_name}_C{i:02}" for i in range(1, 17)]


# Per-band statistics variables to drop (scalar metadata, not useful for visualization)
DROP_VARIABLES = (
    _per_band_var_names("band_id")
    + _per_band_var_names("min_reflectance_factor")
    + _per_band_var_names("max_reflectance_factor")
    + _per_band_var_names("mean_reflectance_factor")
    + _per_band_var_names("std_dev_reflectance_factor")
    + _per_band_var_names("min_brightness_temperature")
    + _per_band_var_names("max_brightness_temperature")
    + _per_band_var_names("mean_brightness_temperature")
    + _per_band_var_names("std_dev_brightness_temperature")
    + _per_band_var_names("outlier_pixel_count")
)


def main():
    print("Setting up anonymous S3 access to noaa-goes16 bucket...")
    parsed = urlparse(GOES16_URL)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    store = obs.store.S3Store(
        bucket="noaa-goes16",
        region="us-east-1",
        skip_signature=True,
    )
    registry = ObjectStoreRegistry({base_url: store})

    print(f"Parsing GOES-16 file: {GOES16_URL}")
    parser = vz.parsers.HDFParser(
        drop_variables=DROP_VARIABLES,
    )
    manifest_store = parser(GOES16_URL, registry=registry)
    print("ManifestStore created!")

    print("\nLaunching dashboard...")
    dashboard = vzviz.manifest_dashboard(manifest_store)
    dashboard.show(title="GOES-16 Manifest Explorer")


if __name__ == "__main__":
    main()
