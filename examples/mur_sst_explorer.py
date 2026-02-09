#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "earthaccess",
#     "virtualizarr[hdf]",
#     "vzviz @ git+https://github.com/virtual-zarr/vzviz",
#     "obspec-utils",
#     "aiohttp",
#     "panel",
#     "holoviews",
#     "bokeh",
#     "colorcet",
# ]
# ///
"""
MUR SST Manifest Explorer

Launch an interactive dashboard to explore the chunk manifest of a MUR SST file.

Usage:
    uv run examples/mur_sst_explorer.py
"""

from urllib.parse import urlparse

import earthaccess
import virtualizarr as vz
import vzviz

from obspec_utils.registry import ObjectStoreRegistry
from obspec_utils.stores import AiohttpStore


def main():
    print("Authenticating with NASA Earthdata...")
    earthaccess.login()

    print("Searching for MUR SST data...")
    results = earthaccess.search_data(
        concept_id="C1996881146-POCLOUD",
        count=1,
        temporal=("2002-06-01", "2002-06-01"),
    )

    https_links = earthaccess.results.DataGranule.data_links(
        results[0], access="external"
    )
    https_url = https_links[0]
    print(f"URL: {https_url}")

    # Parse URL and get auth token
    parsed = urlparse(https_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    token = earthaccess.get_edl_token()["access_token"]

    # Create store with authentication
    store = AiohttpStore(
        base_url,
        headers={"Authorization": f"Bearer {token}"},
    )
    registry = ObjectStoreRegistry({base_url: store})

    print("Parsing NetCDF file...")
    parser = vz.parsers.HDFParser()
    manifest_store = parser(https_url, registry=registry)
    print("ManifestStore created!")

    # Create and serve dashboard
    print("\nLaunching dashboard...")
    dashboard = vzviz.manifest_dashboard(manifest_store)
    dashboard.show(title="MUR SST Manifest Explorer")


if __name__ == "__main__":
    main()
