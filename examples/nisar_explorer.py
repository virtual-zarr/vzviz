#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "earthaccess",
#     "virtualizarr[hdf] @ git+https://github.com/maxrjones/VirtualiZarr@c-dtype",
#     "vzviz @ git+https://github.com/virtual-zarr/vzviz",
#     "obspec-utils @ git+https://github.com/developmentseed/obspec-utils",
#     "aiohttp",
#     "panel",
#     "holoviews",
#     "bokeh",
#     "colorcet",
# ]
# ///
"""
NISAR Manifest Explorer

Launch an interactive dashboard to explore the chunk manifest of a NISAR HDF5 file.

Usage:
    uv run examples/nisar_explorer.py
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

    print("Searching for NISAR data...")
    query = earthaccess.DataGranules()
    query.short_name("NISAR_L2_GCOV_BETA_V1")
    query.params["attribute[]"] = "int,FRAME_NUMBER,77"
    query.params["attribute[]"] = "int,TRACK_NUMBER,5"
    results = query.get_all()
    print(f"Found {len(results)} granules")

    # Get the HTTPS URL
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

    print("Parsing HDF5 file...")
    parser = vz.parsers.HDFParser()
    manifest_store = parser(https_url, registry=registry)
    print("ManifestStore created!")

    # Create and serve dashboard
    print("\nLaunching dashboard...")
    dashboard = vzviz.manifest_dashboard(manifest_store)
    dashboard.show(title="NISAR Manifest Explorer")


if __name__ == "__main__":
    main()
