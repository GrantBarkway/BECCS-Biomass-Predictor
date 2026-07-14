import openeo
import json
import random
import time
import traceback
import os
import csv
from datetime import date, timedelta
import shutil

import rasterio
import numpy as np
from pystac_client import Client

STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
STAC_COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
MAX_CLOUD_COVER = 10
MIN_VALID_FRACTION = 0.90
OUTPUT_DIR = "data/raw"
MANIFEST_PATH = "data/raw/manifest.csv"

def date_to_string(d):
    return d.strftime("%Y-%m-%d")

def random_date_pair(year_start=2019, year_end=2023, gap_days=14):
    range_start = date(year_start, 1, 1)
    range_end = date(year_end, 12, 31) - timedelta(days=gap_days)
    total_days = (range_end - range_start).days
    offset = random.randint(0, total_days)
    date1 = range_start + timedelta(days=offset)
    date2 = date1 + timedelta(days=gap_days)
    return date1, date2

def has_data(catalog, geometry, start_str, end_str, max_cloud_cover=MAX_CLOUD_COVER):
    """Cheap pre-check so no credits used on empty date ranges."""
    try:
        search = catalog.search(
            collections=[STAC_COLLECTION],
            intersects=geometry,
            datetime=f"{start_str}/{end_str}",
            filter={"op": "<=", "args": [{"property": "eo:cloud_cover"}, max_cloud_cover]},
            max_items=1,
        )
        return next(search.items(), None) is not None
    except Exception as e:
        print(f"  STAC search failed: {e}")
        return False

def validate_tif(filepath, expected_band_count=len(BANDS), min_valid_fraction=MIN_VALID_FRACTION):
    """
    Confirms the downloaded file is actually usable training data, 
    not just that download_files() ran without raising.
    """
    try:
        with rasterio.open(filepath) as src:
            if src.count != expected_band_count:
                return False, {"reason": f"expected {expected_band_count} bands, got {src.count}"}

            arr = src.read()
            nodata = src.nodata
            is_float = np.issubdtype(arr.dtype, np.floating)

            if nodata is not None:
                valid_mask = arr != nodata
            elif is_float:
                valid_mask = ~np.isnan(arr)
            else:
                valid_mask = np.ones_like(arr, dtype=bool)
            
            valid_fraction = float(np.mean(valid_mask))
            has_nan = bool(np.isnan(arr).any()) if is_float else False

            stats = {
                "shape": list(src.shape),
                "band_count": src.count,
                "dtype": str(src.dtypes[0]),
                "valid_fraction": round(valid_fraction, 4),
                "has_nan": has_nan,
            }
            
            if valid_fraction < min_valid_fraction:
                stats["reason"] = f"valid_fraction {valid_fraction:.2%} below threshold {min_valid_fraction:.2%}"
                return False, stats
            
            if has_nan and nodata is not None:
                stats["reason"] = "NaNs present alongside a defined nodata value (unexpected)"
                return False, stats
            
            return True, stats
    
    except Exception as e:
        return False, {"reason": f"could not open/read file: {e}"}

def append_manifest(row):
    file_exists = os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def get_images(geojson_filepath, max_retries=5, samples_per_province=1):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()
    catalog = Client.open(STAC_URL)
    
    with open(geojson_filepath) as f:
        as_json = json.load(f)

    provinces = list(as_json["features"])
    random.shuffle(provinces)

    for prov in provinces:
        province_name = prov["properties"]["NAME_1"]
        province_geom = prov["geometry"]
        successes = 0
        attempt = 0
        
        while successes < samples_per_province and attempt < max_retries:
            attempt += 1
            start_date, end_date = random_date_pair(2019, 2023, 14)
            start_str, end_str = date_to_string(start_date), date_to_string(end_date)
            target = f"{OUTPUT_DIR}/{start_str}_{end_str}_{province_name}"  # no extension, matches known-good behavior
            out_path = f"{target}.tif"
            
            print(f"[{province_name}] attempt {attempt}/{max_retries}: {start_str} to {end_str}")

            if not has_data(catalog, province_geom, start_str, end_str):
                print("  No suitable scenes found, trying a different date range.")
                continue
            
            job = None
            try:
                connection.authenticate_oidc()
                
                datacube = connection.load_collection(
                    "SENTINEL2_L2A",
                    spatial_extent=province_geom,
                    temporal_extent=[start_date, end_date],
                    bands=BANDS,
                    properties={"eo:cloud_cover": lambda x: x <= MAX_CLOUD_COVER},
                )
                datacube = datacube.resample_spatial(resolution=10)
                datacube = datacube.reduce_dimension(dimension="t", reducer="median")
                
                job = datacube.create_job(out_format="GTiff")
                job.start_and_wait()

                if job.status() != "finished":
                    print(f"  Job status was '{job.status()}', not 'finished' — retrying.")
                    continue

                results = job.get_results()
                assets = results.get_assets()
                if not assets:
                    print("  Job finished but returned zero assets — retrying.")
                    continue
                
                downloaded_paths = results.download_files(target)
                tif_paths = [p for p in downloaded_paths if str(p).lower().endswith((".tif", ".tiff"))]
                
                if not tif_paths:
                    print(f"  No .tif among downloaded files: {downloaded_paths} — retrying.")
                    continue

                if len(tif_paths) > 1:
                    print(f"  WARNING: multiple .tif files returned for {province_name}, using first: {tif_paths}")

                raw_tif_path = str(tif_paths[0])
                final_path = f"{OUTPUT_DIR}/{start_str}_{end_str}_{province_name}.tif"
                
                shutil.move(raw_tif_path, final_path)
                
                leftover_dir = os.path.dirname(raw_tif_path)
                if leftover_dir != OUTPUT_DIR and os.path.isdir(leftover_dir):
                    shutil.rmtree(leftover_dir, ignore_errors=True)

                out_path = final_path

                is_valid, stats = validate_tif(out_path)
                if not is_valid:
                    print(f"  Failed validation ({stats.get('reason')}) — deleting and retrying.")
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    continue
                successes += 1
                print(f"  Valid ({stats['valid_fraction']:.1%} valid pixels).")
                append_manifest({
                    "province": province_name,
                    "start_date": start_str,
                    "end_date": end_str,
                    "path": out_path,
                    **stats,
                })
            
            except Exception as e:
                print(f"  Error: {e}")
                traceback.print_exc()
                if job is not None:
                    try:
                        print(job.logs())
                    except Exception:
                        pass
                time.sleep(10)
        
        if successes < samples_per_province:
            print(f"WARNING: only {successes}/{samples_per_province} valid samples for {province_name} after {max_retries} attempts.")

get_images("data/thailand-provinces-no-islands.geojson", max_retries=5, samples_per_province=5)