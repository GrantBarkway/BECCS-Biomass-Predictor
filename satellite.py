import openeo
import json
import random
from datetime import date, timedelta
import time

def date_to_string(date):
    return date.strftime("%Y-%m-%d")

def random_date_pair(year_start=2019, year_end=2023, gap_days=30):
    """
    Randomly select a start date such that both the start date
    and (start date + gap_days) fall within [year_start, year_end].
    Returns a tuple (date1, date2).
    """
    range_start = date(year_start, 1, 1)
    range_end = date(year_end, 12, 31) - timedelta(days=gap_days)
    
    total_days = (range_end - range_start).days
    offset = random.randint(0, total_days)

    date1 = range_start + timedelta(days=offset)
    date2 = date1 + timedelta(days=gap_days)
    
    return date1, date2

def get_images(geojson_filepath, max_retries=3):
    # connect and authenticate
    connection = openeo.connect("openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()
    
    with open(geojson_filepath) as f:
        as_json = json.load(f)

    for prov in as_json["features"]:
        province_name = prov["properties"]["NAME_1"]
        province_geom = prov["geometry"]
        
        for attempt in range(1,max_retries):
            start_date, end_date = random_date_pair(2019,2023,14)
            start_str, end_str = date_to_string(start_date), date_to_string(end_date)
            
            try:
                datacube = connection.load_collection(
                    "SENTINEL2_L2A",
                    spatial_extent=province_geom,
                    temporal_extent=[start_date, end_date],
                    bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                    properties={"eo:cloud_cover": lambda x: x <= 10},
                )
                
                datacube = datacube.resample_spatial(resolution=10)
                datacube = datacube.reduce_dimension(dimension="t", reducer="median")

                job = datacube.create_job(out_format="GTiff")
                job.start_and_wait()
                job.get_results().download_files(f"data/raw/{start_str}_{end_str}_{province_name}")
                break
            
            except Exception as e:
                print(f"Error: {e} with province: {province_name}")
                time.sleep(10)
    
    return

get_images("data/samut-songkhram-simplified.geojson", 5)