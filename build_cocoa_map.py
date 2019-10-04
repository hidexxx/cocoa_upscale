"""
Ghana processing chain

Inputs needed;
-Area of interest
-Date range
-Maximum cloud cover
-Last cloud-free pixel composite
-Path to model
-Path to output

Step 1: Download best S2 L1 and L2
Step 2: Download nearest S1 that covers AOI
Step 3: Generate cloud mask from S2 L1 and L2 OR generate composite
  --Composite must have all S2 bands
Step 4: Generate indicies from S2 bands
Step 5: Stack indicies, S2 10m bands, S1 bands
Step 6: Segment based on stack
Step 7: Add segment ID layer to stack
Step 8: Apply classifier

cocoa_forest_segregation
"""

import os
import configparser
import shutil
import argparse
from tempfile import TemporaryDirectory

import pyeo.raster_manipulation as ras
import pyeo.queries_and_downloads as query
import pyeo.filesystem_utilities as fu
import pyeo.classification as cls

from cal_veg_index import generate_veg_index_tif


def build_cocoa_map(working_dir, path_to_aoi, start_date, end_date, path_to_s1_folder, path_to_config,
                    epsg_for_map, path_to_model,
                    cloud_cover=20, log_path="build_cocoa_map.log", use_sen2cor=False,
                    sen2cor_path=None, skip_download_and_preprocess = False):

    # Step 0: Get things ready. Folder structure for downloads, load credentials from config.
    fu.init_log(log_path)
    config = configparser.ConfigParser()
    config.read(path_to_config)
    os.chdir(working_dir)
    fu.create_file_structure(working_dir)
    os.mkdir("images/merged/10m")
    os.mkdir("images/merged/20m")
    os.mkdir("images/stacked/with_indices")
    os.mkdir("images/stacked/with_s1")

    if not skip_download_and_preprocess:
        # Step 1: Download S2 3imagery for the timescale
        images_to_download = query.check_for_s2_data_by_date(path_to_aoi, start_date, end_date, cloud_cover)
        if  not use_sen2cor:
            images_to_download = query.filter_non_matching_s2_data(images_to_download)
        else:
            images_to_download = query.filter_to_l1_data(images_to_download)
        query.download_s2_data(images_to_download, "images/L1", "images/L2")

        # Step 2: Preprocess S2 imagery. Perform atmospheric correction if needed, stack and mask 10 and 20m bands.
        if use_sen2cor:
            ras.atmospheric_correction("images/L1" "images/L2", sen2cor_path=sen2cor_path)
        ras.preprocess_sen2_images("images/L2", "images/merged/10m", "images/L1",
                                   cloud_threshold=0, epsg= epsg_for_map,
                                   bands=("B02", "B03", "B04", "B08"),
                                   out_resolution=10)
        ras.preprocess_sen2_images("images/L2", "images/merged/20m", "images/L1",
                                   cloud_threshold=0, epsg= epsg_for_map,
                                   bands=("B05", "B06", "B07", "B8A", "B11", "B12"),
                                   out_resolution=20)

    # Could put composite here, but later.

    # Step 3: Generate the bands. Time for the New Bit.
    with TemporaryDirectory() as td:
        for image in os.listdir("images/merged/10m"):
            if image.endswith(".tif"):
                image_path_10m = os.path.join("images/merged/10m", image)
                image_path_20m = os.path.join("images/merged/20m", image)
                resample_path_20m = os.path.join(td, image)  # You'll probably regret this later, roberts.
                shutil.copy(image_path_20m, resample_path_20m)
                ras.resample_image_in_place(resample_path_20m, 10)

                # This bit's your show, Qing
                generate_veg_index_tif(image_path_10m, image_path_20m, "images/stacked/with_indices"+image)

    # Step 4: Stack the new bands with the S1 rasters
    for image in os.listdir("images/stacked/with_indices"):
        if image.endswith(".tif"):
            image_granule = fu.get_sen_2_granule_id(image)
            s1_granule_images = [s1_image for s1_image in os.listdir(path_to_s1_folder) if image_granule in s1_image]
            s1_granule_path = os.path.join(path_to_s1_folder, s1_granule_images[0])  # Going with the first image with the granule in the name for nwo
            ras.stack_images([image, s1_granule_path], os.path.join("images/stacked/with_s1", image))

    # Step 5: Classify with trained model
    cls.classify_directory("images/stacked/with_s1", path_to_model, "output", None, apply_mask=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script for performing cocoa classification in Ghana")
    parser.add_argument("working_dir", help="The location to keep the working imagery.")
    parser.add_argument("path_to_aoi", help="A path to a geojson containing the AOI.")
    parser.add_argument("start_date", help="The earlisest date to search for imagery. Format yyyymmdd.")
    parser.add_argument("end_date", help="The latest date to search for imagery. Format yyyymmdd.")
    parser.add_argument("path_to_s1_folder", help="The path to the folder containg a stack of hh and hv imagery")
    parser.add_argument("path_to_config", help="Path to the .ini file with your scihub credentials.")
    parser.add_argument("epsg_for_map", help="The EPSG number for the output map.")
    parser.add_argument("path_to_model", help="The path to a .pkl of a trained scikit-learn model")
    parser.add_argument("--cloud_cover", default=20, help="The maximum cloud cover")
    parser.add_argument("--path_to_log", default="build_cocoa_map.log", help="Where to store the logfile for this run.")
    parser.add_argument("--use_sen2cor", default=False, action="store_true",
                        help="If present, skip downloading of L2 data and use sen2cor for atmospheric processing.")
    parser.add_argument("--sen2cor_path", default=None, help="The path to L2AProcess. Only required if use_sen2cor"
                                                             "is True.")

    parser.add_argument("--skip_download_and_preprocess", default=False, action="store_true",
                        help="If present, skips download and preprocess")

    args = parser.parse_args()

    build_cocoa_map(args.working_dir, args.path_to_aoi, args.start_date, args.end_date, args.path_to_s1_folder,
                    args.path_to_config, args.espg_for_map, args.path_to_model, args.cloud_cover, args.path_to_log,
                    args.use_sen2cor, args.sen2cor_path)
