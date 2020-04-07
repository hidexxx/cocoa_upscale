"""
Ghana processing chain

Inputs needed;
-Area of interest
-Date range
-Maximum cloud cover
-Last cloud-free pixel composite
-Path to model
-Path to output
-Path to s1

Step 1: Download best S2 L1 and L2
Step 2: Download nearest S1 that covers AOI(now with the change of ESA download, s1 need to be processed separately)
Step 3: Generate cloud mask from S2 L1 and L2 OR generate composite
  --Composite must have all S2 bands
Step 4: Generate indicies from S2 bands
Step 5: Segment based on S2 bands - calling SAGA QGIS
Step 6: Stack indicies, S2  bands, S1 bands and segment bands
Step 7: Apply classifier
Step 8: Histogram matching over scenes

cocoa_forest_segregation
"""

import os, sys
import configparser
import shutil
import argparse
from tempfile import TemporaryDirectory
import subprocess
import numpy as np

import gdal
import osr

import pyeo.raster_manipulation as ras
import pyeo.queries_and_downloads as query
import pyeo.filesystem_utilities as fu
import pyeo.classification as cls

from cal_veg_index import generate_veg_index_tif

from s2_process import sort_into_tile

import general_functions
from segmentation_raster import cal_seg_mean
import PYEO_model

#TODO: Put this into Pyeo proper.
def segment_image(in_path, out_path, seeds_band_width = 5, cores = 1, shape_dir=None):
    """Uses SAGA to create a segmentation of the input raster."""
    with TemporaryDirectory() as td:
        temp_shape_path = os.path.join(td, "shapes.shp")
        saga_cmd = [
            "saga_cmd",
            #"--cores", str(corepaths),
            "imagery",
            "obia",
            "-FEATURES", in_path,
            "-OBJECTS", temp_shape_path,
            "-SEEDS_BAND_WIDTH", str(seeds_band_width)
        ]
        subprocess.run(saga_cmd)

        if shape_dir:
            shutil.copy(td, shape_dir)

        in_raster = gdal.Open(in_path)
        shape_projection = osr.SpatialReference()
        shape_projection.ImportFromWkt(in_raster.GetProjection())
        image_gt = in_raster.GetGeoTransform()
        x_res, y_res = image_gt[1], image_gt[5] * -1 # Negative 'y' values, you've foiled me again!
        width = in_raster.RasterXSize
        height = in_raster.RasterYSize

        ras_params = gdal.RasterizeOptions(
            noData=0,
            attribute="ID",
            xRes=x_res,
            yRes=y_res,
            outputType=gdal.GDT_UInt32,
            outputSRS=shape_projection,
            width = width,
            height= height
        )

        out_path = os.path.abspath(out_path)

        gdal.Rasterize(out_path, temp_shape_path, options=ras_params)

def make_directory(in_directory):
    try:
        os.mkdir(in_directory)
    except FileExistsError:
        pass


def build_cocoa_map(working_dir, path_to_aoi, start_date, end_date, path_to_s1_image, path_to_config,
                    epsg_for_map, path_to_model,
                    cloud_cover=20, log_path="build_cocoa_map.log", use_sen2cor=False,
                    sen2cor_path=None, skip_download_and_preprocess=False, skip_composite=False):

    # Step 0: Get things ready. Folder structure for downloads, load credentials from config.
    fu.init_log(log_path)
    os.chdir(working_dir)
    # fu.create_file_structure(os.getcwd())
    #
    # make_directory("images/merged/10m")
    # make_directory("images/merged/20m")
    #
    # make_directory("images/stacked/with_indices")
    # make_directory("images/stacked/with_s1_seg")
    # make_directory("images/stacked/all_19bands")
    #
    # make_directory("segmentation")
    # make_directory("composites")
    # make_directory("composites/10m")
    # make_directory("composites/20m")
    #
    # make_directory("composites/10m_full")
    # make_directory("composites/20m_full")
    general_functions.make_all_dirs(working_dir)


    # Step 1: Download S2 3imagery for the timescale
    if not skip_download_and_preprocess:
        config = configparser.ConfigParser()
        config.read(path_to_config)

        images_to_download = query.check_for_s2_data_by_date(path_to_aoi, start_date, end_date, config, cloud_cover)
        if not use_sen2cor:
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
                                   bands=("B02", "B03", "B04","B05", "B06", "B07", "B8A", "B11", "B12"),
                                   out_resolution=20)

    if not skip_composite:
        # Step 2.5: Build a pair of cloud-free composites
        sort_into_tile("images/merged/10m")
        sort_into_tile("images/merged/20m")

        for tile in os.listdir("images/merged/10m"):
            tile_path = os.path.join("images/merged/10m", tile)
            this_composite_path = ras.composite_directory(tile_path, "composites/10m")
            new_composite_path = "{}_{}.tif".format(this_composite_path.rsplit('.')[0], tile)
            os.rename(this_composite_path, new_composite_path)

        for tile in os.listdir("images/merged/20m"):
            tile_path = os.path.join("images/merged/20m", tile)
            this_composite_path = ras.composite_directory(tile_path, "composites/20m")
            new_composite_path = "{}_{}.tif".format(this_composite_path.rsplit('.')[0], tile)
            os.rename(this_composite_path, new_composite_path)

    # Step 3: Generate the bands. Time for the New Bit.
    clip_to_aoi = False
    if clip_to_aoi:
        for image in os.listdir("composites/10m_full"):
            if image.endswith(".tif"):
                image_path_10m_full = os.path.join("composites/10m_full", image)
                image_path_20m_full = os.path.join("composites/20m_full", image)

                image_path_10m_clipped = os.path.join("composites/10m", image)
                image_path_20m_clipped = os.path.join("composites/20m", image)

                # config = configparser.ConfigParser()
                # conf = config.read(path_to_config)
                # print(conf)
                # aoi = config['cocoa_mapping']['path_to_aoi']

                aoi = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"

                ras.clip_raster(raster_path=image_path_10m_full, aoi_path=aoi,
                                out_path=image_path_10m_clipped,srs_id = 32630)
                ras.clip_raster(raster_path=image_path_20m_full, aoi_path=aoi,
                                out_path=image_path_20m_clipped, srs_id=32630)


    do_segmentation = True
    if do_segmentation == True:
        for image in os.listdir("composites/10m"):
            if image.endswith(".tif"):
                with TemporaryDirectory() as td:
                    image_path_10m = os.path.join("composites/10m", image)
                    image_path_20m = os.path.join("composites/20m", image)
                    resample_path_20m_v1 = os.path.join(td, image)  # You'll probably regret this later, roberts.
                    shutil.copy(image_path_20m, resample_path_20m_v1)
                    ras.resample_image_in_place(resample_path_20m_v1, 10)

                    index_image_path = os.path.join(td, "index_image.tif")
                    temp_pre_seg_path = os.path.join(td, "pre_seg.tif")
                    temp_seg_path = os.path.join(td,"seg.tif")
                    temp_shp_path = os.path.join(td, "outline.shp")
                    temp_clipped_seg_path = os.path.join(td,"seg_clip.tif")

                    # This bit's your show, Qing
                    temp_s1_outline_path = os.path.join(td, "s1_outline.shp")
                    ras.get_extent_as_shp(
                        in_ras_path=image_path_10m,
                        out_shp_path=temp_s1_outline_path
                    )

                    resample_path_20m = os.path.join(td, image[:-4]+'_to_10moutline.tif')
                    general_functions.clip_rst(in_tif=resample_path_20m_v1, outline_shp=temp_s1_outline_path,
                                               out_tif=resample_path_20m, keep_rst_extent=False)


                    generate_veg_index_tif(image_path_10m, resample_path_20m, index_image_path)
                    ras.stack_images([index_image_path, image_path_10m], "images/stacked/with_indices/" + image)

                    # Now, we do Highly Experimental Image Segmentation. Please put on your goggles.
                    # SAGA, please.
                    # Meatball science time
                    vis_10m = gdal.Open(image_path_10m)
                    vis_20m_resampled = gdal.Open(resample_path_20m)
                    vis_10m_array = vis_10m.GetVirtualMemArray()
                    vis_20m_array = vis_20m_resampled.GetVirtualMemArray()
                    # NIR, SWIR, red
                    array_to_classify = np.stack([
                        vis_10m_array[3,...],
                        vis_20m_array[7,...],
                        vis_10m_array[2,...]
                    ]
                    )

                    g,arr = general_functions.read_tif(intif=image_path_10m)
                    general_functions.create_tif(filename=temp_pre_seg_path,g=g,Nx=arr.shape[1],Ny=arr.shape[2],
                                                 new_array=array_to_classify,noData=0,data_type=gdal.GDT_UInt32)
                    out_segment_tif = os.path.join("segmentation", image)
                    segment_image(temp_pre_seg_path, out_segment_tif)


                    print('Generate brighness raster from the segments')
                    make_directory("segmentation/brightness")

                    out_brightness_value_ras = os.path.join("segmentation/brightness", image)
                    output_filtered_value_ras = False

                    ras.get_extent_as_shp(
                        in_ras_path=temp_pre_seg_path,
                        out_shp_path=temp_shp_path
                    )


                    general_functions.clip_rst(in_tif=out_segment_tif, outline_shp=temp_shp_path,
                                               out_tif=temp_clipped_seg_path, keep_rst_extent=False)

                    cal_seg_mean(temp_pre_seg_path, temp_clipped_seg_path, out_brightness_value_ras,
                                 output_filtered_value_ras=output_filtered_value_ras)

                # image_20m_6bands_array = vis_20m_array[3:,:,:]
                # try:
                #     os.mkdir("composites/20m/20m_6bands")
                # except FileExistsError:
                #     pass
                #
                # out_20m_tif_for_stack = os.path.join("composites/20m/20m_6bands", image)
                # general_functions.create_tif(filename=out_20m_tif_for_stack,g=g,Nx=arr.shape[1],Ny=arr.shape[2],
                #                              new_array=image_20m_6bands_array,data_type=gdal.GDT_UInt16,noData=0)


    do_stack = True
    if do_stack == True:
    # Step 4: Stack the new bands with the S1, seg, and 6 band 20m rasters
        for image in os.listdir("images/stacked/with_indices"):
            if image.endswith(".tif"):
                path_to_image = os.path.join("images/stacked/with_indices", image)
                path_to_brightness_image = os.path.join("segmentation/brightness", image)
               # path_to_20m_image = os.path.join("composites/20m/20m_6bands", image)
               # ras.stack_images([path_to_image, path_to_s1_image,path_to_20m_image,out_brightness_value_ras], os.path.join("images/stacked/all_19bands", image))
                ras.stack_images([path_to_image, path_to_s1_image, path_to_brightness_image],
                                 os.path.join("images/stacked/with_s1_seg", image))

    #sys.exit()
    #
    # Step 5: Classify with trained model
    for image in os.listdir("images/stacked/with_s1_seg"):
        if image.endswith(".tif"):
            path_to_image = os.path.join("images/stacked/with_s1_seg", image)
            path_to_out = os.path.join("output", image)
            PYEO_model.classify_image(path_to_image, path_to_model, path_to_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A script for performing cocoa classification in Ghana. See cocoa_config.ini"
                                     "for the parameter information.")
    parser.add_argument("--path_to_config", help="The path to the config file containing the parameters"
                                                 "for this run of the cocoa map")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.path_to_config)

    cocoa_args = config['cocoa_mapping']

    # Note for future maintainers; the '**' operator unpacks a dictionary (in this case, cocoa_args) into a set
    # of keyword augments for a function. So this should pass the keywords in the config file straight into
    # the build_cocoa_map function without us having to retype them by hand every time they change.
    build_cocoa_map(path_to_config=args.path_to_config, **cocoa_args)
