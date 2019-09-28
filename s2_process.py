import os
import shutil

from pyeo.apps.model_creation.download_and_preproc_area import main as dl_and_preproc
from pyeo import raster_manipulation as ras
from pyeo import filesystem_utilities as fs

# # 2. s2
# # 2.1 download and preprocess :output 20m merge. tif (10 bands)
sen2cor_path = "/home/ubuntu/Downloads/Sen2Cor-02.08.00-Linux64/bin/L2A_Process"
pyeo_path = "/home/ubuntu/Documents/Code/pyeo/pyeo/apps/model_creation/download_and_preproc_area.py"
aoi_path = "/media/ubuntu/storage/Ghana/shp/geojson/cocoa_big.geojson"
l1_dir = "/media/ubuntu/storage/Ghana/s2/cocoa_big/L1"
l2_dir = "/media/ubuntu/storage/Ghana/s2/cocoa_big/L2"
merge_10m_dir = "/media/ubuntu/storage/Ghana/s2/cocoa_big/merge_10m"
merge_20m_dir = "/media/ubuntu/storage/Ghana/s2/cocoa_big/merge_20m"
conf = "/media/ubuntu/storage/Ghana/s2/cocoa_big/cocoa_big.ini"

#
dl_and_preproc(aoi_path=aoi_path, start_date='20180401', end_date='20180501',
               l1_dir=l1_dir, l2_dir=l2_dir, merge_dir=merge_10m_dir, conf_path=conf,
         download_l2_data=False, sen2cor_path=sen2cor_path, stacked_dir=None, resolution=10, cloud_cover='20',
               bands = ("B02", "B03", "B04", "B08"))


ras.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge_20m_dir, l1_dir=l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=("B05","B06","B07","B8A","B11","B12"), out_resolution=20)
# #sort into different scence id
for merged_tif in os.listdir(merge_10m_dir):
    tile_id = fs.get_sen_2_image_tile(merged_tif)
    try:
        os.mkdir(tile_id)
    except FileExistsError:
        pass
    shutil.move(merged_tif, tile_id)
test



# l1_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/L1"
# l2_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/L2"
# merge_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/merge_test"
# merge20_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/merge20m"
#
#
# #pyeo.raster_manipulation.atmospheric_correction(in_directory = l1_dir, out_directory = l2_dir, sen2cor_path= sen2cor_path, delete_unprocessed_image=False)
#
# pyeo.raster_manipulation.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge20_dir, l1_dir=l1_dir, cloud_threshold=0,
#                                                     buffer_size=5, bands=("B05","B06","B07","B8A","B11","B12"), out_resolution=20)

# pyeo.raster_manipulation.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge20_dir, l1_dir=l1_dir, cloud_threshold=0,
#                                                     buffer_size=5, bands=("B05","B06","B07"), out_resolution=20)
###### 10m bands


# mosaic_tif_out = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219.tif"
# s2_functions.mosaicTiff_text(input_dir="/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2",
#                                          outname= mosaic_tif_out,search_suffix='20180219T172348.tif')
#
# s2_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite.tif"
# os.system('gdalwarp -cutline '+  shp +' -crop_to_cutline '+ '-overwrite ' +mosaic_tif_out + ' '+ s2_testsite_tif)

##########20m bands
#
# l2_safe_list = ['/media/ubuntu/storage/Ghana/cocoa_upscale_test/L2/S2B_MSIL2A_20180219T103049_N0206_R108_T30NVN_20180219T172348.SAFE/GRANULE/L2A_T30NVN_A004996_20180219T104805/IMG_DATA/R20m/',
#                 '/media/ubuntu/storage/Ghana/cocoa_upscale_test/L2/S2B_MSIL2A_20180219T103049_N0206_R108_T30NWN_20180219T172348.SAFE/GRANULE/L2A_T30NWN_A004996_20180219T104805/IMG_DATA/R20m/']
#
# out_list = [os.path.join('/media/ubuntu/storage/Ghana/cocoa_upscale_test/20m','L2A_T30NVN_A004996_20180219T104805_20m.tif'),
#             os.path.join('/media/ubuntu/storage/Ghana/cocoa_upscale_test/20m', 'L2A_T30NWN_A004996_20180219T104805_20m.tif')]
# n=0
# for band_dir in l2_safe_list:
#     s2_functions.mergeTiff_text(input_dir=band_dir,outname=out_list[n],search_suffix='.jp2')
#     n+=1

# mosaic_tif_out = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_20m.tif"
# s2_functions.mosaicTiff_text(input_dir="/media/ubuntu/storage/Ghana/cocoa_upscale_test/20m",
#                                          outname= mosaic_tif_out,search_suffix='.tif')
# #resample to 10m
# mosaic_20m_resample = mosaic_tif_out[:-4] + '_10m.tif'
# os.system('gdalwarp -overwrite -tr 10 10 -r cubic ' + mosaic_tif_out + ' ' + mosaic_20m_resample)
#
#
# s2_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_20m_resample.tif"
# os.system('gdalwarp -cutline '+  shp +' -crop_to_cutline '+ '-overwrite ' +mosaic_20m_resample + ' '+ s2_testsite_tif)
#