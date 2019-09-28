import os
import shutil
from pyeo.apps.model_creation.download_and_preproc_area import main as dl_and_preproc
from pyeo import raster_manipulation as ras
from pyeo import filesystem_utilities as fs

# # 2. s2
# # 2.1 download and preprocess :output 20m merge. tif (10 bands)
sen2cor_path = "/home/ubuntu/Downloads/Sen2Cor-02.08.00-Linux64/bin/L2A_Process"

aoi_path = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big_simp.geojson"
conf = "/media/ubuntu/Data/Ghana/cocoa_big/s2/cocoa_big.ini"

l1_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/L1"
l2_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/L2"
merge_10m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/merge_10m"
merge_20m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/merge_20m"


#
def do_dl_and_preproc():
    dl_and_preproc(aoi_path=aoi_path, start_date='20180401', end_date='20180501',
                   l1_dir=l1_dir, l2_dir=l2_dir, merge_dir=merge_10m_dir, conf_path=conf,
             download_l2_data=False, sen2cor_path=sen2cor_path, stacked_dir=None, resolution=10, cloud_cover='20',
                   bands = ("B02", "B03", "B04", "B08"))
    ras.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge_20m_dir, l1_dir=l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=("B05","B06","B07","B8A","B11","B12"), out_resolution=20)
def do_preproc_only():
    ras.atmospheric_correction(in_directory=l1_dir, out_directory=l2_dir,
                                                    sen2cor_path=sen2cor_path, delete_unprocessed_image=False)
    ras.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge_10m_dir, l1_dir=l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=("B02", "B03", "B04", "B08"), out_resolution=10)

    ras.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge_20m_dir, l1_dir=l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=("B05","B06","B07","B8A","B11","B12"), out_resolution=20)

def sort_into_tile(indir):
    for merged_tif in os.listdir(indir):
        tile_id = fs.get_sen_2_image_tile(merged_tif)
        tile_path = os.path.join(indir,tile_id)
        try:
            os.mkdir(tile_path)
        except FileExistsError:
            pass
        shutil.move(os.path.join(indir,merged_tif), tile_path)

def test_sort_into_tile():
    sort_into_tile(indir= "/media/ubuntu/Data/Ghana/cocoa_big/s2/merge_test")

def do_sort_into_tile():
    sort_into_tile(indir=l1_dir)
    sort_into_tile(indir=l2_dir)


def do_cloud_free_compoiste(indir):
    print('here')

if __name__ == "__main__":
    do_preproc_only()
    do_sort_into_tile()



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