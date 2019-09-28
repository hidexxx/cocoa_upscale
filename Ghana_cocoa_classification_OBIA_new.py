import os
import pdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import site
import glob2
from osgeo import ogr,osr,gdal, gdal_array
import sys
import PYEO_model
from sklearn import svm
lib_path = "/home/ubuntu/Documents/Code/pyeo/"
sys.path.append(lib_path)
#from pyeo.apps.model_creation.download_and_preproc_area import main as dl_and_preproc
import s2_functions
import pandas as pd

#import pyeo.queries_and_downloads
#import pyeo.raster_manipulation
#import pyeo.filesystem_utilities

import pdb
from skimage import segmentation




def scale_array_to_255(in_array):
    arr = in_array
    new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype(np.uint8)
    return new_arr

def scale_to_255(intif, outtif):
    g, array = read_tif(intif=intif)
    scaled_array = np.zeros(array.shape)
    for n in range(array.shape[0]):
        arr = array[n,:,:]
        arr[arr<0] = 0
        new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype(np.uint8)
        scaled_array[n,:,:] = new_arr
    s2_functions.create_tif(filename=outtif,g=g,Nx=array.shape[1], Ny= array.shape[2],new_array= scaled_array,data_type=gdal.GDT_Int16,noData=0)
    array = None
    scaled_array = None


#########################################################################
# data pre-processing.. including preprocessing s1, s2 and segmentation
########################################################################
shp = "/media/ubuntu/storage/Ghana/shp/geojson/cocoa_upscale_testsite.shp"


# 1. s1
# 1.1 download and preprocess : output: hv and vv .tif (2 bands )
# methods: https://bitbucket.org/sambowers/sen1mosaic/src/master/
# This is not working due to the change of how archive data is stored in ESA hub.
# ASF is used for downlaod instead. https://search.asf.alaska.edu/#/

# preprocess:
#python preprocess.py -o /media/ubuntu/Data/Ghana/cocoa_big/s1/Processed -t /media/ubuntu/Data/Ghana/cocoa_big/s1/temp -p 4 /media/ubuntu/Data/Ghana/cocoa_big/s1/GRD

#python /home/ubuntu/Documents/Code/sen1mosaic/cli/preprocess.py -o /media/ubuntu/Data/Ghana/cocoa_big/s1_test/Processed -t /media/ubuntu/Data/Ghana/cocoa_big/s1_test/temp -f -v -p 4 /media/ubuntu/Data/Ghana/cocoa_big/s1_test/GRD

# vv_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vv_avg_TC.tif"
# vh_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC.tif"
#
# g_vv, a_vv = read_tif(vv_tif)
# a_vv_int = a_vv* 100000
#
# vv_tif_int = vv_tif[:-4]+'_int.tif'
# #s2_functions.create_tif(filename=vv_tif_int,g=g_vv,Nx=a_vv.shape[0],Ny=a_vv.shape[1],new_array=a_vv_int,data_type=gdal.GDT_Int16,noData=0)
#
s1_vv_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s1/s1_vv_20180402_testsite_10m.tif"
# #os.system('gdalwarp -cutline '+  shp +' -crop_to_cutline '+ '-overwrite ' +vv_tif_int + ' '+ s1_vv_testsite_tif)
#
#
# g_vh, a_vh = read_tif(vh_tif)
# a_vh_int = a_vh* 100000
#
# vh_tif_int = vh_tif[:-4]+'_int.tif'
# #s2_functions.create_tif(filename=vh_tif_int,g=g_vh,Nx=a_vh.shape[0],Ny=a_vh.shape[1],new_array=a_vh_int,data_type=gdal.GDT_Int16,noData=0)
#
s1_vh_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s1/s1_vh_20180402_testsite_10m.tif"
# #os.system('gdalwarp -cutline '+  shp +' -crop_to_cutline '+ '-overwrite ' +vh_tif_int + ' '+ s1_vh_testsite_tif)
#
# # 2. s2
# # 2.1 download and preprocess :output 20m merge. tif (10 bands)
# sen2cor_path = "/home/ubuntu/Downloads/Sen2Cor-02.08.00-Linux64/bin/L2A_Process"
# pyeo_path = "/home/ubuntu/Documents/Code/pyeo/pyeo/apps/model_creation/download_and_preproc_area.py"
# aoi_path = "/media/ubuntu/storage/Ghana/shp/geojson/cocoa_big.geojson"
# l1_dir = "/media/ubuntu/storage/Ghana/s2/cocoa_big/L1"
# l2_dir = "/media/ubuntu/storage/Ghana/s2/cocoa_big/L2"
# merge_dir = "/media/ubuntu/storage/Ghana/s2/cocoa_big/merge"
# conf = "/media/ubuntu/storage/Ghana/s2/cocoa_big/cocoa_big.ini"
#

# dl_and_preproc(aoi_path=aoi_path, start_date='20180401', end_date='20180501',
#                l1_dir=l1_dir, l2_dir=l2_dir, merge_dir=merge_dir, conf_path=conf,
#          download_l2_data=False, sen2cor_path=sen2cor_path, stacked_dir=None, resolution=20, cloud_cover='20')

# #
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

## generate objects:

#https://www.mdpi.com/2072-4292/11/6/658/htm

intif_10m = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite.tif"
#g_10m, a_10m = read_tif(intif_10m)
#a_10m_trans = np.transpose(a_10m, (1,2,0))

intif_20m = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_20m_resample.tif"
#g_20m, a_20m = read_tif(intif_20m)
#a_20m_trans = np.transpose(a_20m, (1,2,0))

# a_3band = np.zeros((a_10m_trans.shape[0],a_10m_trans.shape[1],3), dtype= int)
# #
# # # version 3: NIR, SWIR, and red
# a_3band[:,:,0] = a_10m_trans[:,:,3]
# a_3band[:,:,1] = a_20m_trans[:,:,7]
# a_3band[:,:,2] = a_10m_trans[:,:,2]
#
# a_3band_trans = np.transpose(a_3band, (2,0,1))
# a_3band_out = intif_10m[:-4] + '_NIR_SWIR_red.tif'
# s2_functions.create_tif(filename=a_3band_out, g = g_20m, Nx= a_20m.shape[1], Ny= a_20m.shape[2],new_array=a_3band_trans, noData= 0,data_type=gdal.GDT_Int16)

#objects = segmentation.quickshift(image=a_3band,ratio=0.75,kernel_size=10,max_dist=5)
#
#objects = segmentation.quickshift(image=a_3band,ratio=0.75,kernel_size=20,max_dist=20)
#objects_path = intif_10m[:-4] +'_obj2.pkl'
#PYEO_model.save_model(objects,objects_path)

#objects = PYEO_model.load_model(objects_path)

#print(objects.shape)
#outtif = intif_10m[:-4] +'_obj2.tif'

#s2_functions.create_tif(filename=outtif,g = g_10m, Nx= a_10m_trans.shape[0], Ny= a_10m_trans.shape[1], new_array=objects, data_type=gdal.GDT_Int32)
#
# # version1:
# a_3band = a_trans[:,:,0:3]
#
# #version 2: PCA


#
# # 2.2 cacluating vegetation Index : VI.tif (6 bands)
#
#veg_array = cal_vegIndex(s0_10m=a_10m, s0_20m=a_20m)
#print(veg_array.shape)

#veg_array_trans = np.transpose(veg_array, (2,0,1))
#veg_array_out = intif_10m[:-4] + '_vegIndex.tif'

#s2_functions.create_tif(filename=veg_array_out, g = g_20m, Nx= a_20m.shape[1], Ny= a_20m.shape[2],new_array=veg_array_trans, noData= 0,data_type=gdal.GDT_Int32)

# 2.3 segementation  brightness.tif (1 band)

#3.1 all layer normalised to 255



#3.2 train SVM using parameters:
# s2_veg_s1_bands = np.zeros((18, a_20m.shape[1], a_20m.shape[2])).astype(int)
#
# s2_veg_s1_bands[0:4,:,:] = a_10m
# s2_veg_s1_bands[4:10,:,:] = a_20m[3:,:,:]
# s2_veg_s1_bands[10:16,:,:] = veg_array_trans
# s2_veg_s1_bands[16,:,:] = vh
# s2_veg_s1_bands[17,:,:] = vv
#
# pdb.set_trace()
#veg_index_tif = veg_array_out
#s2_10m_tif = intif_10m
#vh_int_tif = s1_vh_testsite_tif
#vv_int_tif = s1_vv_testsite_tif

#s2simp_veg_s1_bands_out = intif_10m[:-4] + '_vegIndex_s1.tif'
# os.system('gdal_merge.py -separate -ot Int16 -o ' + s2simp_veg_s1_bands_out + ' '
#           + veg_index_tif + ' ' + s2_10m_tif + ' ' + vh_int_tif + ' ' + vv_int_tif)
s2simp_veg_s1_bands = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_vegIndex_s1.tif"

#scale to 255
s2simp_veg_s1_bands_255 = s2simp_veg_s1_bands[:-4] +'_255.tif'
training_image_255 = scale_to_255(intif=s2simp_veg_s1_bands,outtif= s2simp_veg_s1_bands_255)


#4. classify image
