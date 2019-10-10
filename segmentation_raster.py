from osgeo import gdal
import s2_functions
import general_functions
import numpy as np
import os
import pdb
import scipy.ndimage.measurements
seg_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/segmentation/segmentation_testsite_parameter3.tif"
#value_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/segmentation/s2_20180219_testsite_NIR_SWIR_red.tif"
value_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/s2_20180219_testsite_vegIndex_s1.tif"


shp = "/media/ubuntu/storage/Ghana/shp/geojson/cocoa_upscale_testsite.shp"

clip_seg_tif = seg_tif[:-4]+'_clip.tif'
#os.system('gdalwarp -cutline ' + shp + ' -crop_to_cutline -tr 10 10 ' + '-overwrite ' + seg_tif + ' ' + clip_seg_tif)

g_seg, seg_arr = general_functions.read_tif(clip_seg_tif,type=np.int64)

g_value, value_arr = general_functions.read_tif(value_tif, type=np.int64)

#seg_arr_3d = seg_arr[np.newaxis,:,:]

out_array = np.zeros(value_arr.shape)

unique_segvals = np.unique(seg_arr)

band_number = value_arr.shape[0]
for n in range(band_number):
    print('working on band: ' + str(n+1))
    band_val = value_arr[n,:,:]

    mean_arr = scipy.ndimage.measurements.mean(band_val,labels = seg_arr,index = unique_segvals)

    new_arr = np.zeros(band_val.shape)

    lookup_array=np.zeros(unique_segvals.max()+1)
    lookup_array[unique_segvals]=mean_arr
    new_arr=lookup_array[seg_arr]

    out_array[n,:,:] = new_arr

value_obj_tif = value_tif[:-4] +'_obj.tif'

#general_functions.create_tif(filename=value_obj_tif,g=g_seg,Nx=seg_arr.shape[0],Ny=seg_arr.shape[1],new_array=out_array, data_type=gdal.GDT_UInt32,noData=0)

model_path = "/home/ubuntu/Desktop/cocoa_upscale_northregion/most_consistent_model.pkl"
value_obj_classified_out = value_tif[:-4] +'_obj_classified.tif'


import pyeo.classification as cls
cls.classify_image(image_path=value_obj_tif,model_path=model_path,class_out_path=value_obj_classified_out)
