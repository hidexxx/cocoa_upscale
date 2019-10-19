from osgeo import gdal
import s2_functions
import general_functions
import numpy as np
import os
import pdb
import scipy.ndimage.measurements
import pyeo.classification as cls


def cal_seg_mean(in_value_ras, seg_ras, out_value_ras, output_filtered_value_ras = True):
    g_seg, seg_arr = general_functions.read_tif(seg_ras, type=np.int64)
    g_value, value_arr = general_functions.read_tif(in_value_ras, type=np.int64)

    out_array = np.zeros(value_arr.shape)

    unique_segvals = np.unique(seg_arr)

    band_number = value_arr.shape[0]
    for n in range(band_number):
        print('working on band: ' + str(n + 1))
        band_val = value_arr[n, :, :]
        mean_arr = scipy.ndimage.measurements.mean(band_val, labels=seg_arr, index=unique_segvals)

        # new_arr = np.zeros(band_val.shape)
        lookup_array = np.zeros(unique_segvals.max() + 1)
        lookup_array[unique_segvals] = mean_arr
        new_arr = lookup_array[seg_arr]
        out_array[n, :, :] = new_arr

    if output_filtered_value_ras:
        general_functions.create_tif(filename=out_value_ras, g=g_seg, Nx=seg_arr.shape[0], Ny=seg_arr.shape[1],
                                     new_array=out_array, data_type=gdal.GDT_UInt32, noData=0)
    else:
        print('Caculating brightness layer...')
        brightness = np.mean(out_array,axis= 0)
        print(brightness.shape)
        general_functions.create_tif(filename=out_value_ras, g=g_seg, Nx=seg_arr.shape[0], Ny=seg_arr.shape[1],
                                     new_array=brightness, data_type=gdal.GDT_UInt32, noData=0)

def test_generate_brightness_rst():
    in_value_ras= "/media/ubuntu/Data/Ghana/cocoa_upscale_test/segmentation/s2_20180219_testsite_NIR_SWIR_red.tif"
    seg_ras="/media/ubuntu/Data/Ghana/cocoa_upscale_test/segmentation/segmentation_testsite_parameter3.tif"
    out_value_ras = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/segmentation/s2_20180219_testsite_brightness.tif"
    output_filtered_value_ras = False

    general_functions.clip_rst_by_rst(in_tif=seg_ras,ref_tif=in_value_ras,out_tif= seg_ras[:-4]+"_reshape.tif")
    cal_seg_mean(in_value_ras, seg_ras[:-4]+"_reshape.tif", out_value_ras, output_filtered_value_ras=output_filtered_value_ras)


def ave_value_rst_by_seg(in_value_rst_dir, in_seg_rst_dir,out_value_rst_dir):
    for image in os.listdir(in_value_rst_dir):
        if image.endswith(".tif"):
            seg_image = os.path.join(in_seg_rst_dir, image)
            out_value_image = os.path.join(out_value_rst_dir,image)

            cal_seg_mean(in_value_ras=os.path.join(in_value_rst_dir,image),seg_ras=seg_image,out_value_ras=out_value_image)


def test_cal_seg_mean():
    ave_value_rst_by_seg(
        in_value_rst_dir="/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_temp",
        in_seg_rst_dir= "/media/ubuntu/Data/Ghana/north_region/s2/segmentation_temp",
        out_value_rst_dir="/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_obj")

def test_classify():
    image_dir = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_obj"
    model_path = "/media/ubuntu/Data/Ghana/cocoa_big/model/PCA_to_ET_with_George_data_all_data_cleaned.pkl"
    out_dir = "/media/ubuntu/Data/Ghana/north_region/s2/output"
    for image in os.listdir(image_dir):
        if image.endswith(".tif"):
            classify_out = os.path.join(out_dir,image + '_obj_classified.tif')
            cls.classify_image(image_path=os.path.join(image_dir,image),model_path=model_path, class_out_path=classify_out)

if __name__ == "__main__":
    #test_cal_seg_mean()
    #test_classify()
    test_generate_brightness_rst()

#
#
# g_seg, seg_arr = general_functions.read_tif(clip_seg_tif,type=np.int64)
#
# g_value, value_arr = general_functions.read_tif(value_tif, type=np.int64)
#
# #seg_arr_3d = seg_arr[np.newaxis,:,:]
#
# out_array = np.zeros(value_arr.shape)
#
# unique_segvals = np.unique(seg_arr)
#
# band_number = value_arr.shape[0]
# for n in range(band_number):
#     print('working on band: ' + str(n+1))
#     band_val = value_arr[n,:,:]
#
#     mean_arr = scipy.ndimage.measurements.mean(band_val,labels = seg_arr,index = unique_segvals)
#
#     new_arr = np.zeros(band_val.shape)
#
#     lookup_array=np.zeros(unique_segvals.max()+1)
#     lookup_array[unique_segvals]=mean_arr
#     new_arr=lookup_array[seg_arr]
#
#     out_array[n,:,:] = new_arr
#
# value_obj_tif = value_tif[:-4] +'_obj.tif'

#general_functions.create_tif(filename=value_obj_tif,g=g_seg,Nx=seg_arr.shape[0],Ny=seg_arr.shape[1],new_array=out_array, data_type=gdal.GDT_UInt32,noData=0)


#
#






