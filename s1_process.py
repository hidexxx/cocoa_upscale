# 1. data downlaod: https://search.asf.alaska.edu/#/
# 2. data process:
# methods: https://bitbucket.org/sambowers/sen1mosaic/src/master/
# run on command line:
#python /home/ubuntu/Documents/Code/sen1mosaic/cli/preprocess.py -o /media/ubuntu/Data/Ghana/cocoa_big/s1_test/Processed -t /media/ubuntu/Data/Ghana/cocoa_big/s1_test/temp -f -v -p 4 /media/ubuntu/Data/Ghana/cocoa_big/s1_test/GRD
import os

from osgeo import gdal, gdal_array
import numpy as np
import s2_functions

def read_tif(intif,type = np.float):
    g = gdal.Open(intif)
    a = gdal_array.DatasetReadAsArray(g).astype(type)
    print("Read " + os.path.basename(intif))
    if a.ndim == 2:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]))
    else:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]) + ', ' + str(a.shape[2]))

    return g, a

def scale_s1_tif(in_tif, out_tif, scale = 100000):
    g, array = read_tif(in_tif)
    array_int = array * scale
    s2_functions.create_tif(filename=out_tif,g=g,Nx=array.shape[0],Ny=array.shape[1],new_array=array_int,data_type=gdal.GDT_Int16,noData=0)

def test_s1_to_int():
    vv_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vv_avg_TC.tif"
    vh_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC.tif"

    vv_tif_int = vv_tif[:-4] + '_int.tif'
    vh_tif_int = vv_tif[:-4] + '_int.tif'

    scale_s1_tif(in_tif = vv_tif, out_tif= vv_tif_int, scale=100000)
    scale_s1_tif(in_tif = vh_tif, out_tif= vh_tif_int, scale=100000)

def clip_rst(in_tif, outline_shp, out_tif):
    os.system('gdalwarp -cutline ' + outline_shp + ' -crop_to_cutline ' + '-overwrite ' + in_tif + ' ' + out_tif)

def test_s1_cut_to_shp():
    vv_tif_int = "/media/ubuntu/Data/Ghana/cocoa_big/s1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vv_avg_TC_int.tif"
    vh_tif_int = "/media/ubuntu/Data/Ghana/cocoa_big/s1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC.tif"

    s1_vv_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s1/s1_vv_20180402_testsite_10m.tif"
    s1_vh_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s1/s1_vh_20180402_testsite_10m.tif"

    shp = "/media/ubuntu/storage/Ghana/shp/geojson/cocoa_upscale_testsite.shp"

    clip_rst(in_tif=vv_tif_int, outline_shp= shp,out_tif= s1_vv_testsite_tif)
    clip_rst(in_tif=vh_tif_int, outline_shp= shp, out_tif= s1_vh_testsite_tif)

if __name__ == "__main__":
    test_s1_to_int()
    test_s1_cut_to_shp()
