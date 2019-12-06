# 1. data downlaod: https://search.asf.alaska.edu/#/
# 2. data process:
# methods: https://bitbucket.org/sambowers/sen1mosaic/src/master/
# run on command line:
#python /home/ubuntu/Documents/Code/sen1mosaic/cli/preprocess.py -o /media/ubuntu/Data/Ghana/cocoa_big/s1_test/Processed -t /media/ubuntu/Data/Ghana/cocoa_big/s1_test/temp -f -v -p 4 /media/ubuntu/Data/Ghana/cocoa_big/s1_test/GRD
import os

from osgeo import gdal, gdal_array
import numpy as np
import general_functions
def read_tif(intif,type = np.float):
    g = gdal.Open(intif)
    a = gdal_array.DatasetReadAsArray(g).astype(type)
    print("Read " + os.path.basename(intif))
    if a.ndim == 2:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]))
    else:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]) + ', ' + str(a.shape[2]))

    return g, a

def scale_s1_tif(in_tif, out_tif, scale = 10000):
    g, array = read_tif(in_tif)
    array[array<0] = 0
    array_int = array * scale
    if array_int.ndim == 2:
        general_functions.create_tif(filename=out_tif, g=g, Nx=array.shape[0], Ny=array.shape[1], new_array=array_int,
                                     data_type=gdal.GDT_UInt32, noData=0)
    elif array_int.ndim == 3:
        general_functions.create_tif(filename=out_tif,g=g,Nx=array.shape[1],Ny=array.shape[2],new_array=array_int,data_type=gdal.GDT_UInt32,noData=0)
    else:
        print('wrong')

def test_s1_to_int():
    vv_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_all/s1_path1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vv_avg_TC.tif"
    vh_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_all/s1_path1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC.tif"

    vv_tif_int = vv_tif[:-4] + '_int.tif'
    vh_tif_int = vh_tif[:-4] + '_int.tif'

    scale_s1_tif(in_tif = vv_tif, out_tif= vv_tif_int, scale=10000)
    scale_s1_tif(in_tif = vh_tif, out_tif= vh_tif_int, scale=10000)

def clip_rst(in_tif, outline_shp, out_tif):
    os.system('gdalwarp -cutline ' + outline_shp + ' -crop_to_cutline ' + '-overwrite ' + in_tif + ' ' + out_tif)

def test_s1_cut_to_shp():
    vv_tif_int = "/media/ubuntu/Data/Ghana/cocoa_big/s1_all/s1_path1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vv_avg_TC_int.tif"
    vh_tif_int = "/media/ubuntu/Data/Ghana/cocoa_big/s1_all/s1_path1_multi_temporal/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC_int.tif"

    s1_vv_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s1/s1_vv_20180402_testsite_10m.tif"
    s1_vh_testsite_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s1/s1_vh_20180402_testsite_10m.tif"

    shp = "/media/ubuntu/storage/Ghana/shp/geojson/cocoa_upscale_testsite.shp"

    clip_rst(in_tif=vv_tif_int, outline_shp= shp,out_tif= s1_vv_testsite_tif)
    clip_rst(in_tif=vh_tif_int, outline_shp= shp, out_tif= s1_vh_testsite_tif)


def do_stack(vv_tif, vh_tif):
    stack_out = vh_tif[:-4] +'_vv.tif'

    os.system('gdal_merge.py -separate -ot UInt32 -o ' + stack_out + ' ' + vh_tif + ' ' + vv_tif)


def do_int_stack_s1(vv_tif, vh_tif):
    scale_s1_tif(vv_tif,out_tif=vv_tif[:-4]+'_int.tif')
    scale_s1_tif(vh_tif, out_tif=vh_tif[:-4] + '_int.tif')
    stack_int_out = vv_tif[:-4] + 'vh_int.tif'
    os.system('gdal_merge.py -separate -ot Int32 -o ' + stack_int_out  + ' ' +vv_tif[:-4]+'_int.tif' + ' '
               + vh_tif[:-4] + '_int.tif')
def do_mosaic(s1_a,s1_b, mosaic_out):

    os.system('gdal_merge.py -ot Int32 -n 0 -a_nodata 0 -o ' + mosaic_out  + ' ' + s1_a + ' ' + s1_b)


if __name__ == "__main__":
  test_s1_to_int()
  test_s1_cut_to_shp()
  # in_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_path2/S1B_IW_GRDH_1SDV_20180420T182516_20180420T182550_010571_01346E_13BD_processed_cal_mtl_1t_TC.tif"
  # scale_s1_tif(in_tif=  in_tif,
  #              out_tif= in_tif[:-4] +'_int.tif')

  # in_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1_path2/S1B_IW_GRDH_1SDV_20180420T182516_20180420T182550_010571_01346E_13BD_processed_cal_mtl_1t_TC.tif"
  # scale_s1_tif(in_tif=  in_tif,
  #              out_tif= in_tif[:-4] +'_int.tif')

  #do_int_stack_s1(vv_tif="/media/ubuntu/Data/Ghana/cocoa_big/north_region/s1/path1/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vv_avg_TC.tif",
  #                vh_tif= "/media/ubuntu/Data/Ghana/cocoa_big/north_region/s1/path1/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC.tif")
#  do_stack(vv_tif="/media/ubuntu/Data/Ghana/cocoa_big/s1_path1/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vv_avg_TC_int.tif",
#            vh_tif="/media/ubuntu/Data/Ghana/cocoa_big/s1_path1/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC_int.tif")
  #    mosaic_out = os.path.join(os.path.dirname(s1_a),'s1_mosaic.tif')
  # do_mosaic(
  #     s1_a="/media/ubuntu/Data/Ghana/cocoa_big/north_region/s1/path1/S1A_IW_GRDH_1SDV_20180402T182625_20180402T182650_021292_024A1D_9AEF_processed_cal_Stack_Spk_vh_avg_TC_int_vv.tif",
  #     s1_b="/media/ubuntu/Data/Ghana/cocoa_big/north_region/s1/path2/S1B_IW_GRDH_1SDV_20180420T182516_20180420T182550_010571_01346E_13BD_processed_cal_mtl_1t_TC_int.tif")

#  s1 = "/media/ubuntu/Data/Ghana/cocoa_big/s1/s1_path4/S1A_IW_GRDH_1SDV_20180421T181818_20180421T181843_021569_0252C2_A642_processed_cal_mtl_2t_Spk_TC.tif"
#  s1_int = s1[:-4] + "_int.tif"
  # scale_s1_tif(in_tif=s1,out_tif=s1_int)
#
#   s1_path = "/media/ubuntu/Data/Ghana/cocoa_big/s1/"
#   all_s1_list = s2_functions.search_files_fulldir(input_path=s1_path, search_type='end',search_key='_int.tif')
#   outline_shp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
#
#   print(all_s1_list)
#
#   for s1 in all_s1_list:
#       s1_clip = s1[:-4]+'_clip.tif'
#       os.system('gdalwarp -cutline ' + outline_shp + ' -overwrite -srcnodata 0 -dstnodata 0 ' + s1 + ' ' + s1_clip)
#
# #  do_mosaic(s1_a=s1[:-4]+"_int.tif", s1_b="/media/ubuntu/Data/Ghana/north_region/s1/s1_mosaic.tif", mosaic_out=mosic_out )
