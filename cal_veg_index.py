import numpy as np
from osgeo import gdal

import general_functions

import s2_functions
def clear_nan(inarr):
    inarr[np.isnan(inarr)] = 0
    return inarr

def clear_inf(inarr):
    inarr[np.isinf(inarr)] = 0
    return inarr

def NDVI(s0, s2res='10m', scale=10000):
    '''
    :param s0:array read from s2 20m 9 bands data .tif, or 10m 4 bands data.tif
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    calculate NDVI with equation: NDVI = (NIR-Red)/(NIR+red); NIR -- band 8/8a, red -- band 4
    :return: a 2d-array of NDVI
    '''
    red = s0[2, :, :]  # band 4
    if s2res == '20m':
        NIR = s0[6, :, :]  # band 8
    elif s2res == '10m':
        NIR = s0[3, :, :]  # band 8
    else:
        print('wrong input, NDVI not calculated')
        NIR = 0
    ndvi = (NIR - red) / (NIR + red)
    ndvi[np.isnan(ndvi)] = 0
    ndvi_int = ndvi * scale
    return ndvi_int

def CI(s0_20m, scale=10000):
    '''
    ci = Rededge3/Rededge1 -1
    :param s0: the 9 band from s2 20m resolution. Rededge is only in 20m resolution
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    Rededge 1, 2, 3 are band 5, 6, 7
    :return: a 2d-array of CI
    '''
    rededge3 = s0_20m[5, :, :]
    rededge1 = s0_20m[3, :, :]
    CI = (rededge3 / rededge1 - 1)
    CI[np.isnan(CI)] = 0
    CI[np.isinf(CI)] = 0
    CI_int = CI *  scale
    return CI_int

def PSRI(s0_10m, s0_20m, scale=10000):
    '''
    PSRI = (red-blue)/Rededge2
    :param s0_20m: the 9 band from s2 20m resolution. s0_10m: the 4 band from s2 10m
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    red: band 4, blue band 2, Rededge2 band6
    :return: a 2d-array of PSRI
    '''
    red = s0_10m[2, :, :]
    blue = s0_10m[0, :, :]
    rededge2 = s0_20m[4, :, :]
    psri = ((red - blue) / rededge2)
    psri[np.isnan(psri)] = 0
    psri[np.isinf(psri)] = 0
    psri_int = psri * scale
    return psri_int

def GNDVI(s0, s2res='10m', scale=10000):
    '''
    :param s0:array read from s2 20m 9 bands data .tif or 10m 4 bands data.tif
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    calculate GNDVI with equation: GNDVI = (NIR-Green)/(NIR+Green); NIR -- band 8/8a, green-- band 3
    :return: a 2d-array of GNDVI
    '''
    green = s0[1, :, :]  # band 3

    if s2res == '20m':
        NIR = s0[6, :, :]  # band 8
    elif s2res == '10m':
        NIR = s0[3, :, :]  # band 8
    else:
        print('wrong input, NDVI not calculated')
        NIR = 0
    gndvi = (NIR - green) / (NIR + green)
    gndvi[np.isnan(gndvi)] = 0
    gndvi[np.isinf(gndvi)] = 0
    gndvi_int = gndvi * scale
    return gndvi_int

def S2REP(s0_10m, s0_20m):
    '''

    :return:
    '''
    NIR = s0_10m[3, :, :]
    red = s0_10m[2, :, :]
    rededge1 = s0_20m[3, :, :]
    rededge2 = s0_20m[4, :, :]
    s2rep = 705 + 35 * ((((NIR + red) / 2) - rededge1) / (rededge2 - rededge1))
    s2rep[np.isnan(s2rep)] = 0
    s2rep[np.isinf(s2rep)] = 0
    return s2rep

def IRECI(s0_10m, s0_20m):
    '''
    :param s0_10m:
    :param s0_20m:
    :return:
    '''
    NIR = s0_10m[3, :, :]
    red = s0_10m[2, :, :]
    rededge1 = s0_20m[3, :, :]
    rededge2 = s0_20m[4, :, :]
    ireci = (NIR - red) / (rededge1 / rededge2)
    ireci[np.isnan(ireci)] = 0
    ireci[np.isinf(ireci)] = 0
    return ireci

def print_info(s0):
    print("------")
    print(str(np.max(s0)))
    print(str(np.min(s0)))
    print(str(np.mean(s0)))

def cal_vegIndex(s0_10m, s0_20m):
    ndvi = NDVI(s0=s0_10m)
    ci = CI(s0_20m=s0_20m)
    psri = PSRI(s0_10m=s0_10m, s0_20m=s0_20m)
    gndvi = GNDVI(s0=s0_10m)
    s2rep = S2REP(s0_10m=s0_10m, s0_20m=s0_20m)
    ireci = IRECI(s0_10m=s0_10m, s0_20m=s0_20m)
    print_info(ndvi)
    print_info(ci)
    print_info(psri)
    print_info(gndvi)
    print_info(s2rep)
    print_info(ireci)
    result_array = np.dstack((ndvi, ci, psri, gndvi, s2rep, ireci)).astype(int)
   # result_array = np.dstack((ndvi, ci, gndvi, s2rep, ireci)).astype(int)
    return result_array

def generate_veg_index_tif(tif_10m,tif_20m,out_tif):
    g_10m, arr_10m = general_functions.read_tif(intif=tif_10m, type=np.int32)
    g_20m, arr_20m = general_functions.read_tif(intif=tif_20m, type=np.int32)
    veg_arr = cal_vegIndex(arr_10m,arr_20m)

    veg_array_trans = np.transpose(veg_arr, (2, 0, 1))

    general_functions.create_tif(filename=out_tif, g = g_10m, Nx= arr_20m.shape[1], Ny= arr_20m.shape[2],new_array=veg_array_trans, noData= 0,data_type=gdal.GDT_Int32)

def test_generate_veg_tif():
    tif_10m = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_10m.tif"
    tif_20m = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_20m_resample.tif"
    out_tif = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/veg_withoutPSRI.tif"
    generate_veg_index_tif(tif_10m,tif_20m,out_tif)

def do_generate_veg_tif():
    merge_10m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/north_region/s2/merge_10m"
    merge_20m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/north_region/s2/merge_20m"
    tif_10m_list = s2_functions.search_files_fulldir(input_path=merge_10m_dir,search_type='end',search_key='_clip.tif')
    tif_20m_list = s2_functions.search_files_fulldir(input_path=merge_20m_dir,search_type='end',search_key='10m_clip.tif')


    for n in range(len(tif_10m_list)):
        tif_10m = tif_10m_list[n]
        tif_20m = tif_20m_list[n]
        print(tif_10m)
        print(tif_20m)
        out_tif = tif_10m[:-4] + "_vegIndex6.tif"
        generate_veg_index_tif(tif_10m=tif_10m,tif_20m=tif_20m,out_tif=out_tif)


if __name__ == "__main__":
    print("here")
    #test_generate_veg_tif()
    #do_generate_veg_tif()