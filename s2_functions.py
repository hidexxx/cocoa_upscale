import os
import shutil
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
import numpy as np
import site
import glob,glob2
#lib_path = "/media/ubuntu/Seagate Expansion Drive/backup Dec/remoteDriveS1251670/code/bin"
#site.addsitedir(lib_path)
#import geoTiff
import zipfile
#lib_path = "/media/ubuntu/Seagate Expansion Drive/backup Dec/remoteDriveS1251670/code/bin/classification"
#site.addsitedir(lib_path)

#from random_forest_clean import random_forest
#from validate_classes_clean import validate_classes
from sklearn.externals import joblib
from tqdm import tqdm
import pdb
lib_path = "/home/ubuntu/Documents/Code/pyeo/pyeo"
site.addsitedir(lib_path)
#import core
import pandas as pd
from scipy import ndimage


def plotArray(inputarray):
    plt.figure(1)
    plt.imshow(inputarray)
    plt.show()

def unzip_file(zip_filename,create_folder = False):
    zip_handler = zipfile.ZipFile(zip_filename, "r")
    if create_folder == True:
        dir_name = os.path.splitext(zip_filename)[0]
        os.mkdir(dir_name)
        zip_handler.extractall(dir_name)
    else:
        zip_handler.extractall(os.path.dirname(zip_filename))

def search_files(input_path,search_key,search_type):
    search_types = ['start', 'end', 'contain']
    if search_type not in search_types:
        raise ValueError("Invalid search type. Expected one of: %s" % search_types)
    elif search_type == 'start':
        configfiles = [os.path.join(input_path, f)
                       for dirpath, dirnames, files in os.walk(input_path)
                       for f in files if f.startswith(search_key)]
    elif search_type == 'end':#
        configfiles = [os.path.join(input_path, f)
                       for dirpath, dirnames, files in os.walk(input_path)
                       for f in files if f.endswith(search_key)]
    else:
        configfiles = [os.path.join(input_path, f)
                       for dirpath, dirnames, files in os.walk(input_path)
                       for f in files if search_key in f]
    return configfiles

def search_files_fulldir(input_path,search_key,search_type):
    search_types = ['start', 'end', 'contain']
    if search_type not in search_types:
        raise ValueError("Invalid search type. Expected one of: %s" % search_types)
    elif search_type == 'start':
        configfiles = [os.path.join(dirpath,f)
                       for dirpath, dirnames, files in os.walk(input_path)
                       for f in files if f.startswith(search_key)]
    elif search_type == 'end':#
        configfiles = [os.path.join(dirpath,f)
                       for dirpath, dirnames, files in os.walk(input_path)
                       for f in files if f.endswith(search_key)]
    else:
        configfiles = [os.path.join(dirpath,f)
                       for dirpath, dirnames, files in os.walk(input_path)
                       for f in files if search_key in f]
    return configfiles
def search_dir(input_path):
    dirs = [f.path for f in os.scandir(input_path) if f.is_dir()]
    return dirs

def copy_tiffs(old_dir, new_dir, suffix):
    tiff_list = [os.path.join(dirpath, tifs)
        for dirpath, dirnames, files in os.walk(old_dir)
         for tifs in files if tifs.endswith(suffix)]
    for i in tiff_list:
        print('copying....' + os.path.split(i)[1])
        shutil.copy2(i,os.path.join(new_dir,os.path.split(i)[1]))

def move_tiffs(old_dir, new_dir, suffix):
    tiff_list = search_files(input_path=old_dir,search_key=suffix,search_type='end')
    for i in tiff_list:
        print('moving....' + os.path.split(i)[1])
        shutil.move(i,os.path.join(new_dir))

def delete_file(search_fldr, search_type, suffix):
    if search_type == 'in':
        for subdir, dirs, files in os.walk(search_fldr):
            for item in files:
                if suffix in str(item):
                    del_file = os.path.join(subdir, item)
                    print
                    'Deleting file.......' + del_file
                    os.remove(del_file)

    elif search_type == 'end':
        for subdir, dirs, files in os.walk(search_fldr):
            for item in files:
                #            if item.endswith(suffix):
                if suffix in str(item):
                    del_file = os.path.join(subdir, item)
                    print
                    'Deleting file.......' + del_file
                    os.remove(del_file)

    elif search_type == 'start':
        for subdir, dirs, files in os.walk(search_fldr):
            for item in files:
                if item.startswith(suffix):
                    del_file = os.path.join(subdir, item)
                    print
                    'Deleting file.......' + del_file
                    os.remove(del_file)


def create_tif(filename, g, Nx,Ny, new_array, data_type= gdal.GDT_Float32, noData = ''):
    """Creates a new GeoTiff from an array.  Array parameters (i.e. pixel height/width, raster dimensions, projection info) are taken from a pre-existing geotiff.  Geotiff output is GDT_Float32

    Args:
    filename (str): absolute file path and name of the new Geotiff to be created
    g (gdal supported dataset object): This is returned from gdal.Open(filename) where filename is the name of a gdal supported dataset
    Nx and Ny are the x size and y size which defines the new array's shape
    data_type (string): a string of teh data type for the output, e.g. gdal.GDT_Float32, gdal.GDT_Byte, gdal.GDT_Int16. Default is GDT_Float32
    """
    # CR: this needs to be able to create different number types e.g. int
    (X, deltaX, rotation, Y, rotation, deltaY) = g.GetGeoTransform()
    srs_wkt = g.GetProjection()
    driver = gdal.GetDriverByName("GTiff")
    if new_array.ndim == 2:
        Dataset = driver.Create(filename, Ny, Nx, 1, data_type)
    else:
        Dataset = driver.Create(filename, Ny, Nx, new_array.shape[0], data_type) #GDT_Byte for int # CR instead of data_type this used to read gdal.GDT_Float32
    Dataset.SetGeoTransform((X, deltaX, rotation, Y, rotation, deltaY))
    Dataset.SetProjection(srs_wkt)

    if noData == '':
        noData = Dataset.GetRasterBand(1).GetNoDataValue()
#        print 'noData = ' + str(noData)
    else:
        #Dataset.GetRasterBand(1).SetNoDataValue(noData)
         new_array[np.isnan(new_array)] = noData
#        print 'changed no data to: ' +str(noData)
    print ('Create tiff: writing: '+ filename)
    if new_array.ndim == 2:
        Dataset.GetRasterBand(1).WriteArray(new_array)
    else:
        for i, image in enumerate(new_array):
            Dataset.GetRasterBand(i+1).WriteArray(image)
            Dataset.GetRasterBand(i+1).SetNoDataValue(noData)


def findID(dataname):
    '''
    intput a file name, return is the dataID - the acquistion date
    '''
    if '/' in dataname:
        dataname = os.path.split(dataname)[1]
    else:
        dataname = dataname
    if  dataname.startswith('L2A'): #This is level 2A data
        pathID = dataname[4:10]; timeID = dataname[11:26]
    elif dataname.startswith('S2A'): # This is level 1C data
        pathID = dataname[49:55]; timeID = dataname[25:40]
    else:
        print ('Do not know the data format?')
        pathID = ' '; timeID = ' '
    return pathID,timeID

def findID_SAFE(filedir,info_type):
    '''
    :param filedir:
    :param info_type: eith 'path' or 'orbit'
    :return: either path or orbit of the .SAFE data
    '''
    if '/' in filedir:
        dataname = os.path.split(filedir)[1]
    else:
        dataname = filedir
    info_out = {}
    if np.logical_or(dataname.startswith('S2A_MSIL2A_'),dataname.startswith('S2B_MSIL2A_')): #This is the code version
        allinfo = dataname.split('_')
        path = allinfo[5];sensingdate = allinfo[2];orbit= allinfo[4]
   #     info_out[path] = sensingdate
    elif dataname.startswith('S2A_USER_PRD_MSIL2A_PDMC_'):
        imgdir = os.path.join(filedir,'GRANULE')
        for img in imgdir:
            imgname = os.path.split(img)[1]
            allinfo = imgname.split('_')
            path = allinfo[8];sensingdate = allinfo[6];orbit=allinfo[7]
            info_out[path] = sensingdate
    pdb.set_trace()
    if info_type == 'path':
        info_out[path]=sensingdate
    elif info_type == 'orbit':
        info_out[orbit]=sensingdate
    return info_out

def SortByPath(filedir):
    if '/' in filedir:
        dataname = os.path.split(filedir)[1]
    else:
        dataname = filedir

    if np.logical_or(os.path.split(dataname)[1].startswith('S2A_MSIL2A_'),os.path.split(dataname)[1].startswith('S2B_MSIL2A_')):
        outinfo = findID_SAFE(dataname,info_type='path')
        print(outinfo)
      #  date = list(outinfo.values())[0]
        path = list(outinfo.keys())[0]
        newdir = os.path.join(os.path.dirname(filedir)+'/'+path, os.path.basename(dataname))
        os.makedirs(newdir)
        os.rename(dataname,newdir)
    else:
        print('skip...'+ dataname)
        #continue

def SortByOrbit(filedir):
    if '/' in filedir:
        dataname = os.path.split(filedir)[1]
    else:
        dataname = filedir

    if np.logical_or(os.path.split(dataname)[1].startswith('S2A_MSIL2A_'),os.path.split(dataname)[1].startswith('S2B_MSIL2A_')):
        outinfo = findID_SAFE(dataname,info_type='orbit')
        print(outinfo)
      #  date = list(outinfo.values())[0]
        orbit = list(outinfo.keys())[0]
        newdir = os.path.join(os.path.dirname(filedir)+'/'+orbit, os.path.basename(dataname))
        os.makedirs(newdir)
        os.rename(filedir,newdir)
    else:
        print('skip...'+ dataname)
        #continue

def SortByPathOrbit(filedir):
    if '/' in filedir:
        dataname = os.path.split(filedir)[1]
    else:
        dataname = filedir

    if np.logical_or(os.path.split(dataname)[1].startswith('S2A_MSIL2A_'),os.path.split(dataname)[1].startswith('S2B_MSIL2A_')):
        outinfo = findID_SAFE(dataname,info_type='path')
      #  date = list(outinfo.values())[0]
        path = list(outinfo.keys())[0]

        outinfo = findID_SAFE(dataname, info_type='orbit')
        #  date = list(outinfo.values())[0]
        orbit = list(outinfo.keys())[0]
        print('sorting into...' +path + ' and ' +orbit)
        newdir = os.path.join(os.path.dirname(filedir)+'/'+path +'/' + orbit, os.path.basename(dataname))
        os.makedirs(newdir)
        os.rename(filedir,newdir)
    else:
        print('skip...'+ dataname)


def testCompletion(L2A_file, resolution=0):
    """
    Test for successful completion of sen2cor processing.

    Args:
        L1C_file: Path to level 1C granule file (e.g. /PATH/TO/*_L1C_*.SAFE/GRANULE/*)
    Returns:
        A boolean describing whether processing completed sucessfully.
    """
    failure = False
    # Test all expected 10 m files are present
    if resolution == 0 or resolution == 10:
        for band in ['B02', 'B03', 'B04', 'B08', 'AOT', 'TCI', 'WVP']:
 #           if not len(glob.glob('%s/IMG_DATA/R10m/*_%s_10m.jp2' % (L2A_file, band))) == 1:
            if not len(glob.glob('%s/GRANULE/*/IMG_DATA/R10m/*_%s_10m.jp2' % (L2A_file, band))) == 1:
                failure = True
    # Test all expected 20 m files are present
    if resolution == 0 or resolution == 20:
        for band in ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'AOT', 'TCI', 'WVP', 'SCL']:
            if not len(glob.glob('%s/GRANULE/*/IMG_DATA/R20m/*_%s_20m.jp2' % (L2A_file, band))) == 1:
                failure = True
    # Test all expected 60 m files are present
    if resolution == 0 or resolution == 60:
        for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'AOT', 'TCI', 'WVP', 'SCL']:
            if not len(glob.glob('%s/GRANULE/*/IMG_DATA/R60m/*_%s_60m.jp2' % (L2A_file, band))) == 1:
                failure = True
    # At present we only report failure/success, can be extended to type of failure
    return failure == False

    ### have written a little stuff by myself, need to come back checking if this is working or not
    # ### check if the folder is empty to make sure that sen2cor works properly
    # print(s2a)
    # error_list_granule = [];error_list_10m =[]
    # if len(os.listdir(oe folder is not empty, check if the 10m folder is empty or not:
    #     error_list_10m.append(s2a)
    # else:
    #     continue
def print_text(intxt):
    f = open(intxt, 'r')
    file_contents = f.read()
    print(file_contents)
    f.close()


def mergeTiff_text(input_dir,outname,search_suffix = '.tif', driver = 'gdal_merge', filetype = 'Int16'):
    '''
    :param input_dir:
    :return:
    '''
    list_file = input_dir + '/to_merge.txt'
    os.system('ls ' + input_dir + '*' + search_suffix + ' > ' + list_file)
    print('buiding a stack containing... ')
    print_text(list_file)

    if driver == 'gdal_merge':
        os.system('gdal_merge.py -separate -ot '+ filetype + ' -o ' + outname + ' --optfile ' + input_dir + '/to_merge.txt')
    elif driver == 'VRT':
        #gdal.BuildVRT(list_files, vrtname)
        vrtname = outname[:-4] + '.vrt'
        os.system('gdalbuildvrt -input_file_list ' + list_file + ' ' + vrtname)
        os.system('gdal_translate -of GTiff -ot ' + filetype + ' ' + vrtname + ' ' + outname)

    #other things to test in merge band..... because the vrt and stuff
#     # merge_files.append(i);merge_files.append(j)
#     # with open(os.path.join(new_dir,'merge_files.txt'), 'w') as file:
#     #     file.writelines(merge_files)
#
#     # gdalbiuldvrt does not suport heterougenous band numbers so can only use gdal.merge
#    # gdal.BuildVRT(list_files,vrtname)
#    # os.system('gdalbuildvrt -input_file_list ' + list_file2+' '+ vrtname)
#    # os.system('gdal_translate -of GTiff ' + vrtname + ' ' + vrtname[:-4]+'.tif')

#     # os.system('gdal_merge.py -separate -o '+ outname + ' --optfile ' + merge_files)
#     os.system('gdal_merge.py -separate -o ' + outname + ' ' + iout +' ' + jout) #gdal_merge works perfectly.. but you can specify with resampling methods it uses.
#     #os.system('gdalwarp -tr 25 25 -r cubic -overwrite ' + i + ' ' + j + ' ' + outname )# can't deal with bands with different length.

def mosaicTiff_text(input_dir,outname,search_suffix = '.tif', filetype = 'Int16'):
    '''
    :param input_dir:
    :return:
    '''
    list_file = input_dir + '/to_mosaic.txt'
    os.system('ls ' + input_dir + '/*' + search_suffix + ' > ' + list_file)
    print('mosaicing ..... ')
    print_text(list_file)
    os.system('gdal_merge.py -ot '+ filetype + ' -o ' + outname + ' --optfile ' + input_dir + '/to_mosaic.txt')

# def findFilesList(inputdir, search_criteria= "tifs.endswith('.zip')"):
#     '''
#     :param inputdir:
#     :param search_criteria: should be logial expression for criteria for the tiffs
#     eg. tifs.endswith
#     :return: a list of files
#     '''
#     filesList = [os.path.join(dirpath, tifs)
#             for dirpath, dirnames, files in os.walk(inputdir)
#             for tifs in files if search_criteria]
#
#     for i in filesList:
#         print(i)
#         print(i)
#     print (str(len(filesList))+ ' files ' + search_criteria + ' found')
#     return filesList


def findMatchingFile(inputfile, searchdir, searchsuffix):
    '''
    :param inputfile: input raster, the mask of which will be found
    :param maskdir: mask dir, where we will search for the mask using the suffix and Id name for the input raster
    :param searchsuffix: suffix of the mask tiff
    :return: a tiff
    '''
    pathID,nameID = findID(inputfile)

    matchtiffs = [os.path.join(dirpath, tifs)
        for dirpath, dirnames, files in os.walk(searchdir)
        for tifs in files if np.logical_and(tifs.endswith(searchsuffix), nameID[0:8] in tifs)]
       #for tifs in files if np.logical_and(tifs.endswith(searchsuffix), nameID in tifs)]
    matchtiff = matchtiffs[0]
    return matchtiff

def ShptoRst(ref_raster,inputshp, outputrst, nodatavalue= 0, field_name = 'id',datatype = 'int16'):
    '''

    :return:
    '''
    (x_min, pixel_width, rotation, y_max, rotation, pixel_height, rows, cols, bands, x_max, y_min) = geoTiff.getallinfo(ref_raster)
    clipd_shp_rst = os.path.join(inputshp, 'rst.tif')
    os.system('gdal_rasterize -init 0 -a_nodata ' + str(nodatavalue) +' -a ' + field_name + ' -ot '+ str(datatype) + ' -l ' + os.path.basename(inputshp[:-4]) + ' -te ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' -tR ' + str(pixel_width) + ' ' + str(-pixel_height) + ' ' + inputshp + ' ' + outputrst)


def cal_vegIndex(s0,s2res = '20m'):
    '''

    :param s0:array read from s2 20m 9 bands data .tif
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    calculate NDVI with equation: NDVI = (NIR-Red)/(NIR+red); NIR -- band 8/8a, red -- band 4
    NDWI: NDWI = (Green- NIR)/(Green+ NIR); green band 3
    :return: a 3d-array containing two layers: NDVI and NDWI layers
    '''
    red = s0[2, :, :] # band 4

    if s2res == '20m':
        NIR = s0[6, :, :]#band 8
    elif s2res == '10m':
        NIR = s0[3, :, :]#band 8
    else:
        print ('wrong input, NDVI not calculated')
        NIR = 0
        #NDVI
    ndvi = (NIR - red) / (NIR + red) * 10000

        # calculate NDWI: NDWI = (Green- NIR)/(Green+ NIR); green band 3
    green = s0[1, :, :] #band 3
    ndwi = (green - NIR) / (green + NIR) * 10000

    index = np.dstack((ndvi, ndwi)).astype(int)
    index[np.isnan(index)]=0
    return index

def cal_vegIndex_array(s0, s2res='10m'):
    red = s0[:,2]
    NIR = s0[:,3]
    ndvi = (NIR - red) / (NIR + red) * 10000
    return ndvi

def cal_vegIndex_diff(s0_before,s0_after):
    ndvi_before = cal_vegIndex_array(s0_before, s2res='10m')

    ndvi_after = cal_vegIndex_array(s0_after, s2res='10m')
    diff = ndvi_before - ndvi_after

    s0_before=None;s0_after=None
    return diff

def clear_grow_cloud_mask(cloudtif,clear_window_size = 3, grow_window_size =5):
    g = gdal.Open(cloudtif)
    s0 = gdal_array.DatasetReadAsArray(g)

    if s0.ndim == 3: # if it's a 3D array
        mask_array = s0[0, :, :].copy()
    else:
        mask_array = s0.copy()

    mask_array[(mask_array == 1) | (mask_array == 4)] = 0 # 1 is water, 4 is good pixel, not touching them
    mask_array[(mask_array == 2) | (mask_array == 3)] = 1  # 2 is cloud and 3 is shadow, need to be cleared # NEED TO BE CAREFUL, this comes later!!

    print('there are :' + str(np.count_nonzero(mask_array == 1)) + ' pixels to be cleared (== 1)')
    print('there are :' + str(np.count_nonzero(mask_array == 0)) + ' pixels not touched (==0)')

    #remove objects smaller than 3*3 pixel
    print('clear isolated pixels that are smaller than ' + str(clear_window_size) + ' * ' + str(clear_window_size))
    mask_array_errosion = ndimage.binary_erosion(mask_array, np.ones((clear_window_size, clear_window_size), dtype=np.int)).astype(int)  # np.binary_erosion
   # mask_array_errosion2 = ndimage.binary_erosion(mask_array_errosion, np.ones((clear_window_size, clear_window_size), dtype=np.int)).astype(int)
    print('After erosion, cloud pixels decreased from:' + str(np.count_nonzero(mask_array == 1)) + ' to : ' + str(np.count_nonzero(mask_array_errosion == 1)))

    print('grow back by ' + str(grow_window_size) + ' * ' + str(grow_window_size))
    mask_growback = ndimage.binary_dilation(mask_array_errosion, np.ones((grow_window_size, grow_window_size), dtype=np.int)).astype(int)

    # get water
    sea_mask = s0.copy() # for s0, 1 is water
    sea_mask[sea_mask!=1] = 0 # assign where there is not water, to 0

    # geoTiff.create('/media/ubuntu/storage/Indonesia/TRAINING_DATA/clouds/INDONESIA_CIDANAU-1_20171206_001_seamask.tif', g=g, old_array=s0, new_array=sea_mask,
    #                data_type=gdal.GDT_Int16)
    mask_growback = mask_growback + sea_mask

    # where mask_grow_back == 2 means it's cloud on the sea, change it to 1.
    mask_growback[mask_growback==2] = 1

    return mask_growback

def run_RF_validate(intif):
    '''
    a little dirty code to run RF on images
    AOI = '/home/ubuntu/Documents/TimeSeries/S2/1516/AOI/5classes_AOI_V2_clean_prj_clip.tif'  # to UTM
    '''
    rfout = intif[:-4]+'_rf_temp.tif'

    AOI = '/home/ubuntu/Documents/TimeSeries/S2/1516/AOI/5classes_AOI_V2_clean_prj_clip.tif'  # to UTM

    random_forest(raster_tif = intif, training_tif = AOI,output_name= rfout,mask_value=9999)

    # #change projection for outtif for validation
    rfout_prj = rfout[:-4]+'BNG.tif'
    os.system('gdalwarp ' + rfout + ' ' + rfout_prj + ' -t_srs "EPSG:27700"')  # EPSG 27700
    shpdir = '/home/ubuntu/Documents/TimeSeries/S2/1516/AOI/validate_big_shp/'
    vali_fldr = '/home/ubuntu/Documents/TimeSeries/S2/1516/validation/'
    validate_classes(inRaster=rfout_prj, shpdir=shpdir, out_fldr=vali_fldr, field_name='GRIDCODE')

def jp2toTif(injp2):
    outtif = injp2[:-4]+'.tif'
    os.system('gdal_translate -of GTiff ' + injp2 +' ' + outtif)

def extractFile (input):
    zipFile = glob.glob(input + "/*.zip")
    filename = zipFile[0].split("\\")
    folderName = zipped_safe_path + "\\" + filename[-1][:-4]
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    else:
        shutil.rmtree(folderName)
        os.makedirs(folderName)
    zip = zipfile.ZipFile(zipFile[0])
    zip.extractall(folderName)
    return folderName


def shptoJashon(inputShp,outputjson):
    '''
    get the code from here:
    https://gist.github.com/frankrowe/6071443
    :param inputShp:
    :param outputjson:
    :return:
    '''
    import shapefile
    # read the shapefile
    reader = shapefile.Reader(inputShp)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]
    buffer = []
    for sr in reader.shapeRecords():
        atr = dict(zip(field_names, sr.record))
        geom = sr.shape.__geo_interface__
        buffer.append(dict(type="Feature", \
                           geometry=geom, properties=atr))

        # write the GeoJSON file
    from json import dumps
    geojson = open(outputjson, "w")
    geojson.write(dumps({"type": "FeatureCollection", \
                         "features": buffer}, indent=2) + "\n")
    geojson.close()

def find_chunk_size(input_tif):
    '''

    :param input_tif:
    :return: number of chunks
    '''
    g = gdal.Open(input_tif)
    s0 = gdal_array.DatasetReadAsArray(g)
    print(s0.shape)
    shu=s0.shape[1]*s0.shape[2]
    result=[a for a in range(1,shu+1) if shu%a==0]
    print(result)

    return result

def add_field_shp(inputshp, field_name, field_type, field_length):
    '''

    :param inputshp:
    :param field_name:
    :param field_type: could be ogr.OFTInteger, ogr.OFTReal, or ogr.OFTString
    :param field_length:
    :return:
    '''
    from osgeo import ogr
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(inputshp, 1)  # 1 is read/write

    # define integer field named 'classId'
    if field_type == 'int':
        fldDef = ogr.FieldDefn(field_name, ogr.OFTInteger) # or it could be ogr.OFTInteger, ogr.OFTReal, or ogr.OFTString
        fldDef.SetWidth(field_length)  # 16 char string width

    # get layer and add the 2 fields:
    layer = dataSource.GetLayer()
    layer.CreateField(fldDef)


#
def merge_shp(shp_list, output_filename):
    '''
    currently not working.. checking this link:
    http://learningzone.rspsoc.org.uk/index.php/Learning-Materials/Python-Scripting/8.2-Merging-ESRI-shapefiles
    :param input_dir:
    :param output_filename:
    :return:
    '''

    ## merge all shp files
    first = True
    command = ''
    # Iterate through the files.
    for shp in shp_list:
        if first:
            # If the first file make a copy to create the output file
            command = 'ogr2ogr ' + output_filename + ' ' + shp
            first = False
        else:
            os.system('ogr2ogr -update -append ' + output_filename + ' ' +  shp + ' -nln ' + os.path.basename(shp)[:-4])
        # Execute the current command
        os.system(command)


def readTiff(intif, data_type = np.float32):
    g = gdal.Open(intif)
    s0 = gdal_array.DatasetReadAsArray(g).astype(data_type)

    return g,s0



def hist_match(inputImage, templateImage):
    # TODO optimise with either cython or numba

    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Writes to the inputImage dataset so that it matches

    Notes:
    -----------

    As the entire band histogram is required this can become memory
    intensive with big rasters eg 10 x 10k+

    Inspire by/adapted from something on stack on image processing - credit to
    that author

    Parameters
    -----------

    inputImage : string
                 image to transform; the histogram is computed over the flattened array

    templateImage : string
                    template image can have different dimensions to source

    """
    # TODO - cythinis or numba this one
    sourceRas = gdal.Open(inputImage, gdal.GA_Update)
    templateRas = gdal.Open(templateImage)
    # Bands = list()
    bands = sourceRas.RasterCount

   # oldshape = ((sourceRas.RasterXSize, sourceRas.RasterYSize))
    oldshape = ((sourceRas.RasterYSize, sourceRas.RasterXSize))
    for band in tqdm(range(1, bands + 1)):
        # print(band)
        sBand = sourceRas.GetRasterBand(band)
        # seems to be issue with combining properties as with templateRas hence
        # separate lines
        sourceim = sBand.ReadAsArray().astype(np.float64)

        template = templateRas.GetRasterBand(band).ReadAsArray().astype(np.float64)
        sourceim2 = sourceim

        sourceim2[sourceim2==0] = np.nan
        template[template==0] = np.nan

        sourceim3 = sourceim2.ravel()
        template2 = template.ravel()

        # source = source.ravel()
        # template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(sourceim3, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template2, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)

        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        #pdb.set_trace()
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        out_array = interp_t_values[bin_idx].reshape(oldshape)

        # reuse the var from earlier
   #     pdb.set_trace()
        sBand.WriteArray(out_array)

    sourceRas.FlushCache()
    templateRas = None
    sourceRas = None


def hist_match2(inputImage, templateImage):
    # TODO optimise with either cython or numba

    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Writes to the inputImage dataset so that it matches

    Notes:
    -----------

    As the entire band histogram is required this can become memory
    intensive with big rasters eg 10 x 10k+

    Inspire by/adapted from something on stack on image processing - credit to
    that author

    Parameters
    -----------

    inputImage : string
                 image to transform; the histogram is computed over the flattened array

    templateImage : string
                    template image can have different dimensions to source

    """
    # TODO - cythinis or numba this one
    sourceRas = gdal.Open(inputImage, gdal.GA_Update)
    templateRas = gdal.Open(templateImage)
    # Bands = list()
    bands = sourceRas.RasterCount

    oldshape = ((sourceRas.RasterXSize, sourceRas.RasterYSize))

    for band in tqdm(range(1, bands + 1)):
        # print(band)
        sBand = sourceRas.GetRasterBand(band)
        # seems to be issue with combining properties as with templateRas hence
        # separate lines
        source = sBand.ReadAsArray()

        template = templateRas.GetRasterBand(band).ReadAsArray()

        # source = source.ravel()
        # template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source.ravel(), return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template.ravel(), return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        out_array = interp_t_values[bin_idx].reshape(oldshape)

        # reuse the var from earlier
        sBand.WriteArray(out_array)

    sourceRas.FlushCache()
    templateRas = None
    sourceRas = None


