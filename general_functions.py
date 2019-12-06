import os
import shutil
from difflib import get_close_matches
from tempfile import TemporaryDirectory

import numpy as np
from osgeo import gdal, gdal_array, ogr, osr
import pyeo.raster_manipulation as ras
#from build_cocoa_map import segment_image
from segmentation_raster import cal_seg_mean

from sklearn.preprocessing import robust_scale
import pdb
import matplotlib.pyplot as plt
import pylab


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
        try:
            new_array[np.isnan(new_array)] = noData
        except ValueError:
            pass

#        print 'changed no data to: ' +str(noData)
    print ('Create tiff: writing: '+ filename)
    if new_array.ndim == 2:
        Dataset.GetRasterBand(1).WriteArray(new_array)
    else:
        for i, image in enumerate(new_array):
            Dataset.GetRasterBand(i+1).WriteArray(image)
            Dataset.GetRasterBand(i+1).SetNoDataValue(noData)


def read_tif(intif,type = np.float):
    g = gdal.Open(intif)
    a = gdal_array.DatasetReadAsArray(g).astype(type)
    print("Read " + os.path.basename(intif))
    if a.ndim == 2:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]))
    else:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]) + ', ' + str(a.shape[2]))

    return g, a

def clip_rst(in_tif, outline_shp, out_tif, keep_rst_extent=True):
    if not keep_rst_extent:
        os.system('gdalwarp -cutline ' + outline_shp + ' -tr 10 10 -crop_to_cutline -overwrite  -srcnodata 0 -dstnodata 0 -ot UInt32 '
                  + in_tif + ' ' + out_tif)
    else:
        os.system(
            'gdalwarp -cutline ' + outline_shp + ' -tr 10 10 -overwrite -srcnodata 0 -dstnodata 0 -ot UInt32 ' + in_tif + ' ' + out_tif)


def clip_dir(image_dir, outline_shp, out_dir):
    for image in os.listdir(image_dir):
        if image.endswith(".tif"):
            clip_rst(in_tif= os.path.join(image_dir,image),
                     outline_shp= outline_shp,
                     out_tif = os.path.join(out_dir,image))

def create_seg_tif(working_dir, search_suffix = '.tif', export_pre_seg_tif = False, export_brightness = True):
    os.chdir(working_dir)
    with TemporaryDirectory() as td:
        for image in os.listdir("composites/10m"):
            if image.endswith(search_suffix):
                image_path_10m = os.path.join("composites/10m", image)
                image_path_20m = os.path.join("composites/20m", image)
                resample_path_20m = os.path.join(td, image)  # You'll probably regret this later, roberts.
                shutil.copy(image_path_20m, resample_path_20m)
                ras.resample_image_in_place(resample_path_20m, 10)

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
                temp_pre_seg_path = os.path.join(td, "pre_seg.tif")
                #temp_seg_path = os.path.join("seg.tif")

                # shape_projection = osr.SpatialReference()
                # shape_projection.ImportFromWkt(vis_10m.GetProjection())
                # image_gt = vis_10m.GetGeoTransform()
                # ras.save_array_as_image(array_to_classify, temp_pre_seg_path, image_gt, shape_projection)
                g,arr = read_tif(intif=image_path_10m)
                create_tif(filename=temp_pre_seg_path,g=g,Nx=arr.shape[1],Ny=arr.shape[2],new_array=array_to_classify,noData=0,data_type=gdal.GDT_UInt32)
                out_seg_tif = os.path.join("segmentation", image)
                #segment_image(temp_pre_seg_path, out_seg_tif)

                if export_pre_seg_tif == True:
                    shutil.copy(temp_pre_seg_path, os.path.join("pre_segmentation", image))

                if export_brightness == True:

                    in_value_ras = temp_pre_seg_path
                    seg_ras = out_seg_tif

                    try:
                        os.mkdir("segmentation/brightness")
                    except FileExistsError:
                        pass

                    out_value_ras = os.path.join("segmentation/brightness",image)
                    output_filtered_value_ras = False

                    temp_reshape_for_brightness =  os.path.join(td, "pre_brightness.tif")

                    clip_rst_by_rst(in_tif=seg_ras, ref_tif=in_value_ras,
                                                      out_tif=temp_reshape_for_brightness)
                    cal_seg_mean(in_value_ras, temp_reshape_for_brightness, out_value_ras,
                                 output_filtered_value_ras=output_filtered_value_ras)



def get_intersect_shp(raster_path, aoi_path, out_path, srs_id=4326):
    with TemporaryDirectory() as td:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(srs_id)
        intersection_path = os.path.join(td, 'intersection')
        raster = gdal.Open(raster_path)
        in_gt = raster.GetGeoTransform()
        aoi = ogr.Open(aoi_path)
        intersection = ras.get_aoi_intersection(raster, aoi)
        min_x_geo, max_x_geo, min_y_geo, max_y_geo = intersection.GetEnvelope()
        width_pix = int(np.floor(max_x_geo - min_x_geo) / in_gt[1])
        height_pix = int(np.floor(max_y_geo - min_y_geo) / np.absolute(in_gt[5]))
        new_geotransform = (min_x_geo, in_gt[1], 0, min_y_geo, 0, in_gt[5])
        ras.write_geometry(intersection, intersection_path, srs_id=srs_id)

        shutil.copy(intersection_path+"/geometry.shp", out_path)


def create_stack_image(working_dir,path_to_s1_image):
    os.chdir(working_dir)
    for image in os.listdir("images/stacked/with_indices"):
        if image.endswith(".tif"):
            path_to_image = os.path.join("images/stacked/with_indices", image)
            path_to_brightness = os.path.join("segmentation/brightness", image)
            ras.stack_images([path_to_image, path_to_s1_image, path_to_brightness], os.path.join("images/stacked/with_s1_seg", image))


def scale_array_to_255(in_array):
    arr = in_array
    new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype(np.uint8)
    return new_arr

def scale_layer_to_255(intif, outtif):
    g, array = read_tif(intif=intif, type=np.float)
    scaled_array = np.zeros(array.shape)
    for n in range(array.shape[0]):
        band = array[n,:,:]
        band[band==0] = np.nan

        if n == 10: # for  HV:
            band[band>=2000] = np.nan
        elif n == 11: # for VV
            band[band >= 9000] = np.nan
        else:
            band = band

        new_arr = ((band - np.nanmin(band)) * (1 / (np.nanmax(band) - np.nanmin(band)) * 255)).astype(np.uint16)
        scaled_array[n,:,:] = new_arr
    create_tif(filename=outtif,g=g,Nx=array.shape[1], Ny= array.shape[2],new_array= scaled_array,data_type=gdal.GDT_Int16,noData=0)
    array = None
    scaled_array = None

def scale_tif(in_tif,out_tif, type = 'robust'):
    g, arr = read_tif(intif=in_tif, type=np.float)
    arr_scaled = np.zeros(arr.shape)
    if type == 'robust':
        for n in range (arr_scaled.shape[0]):
            print("scaling band: " + str(n))
            band = arr[n]
            band_scaled = robust_scale(X= band)
            arr_scaled[n,:,:] = band_scaled
            print(str(np.mean(band_scaled)))

        print(arr_scaled.shape)
    create_tif(filename=out_tif,g=g,Nx=arr.shape[1], Ny= arr.shape[2],new_array= arr_scaled,data_type=gdal.GDT_Float32)


def plot_hist_all_density_to_one_graph(in_tif):
    g = gdal.Open(in_tif)
    a = g.GetVirtualMemArray()

    for band in range(a.shape[0]):
        print('plotting....' + str(band))
        value_array = a[band,:,:]

        mask = value_array ==0
        #        mask = bm_array<=0
        #    mask = bm_array< -10
        value_ma = value_array[mask == False]

        edg = np.arange(np.nanmin(value_ma), np.nanmax(value_ma), 10)
        # h = np.zeros(len(edg)-1)

        hist, j = np.histogram(value_ma, edg, density=True)
        plt.plot(edg[:-1], hist, label='band : ' +str(band))
        # plt.title( 'histgram of bm generated from different models')
        plt.ylabel('Density')
        plt.xlabel('Band Value')
        plt.xlim(-20, 10000)
        # plt.ylim(0,30000)
        # plt.ylim(0,300000)
        plt.legend(frameon=False)
    plt.show()


def plot_hist(in_tif):
    g = gdal.Open(in_tif)
    a = g.GetVirtualMemArray()
    #fig = pylab.gcf()

    band_name_list = ["ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
                         "s2_r", "s2_nir", "hv", "vv", "seg"]

    for band in range(a.shape[0]):
        print('plotting......' + os.path.basename(in_tif))
        band_name = band_name_list[band]
        print('band : ' + band_name)
        value_array = a[band,:,:]

        mask = value_array ==0
        #        mask = bm_array<=0
        #    mask = bm_array< -10
        value_ma = value_array[mask == False]

        edg = np.arange(np.nanmin(value_ma), np.nanmax(value_ma), 10)
        # h = np.zeros(len(edg)-1)
        hist, j = np.histogram(value_ma, edg, density=True)

        plt.figure(1,figsize=(12,12))

        ax1 = pylab.subplot(a.shape[0],2,band+1)

        ax1.plot(edg[:-1], hist, label='band : ' +band_name)
        # plt.title( 'histgram of bm generated from different models')
       # ax1.ylabel('Density')
       # ax1.xlabel('Band Value')
        #ax1.xlim(-20, 10000)
        #plt.ylim(0,30000)
        ax1.legend(frameon=False)
    pylab.subplots_adjust(hspace=0, bottom=None, top=None)
    pylab.show()
    #plt.show()
    pylab.savefig(in_tif[:-4]+'_hist.png')


def do_scale(ref_image, tobe_scaled_image,out_scaled_image, type = 'linear'):
    if type == 'linear' or type == None:
        print("here")

    elif type == '255':
        scale_array_to_255(intif=tobe_scaled_image,outtif=out_scaled_image)



def fix_class_id_from_name(shape_path, id_field, name_field, name_to_id_dict):
    """DANGER: IN-PLACE FUNCTION. DO NOT USE WITHOUT BACKING UP SHAPEFILE.
    Walks through every feature in shapefile at shape_path, comparing name_field to
    an internal lookup table and replacing id_field appropriately."""
    shapefile = ogr.Open(shape_path, 1)
    layer = shapefile.GetLayer()
    for feature in layer:
        # If the name of a feature is within match_threshold of a key in name_to_id_dict, update id_field
        name = feature.GetField(name_field)
        match = get_close_matches(name, name_to_id_dict.keys(), n = 1)[0]
        try:
            print("Setting {} to id value {}".format(name, name_to_id_dict[match]))
            feature.SetField(id_field, name_to_id_dict[match])
        except KeyError:
            print("No match for {} found in name_to_id_dict".format(name))
        layer.SetFeature(feature)
        feature = None
    layer = None
    shapefile = None


def clean_shp(shape_path, id_field, name_field):

    name_to_id_dict = {
        "Forest": 1,
        "AF Cocoa": 4,
        "AFCocoa": 4,
        "Agricultural land": 5,
        "Built-up": 7,
        "Cocoa": 3,
        "Oil Palm": 6,
        "Palm": 6,
        "Open Forest":2,
        "OpenForest": 2,
        "Transition":8
    }
    fix_class_id_from_name(shape_path, id_field, name_field, name_to_id_dict)

def make_msk(in_image_path, apply_msk = False):
    image, image_array = read_tif(in_image_path)
    print("Generating extent mask for mosaicing: ")
    mask_array = image_array[6, :, :] # first 6 bands are vegetation index, band 7-10 are s2 10m bands
    mask_array[mask_array != 0] = 1
    create_tif(filename=in_image_path[:-4] + '_mask.msk', g=image, Nx=mask_array.shape[0],
                                 Ny=mask_array.shape[1],
                                 new_array=mask_array, data_type=gdal.GDT_UInt16, noData=0)

    if apply_msk:
        image_out = image * mask_array
        create_tif(filename=in_image_path[:-4] + '_masked.tif', g=image, Nx=mask_array.shape[0],
                   Ny=mask_array.shape[1],
                   new_array=image_out, data_type=gdal.GDT_UInt32, noData=0)

def apply_msk_to_classified_img(in_image_path,in_msk_path,out_image_path):
    image, image_array = read_tif(in_image_path)
    msk, msk_array = read_tif(in_msk_path)

    image_masked_array = image_array*msk_array

    create_tif(filename=out_image_path, g=image, Nx=image_array.shape[0],
                                 Ny=image_array.shape[1],
                                 new_array=image_masked_array, data_type=gdal.GDT_UInt16, noData=0)
def make_directory(in_directory):
    try:
        os.mkdir(in_directory)
    except FileExistsError:
        pass

def do_mask(working_dir, generate_mask = True):
    os.chdir(working_dir)

    for classifed_image in os.listdir("output/"):
        if classifed_image.endswith(".tif"):
            if generate_mask:
                stack  = os.path.join("images/stacked/with_s1_seg",classifed_image)
                make_msk(in_image_path=stack)

            out_msk = os.path.join("images/stacked/with_s1_seg",classifed_image[:-4]+'_mask.msk')

            make_directory("output/filtered")

            out_image_path = os.path.join("output/filtered",classifed_image)

            apply_msk_to_classified_img(in_image_path = os.path.join("output/",classifed_image),
                                        in_msk_path = out_msk, out_image_path= out_image_path)

def get_extent_as_shp(in_ras_path, out_shp_path):
    """"""
    #By Qing
    os.system('gdaltindex ' + out_shp_path + ' ' + in_ras_path)
    return out_shp_path


def make_all_dirs(working_dir):
    os.chdir(working_dir)
    make_directory("images")
    make_directory("images/merged")
    make_directory("images/merged/10m")
    make_directory("images/merged/20m")

    make_directory("images/stacked")
    make_directory("images/stacked/with_indices")
    make_directory("images/stacked/with_s1_seg")
    make_directory("images/stacked/all_19bands")

    make_directory("composites")
    make_directory("composites/10m")
    make_directory("composites/20m")
    make_directory("composites/10m_full")
    make_directory("composites/20m_full")

    make_directory("segmentation")

    make_directory("output")

    make_directory("log")


#==============some old code that will be deleted later
# def generate_seg_rst():
#     '''
#     :param s0:array read from s2 20m 9 bands data .tif or 10m 4 bands data.tif
#     for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
#     for s2 10m, the band sequence are: band 2, 3, 4, 8
#     :return:
#     '''
#     tif_10m_list = s2_functions.search_files_fulldir(input_path=merge_10m_dir,search_type='end',search_key='_clip.tif')
#     tif_20m_list = s2_functions.search_files_fulldir(input_path=merge_20m_dir,search_type='end',search_key='10m_clip.tif')
#
#     for n in range(len(tif_10m_list)):
#         tif_10m = tif_10m_list[n]
#         tif_20m = tif_20m_list[n]
#         print(tif_10m)
#         print(tif_20m)
#         out_tif = tif_10m[:-4] + "_NIR_SWIR_red.tif"
#
#         g_10m, a_10m = general_functions.read_tif(tif_10m)
#         a_10m_trans = np.transpose(a_10m, (1,2,0))
#
#         g_20m, a_20m = general_functions.read_tif(tif_20m)
#         a_20m_trans = np.transpose(a_20m, (1,2,0))
#
#         a_3band = np.zeros((a_10m_trans.shape[0],a_10m_trans.shape[1],3), dtype= int)
#
#     # # # version 3: NIR, SWIR, and red
#         a_3band[:,:,0] = a_10m_trans[:,:,3]
#         a_3band[:,:,1] = a_20m_trans[:,:,7]
#         a_3band[:,:,2] = a_10m_trans[:,:,2]
#
#         a_3band_trans = np.transpose(a_3band, (2,0,1))
#         a_3band_out = tif_10m[:-4] + '_NIR_SWIR_red.tif'
#
#         general_functions.create_tif(filename=a_3band_out, g = g_20m, Nx= a_20m.shape[1], Ny= a_20m.shape[2],new_array=a_3band_trans, noData= 0,data_type=gdal.GDT_Int16)




