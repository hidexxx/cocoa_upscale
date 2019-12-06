import os
import shutil
from tempfile import TemporaryDirectory

from osgeo import gdal
from pyeo.apps.model_creation.download_and_preproc_area import main as dl_and_preproc
from pyeo import raster_manipulation as ras
from pyeo import filesystem_utilities as fs
from previous_code import s2_functions
import general_functions
import numpy as np


# # 2. s2
# # 2.1 download and preprocess :output 20m merge. tif (10 bands)
sen2cor_path = "/home/ubuntu/Downloads/Sen2Cor-02.08.00-Linux64/bin/L2A_Process"

aoi_path = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big_simp.geojson"
conf = "/media/ubuntu/Data/Ghana/cocoa_big/s2/cocoa_big.ini"

# l1_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/L1"
# l2_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/L2"
# merge_10m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/merge_10m"
# merge_20m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/merge_20m"
#
# merge_10m_clip_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/images/merged_clip2/10m"
# merge_20m_clip_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2/images/merged_clip2/20m"

# composite_10m_dir = "/media/ubuntu/Data/Ghana/north_region/s2/composites/10m"
# composite_20m_dir = "/media/ubuntu/Data/Ghana/north_region/s2/composites/20m"

l1_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2_batch2/images/L1"
l2_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2_batch2/images/L2"
merge_10m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2_batch2/images/merged/10m"
merge_20m_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2_batch2/images/merged/20m"

composite_10m_dir = "/media/ubuntu/Data/Ghana/north_region/s2/composites/10m"
composite_20m_dir = "/media/ubuntu/Data/Ghana/north_region/s2/composites/20m"

#inshp = '/media/ubuntu/storage/Ghana/shp/geojson/Western_North/Western_North.shp'
inshp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
#
def do_dl_and_preproc():
    dl_and_preproc(aoi_path=aoi_path, start_date='20180401', end_date='20180501',
                   l1_dir=l1_dir, l2_dir=l2_dir, merge_dir=merge_10m_dir, conf_path=conf,
             download_l2_data=False, sen2cor_path=sen2cor_path, stacked_dir=None, resolution=10, cloud_cover='20',
                   bands = ("B02", "B03", "B04", "B08"))
    ras.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge_20m_dir, l1_dir=l1_dir, cloud_threshold=0,
                                                    buffer_size=5, bands=("B02", "B03", "B04","B05","B06","B07","B8A","B11","B12"), out_resolution=20)

def move_l1_to_l2dir(l1_dir, l2_dir):
    for tile in os.listdir(l1_dir):
        if tile.endswith(".zip"):
            pass
        if "MSIL2A" in tile:
                try:
                    os.mkdir(l2_dir)
                except FileExistsError:
                    pass
                print("moving.." + tile + " to: " + l2_dir)
                from_path = os.path.join(l1_dir, tile)
                to_path = os.path.join(l2_dir, tile)
                shutil.move(from_path, to_path)


def do_atmCorr_merging(working_dir):
    os.chdir(working_dir)

    general_functions.make_all_dirs(working_dir)

    l1_dir = os.path.join(working_dir,"images/L1")
    l2_dir = os.path.join(working_dir,"images/L2")
    merge_10m_dir = os.path.join(working_dir,"images/merged/10m")
    merge_20m_dir = os.path.join(working_dir,"images/merged/20m")

    # ras.atmospheric_correction(in_directory=l1_dir, out_directory=l2_dir,
    #                                                  sen2cor_path=sen2cor_path, delete_unprocessed_image=False)
    #
    # move_l1_to_l2dir(l1_dir= l1_dir,l2_dir= l2_dir)
    #
    # ras.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge_10m_dir, l1_dir=l1_dir, cloud_threshold=0,
    #                                                 buffer_size=5, bands=("B02", "B03", "B04", "B08"), out_resolution=10)
    # #
    # ras.preprocess_sen2_images(l2_dir=l2_dir, out_dir=merge_20m_dir, l1_dir=l1_dir, cloud_threshold=0,
    #                                                 buffer_size=5, bands=("B02", "B03", "B04","B05","B06","B07","B8A","B11","B12"), out_resolution=20)
    #
    #sort_into_tile(merge_10m_dir)
    #sort_into_tile(merge_20m_dir)
    cloud_free_composite_dir(working_dir)


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
    sort_into_tile(indir = merge_10m_dir)
    sort_into_tile(indir = merge_20m_dir)



def clip_to_shp(intif, inshp, outtif):
    os.system('gdalwarp -cutline ' + inshp + ' -srcnodata 0 -dstnodata 0 -overwrite ' + intif + ' ' + outtif)

def clip_to_outline():
    intif = "/media/ubuntu/Data/Ghana/north_region/s2/composite/20m/composite_20180219_T30NVN.tif"
    inshp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
    outtif = intif[:-4] + '_clip2.tif'

    ras.clip_raster(raster_path= intif,aoi_path=inshp,
                    out_path=outtif,
                    srs_id = 32630)

def clip_dir():
    working_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2"
    inshp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
    os.chdir(working_dir)

    #for tile in os.listdir("images/merged/10m"):
    with TemporaryDirectory() as td:
        for tile in os.listdir("images/merged_clip2/10m"):
            tile_path = os.path.join("images/merged_clip2/10m", tile)
            for image in os.listdir(tile_path):
                print('Working on ' + image)

                try:
                    outpath_20m = "images/merged_clip2/20m/" + tile
                    os.mkdir(outpath_20m)
                except FileExistsError:
                    pass

                temp_outline = os.path.join(td,image[:-4]+'_outline.shp')
                ras.get_extent_as_shp(
                    in_ras_path=os.path.join(tile_path,image),
                    out_shp_path=temp_outline
                )

                time_stap = image.split("_")[2]

                file_list = s2_functions.search_files_fulldir(input_path= os.path.join("images/merged/20m", tile), search_type='contain', search_key=time_stap)

                image_suffix = image[-4:]

                for i in file_list:
                    if image_suffix in i:
                        image_20m_path = i


                out_image_20m_clip_path = os.path.join(outpath_20m,image)

                os.system('gdalwarp -cutline ' + temp_outline + ' -tr 20 20 -crop_to_cutline -overwrite  -srcnodata 0 -dstnodata 0 -ot UInt32 '+ image_20m_path + ' ' + out_image_20m_clip_path)

def move_merge():
    working_dir = "/media/ubuntu/Data/Ghana/cocoa_big/s2"
    os.chdir(working_dir)

    for tile in os.listdir("images/merged_clip2/10m"):
        tile_path = os.path.join("images/merged_clip2/10m", tile)
        for image in os.listdir(tile_path):
            print('moving.. ' + image)

            try:
                outpath_10m = "images/merged/10m/" + tile
                os.mkdir(outpath_10m)
            except FileExistsError:
                pass
            from_path = os.path.join("images/merged_backup/10m/",tile,image)
            to_path = os.path.join("images/merged/10m/",tile,image)
            shutil.move(from_path, to_path)

    for tile in os.listdir("images/merged_clip2/20m"):
        tile_path = os.path.join("images/merged_clip2/20m", tile)
        for image in os.listdir(tile_path):
            print('moving.. ' + image)

            try:
                outpath_20m = "images/merged/20m/" + tile
                os.mkdir(outpath_20m)
            except FileExistsError:
                pass
            from_path = os.path.join("images/merged_backup/20m/",tile,image)
            to_path = os.path.join("images/merged/20m/",tile,image)
            shutil.move(from_path, to_path)


def cloud_free_composite_dir(working_dir):
    os.chdir(working_dir)

    merge_10m_dir = "images/merged/10m"
    merge_20m_dir = "images/merged/20m"

    for tile in os.listdir(merge_10m_dir):
        tile_path = os.path.join(merge_10m_dir, tile)
        this_composite_path = ras.composite_directory(tile_path, "composites/10m_full")
        new_composite_path = "{}_{}.tif".format(this_composite_path.rsplit('.')[0], tile)
        os.rename(this_composite_path, new_composite_path)

    for tile in os.listdir(merge_20m_dir):
        tile_path = os.path.join(merge_20m_dir, tile)
        this_composite_path = ras.composite_directory(tile_path, "composites/20m_full")
        new_composite_path = "{}_{}.tif".format(this_composite_path.rsplit('.')[0], tile)
        os.rename(this_composite_path, new_composite_path)

def clip_north_region_dir():
    working_dir = "/media/ubuntu/Data/Ghana/north_region/s2"
    inshp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
    os.chdir(working_dir)

    for image in os.listdir("composite/10m"):
        with TemporaryDirectory() as td:
            if image.endswith(".tif"):
                ras.clip_raster(raster_path=os.path.join("composite/10m", image), aoi_path=inshp,
                                out_path=os.path.join("composites/10m",image),srs_id = 32630)

                temp_outline = os.path.join(td, image[:-4] + '_outline.shp')
                ras.get_extent_as_shp(
                    in_ras_path=os.path.join("composites/10m",image),
                    out_shp_path=temp_outline
                )

                general_functions.clip_rst(in_tif=os.path.join("composite/20m", image),
                                           outline_shp=temp_outline,
                                           out_tif=os.path.join("composites/20m", image), keep_rst_extent=False)
                #
                general_functions.clip_rst(in_tif=os.path.join("segmentation/brightness_all", image),
                                           outline_shp=temp_outline,
                                           out_tif=os.path.join("segmentation/brightness", image), keep_rst_extent=False)

def rst_to_10m(intif, outtif):
    os.system('gdalwarp -overwrite -tr 10 10 -r cubic ' + intif + ' ' + intif[:-4] + '_10m.tif')


def do_clip():
    composite_list = s2_functions.search_files_fulldir(input_path=composite_10m_dir, search_type='end',
                                                       search_key='.tif')
    print(composite_list)
    for compsite in composite_list:
        clip_to_shp(intif=compsite, inshp=inshp, outtif=compsite[:-4] + '_clip.tif')

    composite_list = s2_functions.search_files_fulldir(input_path=composite_20m_dir, search_type='end',
                                                       search_key='.tif')
    for compsite in composite_list:
        clip_to_shp(intif=compsite, inshp=inshp, outtif=compsite[:-4] + '_clip.tif')

def do_rst_to_10m():
    indir = merge_20m_dir
    composite_20m_list = s2_functions.search_files_fulldir(input_path=indir, search_key='composite',
                                                           search_type='start')
    for composite_20m in composite_20m_list:
        rst_to_10m(composite_20m, outtif=composite_20m[:-4] + '_10m.tif')

def generate_20m_6bands(in_20m_tif):
    g,arr = general_functions.read_tif(in_20m_tif)
    out_arr = arr[3:,:,:]
    filename = in_20m_tif[:-4] + '_6bands.tif'
    general_functions.create_tif(filename=filename,g=g,Nx=arr.shape[1],Ny=arr.shape[2],new_array=out_arr,data_type=gdal.GDT_Int16,noData=0)

def stack_for_testsite():
    working_dir = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/"
    os.chdir(working_dir)
    s1_vh = os.path.join("s1","s1_vh_20180402_testsite_10m.tif")
    s1_vv = os.path.join("s1","s1_vv_20180402_testsite_10m.tif")
    s2_10m = os.path.join("s2/composites/10m","s2_20180219_testsite.tif")
    s2_20m = os.path.join("s2","s2_20180219_testsite_20m_resample_6bands.tif")
    veg_index = os.path.join("s2/vegetation_index","s2_20180219_testsite.tif")
    seg = os.path.join("s2/segmentation/brightness","s2_20180219_testsite.tif")

    out_stack  = os.path.join(working_dir,'all_19bands_stack.tif')
    os.system('gdal_merge.py -separate -ot Int32 -n 0 -a_nodata 0 -o ' + out_stack + ' ' + veg_index + ' '+ s2_10m+ ' '+ s1_vh+ ' '+ s1_vv+ ' '+ s2_20m+ ' '+ seg )

def stack_for_testsite_13bands():
    working_dir = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/"
    os.chdir(working_dir)
    s1_vh = os.path.join("s1", "s1_vh_20180402_testsite_10m.tif")
    s1_vv = os.path.join("s1", "s1_vv_20180402_testsite_10m.tif")
    s2_10m = os.path.join("s2/composites/10m", "s2_20180219_testsite.tif")
    veg_index = os.path.join("s2/vegetation_index", "s2_20180219_testsite.tif")
    seg = os.path.join("s2/segmentation/brightness", "s2_20180219_testsite.tif")

    out_stack = os.path.join(working_dir, '_13bands_stack.tif')
    os.system(
        'gdal_merge.py -separate -ot Int32 -n 0 -a_nodata 0 -o ' + out_stack + ' ' + veg_index + ' ' + s2_10m + ' ' + s1_vh + ' ' + s1_vv + ' ' + seg)

    #ras.stack_images([veg_index, s2_10m, s1_vh,s1_vv,s2_20m,seg], out_stack)

def generate_seg_rst():
    '''
    :param s0:array read from s2 20m 9 bands data .tif or 10m 4 bands data.tif
    for s2 20m, the band sequence are: band 2, 3, 4, 5, 6, 7, 8a, 11, 12
    for s2 10m, the band sequence are: band 2, 3, 4, 8
    :return:
    '''
    merge_10m_dir = "/media/ubuntu/Data/Ghana/north_region/s2/composites/10m/"
    merge_20m_dir = "/media/ubuntu/Data/Ghana/north_region/s2/composites/20m/"
    tif_10m_list = s2_functions.search_files_fulldir(input_path=merge_10m_dir, search_type='end', search_key='NWM.tif')
    tif_20m_list = s2_functions.search_files_fulldir(input_path=merge_20m_dir, search_type='end', search_key='NWM.tif')

    for n in range(len(tif_10m_list)):
        tif_10m = tif_10m_list[n]
        tif_20m = tif_20m_list[n]
        print(tif_10m)
        print(tif_20m)
        out_tif = tif_10m[:-4] + "_NIR_SWIR_red.tif"

        g_10m, a_10m = general_functions.read_tif(tif_10m)
        a_10m_trans = np.transpose(a_10m, (1,2,0))

        g_20m, a_20m = general_functions.read_tif(tif_20m)
        a_20m_trans = np.transpose(a_20m, (1,2,0))

        a_3band = np.zeros((a_10m_trans.shape[0],a_10m_trans.shape[1],3), dtype= int)

    # # # version 3: NIR, SWIR, and red
        a_3band[:,:,0] = a_10m_trans[:,:,3]
        a_3band[:,:,1] = a_20m_trans[:,:,7]
        a_3band[:,:,2] = a_10m_trans[:,:,2]

        a_3band_trans = np.transpose(a_3band, (2,0,1))
        a_3band_out = tif_10m[:-4] + '_NIR_SWIR_red.tif'

        general_functions.create_tif(filename=a_3band_out, g = g_20m, Nx= a_20m.shape[1], Ny= a_20m.shape[2],new_array=a_3band_trans, noData= 0,data_type=gdal.GDT_UInt16)

def do_scale_to255():
    input_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_19bands_stack.tif"
    output_tif = input_tif[:-4] + "_scaled_255.tif"
   # general_functions.scale_tif(in_tif=input_tif,out_tif=output_tif,type = "robust")
    general_functions.scale_layer_to_255(intif=input_tif,outtif=output_tif)


if __name__ == "__main__":
    fs.init_log("ghana_s2.log")
    #generate_seg_rst()
    #do_preproc_only()
    #do_sort_into_tile()
    #cloud_free_composite_dir()

    #generate_20m_6bands(in_20m_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_20m_resample.tif")
    #stack_for_testsite()
 #   stack_for_testsite_13bands()
    #do_scale()
   # clip_to_outline()
    #clip_dir()
    #clip_north_region_dir()
    #
    do_atmCorr_merging(working_dir="/media/ubuntu/storage/cameroon/s2/")
    #plot_hist(in_tif= "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_13bands_stack.tif")
    #plot_hist(in_tif="/media/ubuntu/Data/Ghana/north_region/s2_NWN/images/stacked/with_s1_seg/composite_20180122T102321_T30NWN.tif")



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