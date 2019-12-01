import os
import s2_functions
import pyeo.raster_manipulation as ras

s2_testsite = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/s2/composites/10m/s2_20180219_testsite.tif"

s1_tif = "/media/ubuntu/Data/Ghana/cocoa_big/s1/s1_mosaic.tif"
shp = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/shp/cocoa_upscale_testsite.shp"

s1_testsite = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/s1/s2_20180219_testsite.tif"
os.system('gdalwarp -cutline '+  shp +' -crop_to_cutline '+ '-overwrite ' + s1_tif + ' '+ s1_testsite)

veg_index_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/s2/vegetation_index/s2_20180219_testsite.tif"

brightness ="/media/ubuntu/Data/Ghana/cocoa_upscale_test/s2/segmentation/brightness/s2_20180219_testsite.tif"

testsite_out_13 = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_13bands_stack_v2.tif"

os.system('gdal_merge.py -separate -ot Int32 -o ' + testsite_out_13 + ' '
            + veg_index_tif + ' ' + s2_testsite + ' ' + s1_testsite + ' ' + brightness)

#ras.stack_images([veg_index_tif, s2_tif, s1_testsite,brightness], testsite_out_13)




