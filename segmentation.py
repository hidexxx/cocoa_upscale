import skimage
from skimage.segmentation import quickshift
from skimage.filters import median
from skimage.morphology import cube
import numpy as np
import gdal
import s2_functions


def exhaustive_quickshift_image_search(in_path, out_dir):
    params = {
        "convert2lab":[True, False],
        "ratio":[]

    }


def create_quickshift_mask(in_image_path, out_path, params = None):
    """Creates an image segmentation mask using quickshift. Saves results at out_path
    Uses quickshift coeffieicents from https://pure.aber.ac.uk/portal/files/29175686/remotesensing_11_00658.pdf,
    table 3"""
    swir_image = gdal.Open(in_image_path)
    swir_raw_array = swir_image.GetVirtualMemArray()
    swir_sklean_format = np.transpose(swir_raw_array, [1, 2, 0])
    #swir_sklean_format = scale_array_to_255(np.transpose(swir_raw_array, [1, 2, 0]))

    if params:
        class_masks = quickshift(swir_sklean_format, *params)

    class_masks = quickshift(swir_sklean_format, convert2lab=True, ratio=0.8, kernel_size=30, max_dist=3, sigma=0)
    #where ratio is the tradeoff between color importance and spatial importance (larger values give more importance to color), kernelsize is the size of the kernel used to estimate the density, and maxdist is the maximum distance between points in the feature space that may be linked if the density is increased.
    # http://www.vlfeat.org/overview/quickshift.html

    s2_functions.create_tif(filename=out_path, g=swir_image, Nx=swir_raw_array.shape[1],
                            Ny=swir_raw_array.shape[2], new_array=class_masks, noData=0, data_type=gdal.GDT_Int32)


def scale_array_to_255(in_array):
    arr = in_array
    new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype(np.uint8)
    return new_arr

# def scale_array_to_255(in_array):
#     scaled_array = np.zeros(in_array.shape)
#
#     for n in range(in_array.shape[2]):
#         arr = in_array[:,:,n]
#         new_arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype(np.uint8)
#         scaled_array[:,:,n] = new_arr
#     return scaled_array
#     scaled_array = None


def test_create_quickshift_mask():
    in_path = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_NIR_SWIR_red_small.tif"
    out_path = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_NIR_SWIR_red_small_obj_v5.tif"
    create_quickshift_mask(in_path, out_path)


if __name__ == "__main__":
    test_create_quickshift_mask()
