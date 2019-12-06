from general_functions import clip_rst, clip_rst_by_rst
import os

# def test_segment_image():
#     segment_image(
#         in_path="/media/ubuntu/storage/Ghana/cocoa_upscale_test/segmentation_tinytest/s2_20180219_testsite_NIR_SWIR_red_small.tif",
#         out_path="/media/ubuntu/storage/Ghana/cocoa_upscale_test/segmentation_tinytest/segment_test.tif",
#         cores = 4,
#         shape_path="/media/ubuntu/storage/Ghana/cocoa_upscale_test/segmentation_tinytest/segment_test.shp"
#     )

def test_clip_image():
    clip_rst(
        in_tif= "/media/ubuntu/Data/Ghana/north_region/s2/segmentation/composite_20180306T103021_T30NWM_clip.tif",
        outline_shp="/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp",
        out_tif= "/media/ubuntu/Data/Ghana/north_region/s2/segmentation_temp/composite_20180306T103021_T30NWM_clip.tif",
        keep_rst_extent= True
    )


def test_clip_dir():
    # image_dir = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1"
    # shp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
    # out_dir = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_temp"

    image_dir = "/media/ubuntu/Data/Ghana/north_region/s2/segmentation"
    shp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
    out_dir = "/media/ubuntu/Data/Ghana/north_region/s2/segmentation_temp"
    for image in os.listdir(image_dir):
        if image.endswith(".tif"):
            clip_rst(in_tif= os.path.join(image_dir,image),
                     outline_shp= shp,
                     out_tif = os.path.join(out_dir,image))

def tes_clip_by_rst():
    in_tif = "/media/ubuntu/Data/Ghana/north_region/s2/segmentation/composite_20180219_T30NVN.tif"
    ref_tif = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_temp/composite_20180219_T30NVN.tif"
    out_tif = "/media/ubuntu/Data/Ghana/north_region/s2/segmentation_temp/composite_20180219_T30NVN.tif"
    clip_rst_by_rst(in_tif= in_tif,ref_tif=ref_tif,out_tif= out_tif)


def test_clip_dir_by_rst():
    # image_dir = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1"
    # shp = "/media/ubuntu/Data/Ghana/cocoa_big/shp/cocoa_big.shp"
    # out_dir = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_temp"

    image_dir = "/media/ubuntu/Data/Ghana/north_region/s2/segmentation"
    ref_dir = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_temp"
    out_dir = "/media/ubuntu/Data/Ghana/north_region/s2/segmentation_temp"
    for image in os.listdir(image_dir):
        if image.endswith(".tif"):
            clip_rst_by_rst(in_tif= os.path.join(image_dir,image),
                            ref_tif=os.path.join(ref_dir, image),
                            out_tif= os.path.join(out_dir,image))

if __name__ == "__main__":
    #test_segment_image()
    #tes_clip_by_rst()
    test_clip_dir_by_rst()