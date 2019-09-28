import os
import numpy as np
from osgeo import gdal, gdal_array


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


def read_tif(intif,type = np.float):
    g = gdal.Open(intif)
    a = gdal_array.DatasetReadAsArray(g).astype(type)
    print("Read " + os.path.basename(intif))
    if a.ndim == 2:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]))
    else:
        print("the shape of the tif are: " + str(a.shape[0]) + ', ' + str(a.shape[1]) + ', ' + str(a.shape[2]))

    return g, a
