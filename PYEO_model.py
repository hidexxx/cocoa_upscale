import numpy as np
import sklearn.ensemble as ens
import gdal
from osgeo import ogr, osr,gdal_array
import os
from tempfile import TemporaryDirectory
from sklearn.model_selection import cross_val_score
import scipy.sparse as sp
import pickle
import learning_model
import pdb
import s2_functions
from tpot import TPOTClassifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import BernoulliNB


def create_matching_dataset(in_dataset, out_path,
                            format="GTiff", bands=1, datatype=None):
    """Creates an empty gdal dataset with the same dimensions, projection and geotransform. Defaults to 1 band.
    Datatype is set from the first layer of in_dataset if unspecified"""
    driver = gdal.GetDriverByName(format)
    if datatype is None:
        datatype = in_dataset.GetRasterBand(1).DataType
    out_dataset = driver.Create(out_path,
                                xsize=in_dataset.RasterXSize,
                                ysize=in_dataset.RasterYSize,
                                bands=bands,
                                eType=datatype)
    out_dataset.SetGeoTransform(in_dataset.GetGeoTransform())
    out_dataset.SetProjection(in_dataset.GetProjection())
    return out_dataset


def reshape_raster_for_ml(image_array):
    """Reshapes an array from gdal order [band, y, x] to scikit order [x*y, band]"""
    bands, y, x = image_array.shape
    image_array = np.transpose(image_array, (1, 2, 0))
    image_array = np.reshape(image_array, (x * y, bands))
    return image_array


def get_training_data(image_path, shape_path, attribute="CODE", shape_projection_id=4326):
    """Given an image and a shapefile with categories, return x and y suitable
    for feeding into random_forest.fit.
    Note: THIS WILL FAIL IF YOU HAVE ANY CLASSES NUMBERED '0'
    WRITE A TEST FOR THIS TOO; if this goes wrong, it'll go wrong quietly and in a way that'll cause the most issues
     further on down the line."""
    with TemporaryDirectory() as td:
        shape_projection = osr.SpatialReference()
        shape_projection.ImportFromEPSG(shape_projection_id)
        image = gdal.Open(image_path)
        image_gt = image.GetGeoTransform()
        x_res, y_res = image_gt[1], image_gt[5]
        ras_path = os.path.join(td, "poly_ras")
        ras_params = gdal.RasterizeOptions(
            noData=0,
            attribute=attribute,
            xRes=x_res,
            yRes=y_res,
            outputType=gdal.GDT_Int16,
            outputSRS=shape_projection
        )
        # This produces a rasterised geotiff that's right, but not perfectly aligned to pixels.
        # This can probably be fixed.
        gdal.Rasterize(ras_path, shape_path, options=ras_params)
        rasterised_shapefile = gdal.Open(ras_path)
        shape_array = rasterised_shapefile.GetVirtualMemArray()
        local_x, local_y = get_local_top_left(image, rasterised_shapefile)
        shape_sparse = sp.coo_matrix(shape_array)
        y, x, features = sp.find(shape_sparse)
        training_data = np.empty((len(features), image.RasterCount))
        image_array = image.GetVirtualMemArray()
        image_view = image_array[:,
                     local_y: local_y + rasterised_shapefile.RasterYSize,
                     local_x: local_x + rasterised_shapefile.RasterXSize
                     ]
        for index in range(len(features)):
            training_data[index, :] = image_view[:, y[index], x[index]]
    return training_data, features

def get_training_data_tif(image_path, training_tif_path):
    """Given an image and a shapefile with categories, return x and y suitable
    for feeding into random_forest.fit.
    Note: THIS WILL FAIL IF YOU HAVE ANY CLASSES NUMBERED '0'
    WRITE A TEST FOR THIS TOO; if this goes wrong, it'll go wrong quietly and in a way that'll cause the most issues
     further on down the line."""

    image = gdal.Open(image_path)
    training_tif = gdal.Open(training_tif_path)
    shape_array = gdal_array.DatasetReadAsArray(training_tif)
   # shape_array = training_tif.GetVirtualMemArray()
    shape_array[shape_array == -2147483648] = 0

    y, x, features = sp.find(shape_array)
    training_data = np.empty((len(features), image.RasterCount))
    image_array = image.GetVirtualMemArray()

    for index in range(len(features)):
       # pdb.set_trace()
        training_data[index, :] = image_array[:, y[index], x[index]]
    return training_data, features


def adjust_training_data_for_Ciaran(features, classes,bands):
    #features, classes = get_training_data(image_path, shape_path, attribute, shape_projection_id)
    # X_train, X_test, y_train, y_test = train_test_split(features.astype(np.float32),
    #                                                                classes.astype(np.float32), train_size=0.75,
    #                                                                test_size=0.25)
    X_train = features; y_train = classes

    all = np.zeros((X_train.shape[0],X_train.shape[1]+1))

    all[:, 0] = y_train

    for i in range(1,bands+1):
       # print(i)
        all[:,i] = X_train[:,i-1]

    all = all[np.isfinite(all).all(axis=1)]
    y_train = all[:,0]
    X_train = all[:,1:]
    return X_train,y_train

def get_training_for_cairan(image_path, shape_path,bands, attribute="CODE", shape_projection_id=4326):
    features, classes = get_training_data(image_path, shape_path, attribute, shape_projection_id)
    X_train,y_train = adjust_training_data_for_Ciaran(features, classes,bands=bands)
    return X_train,y_train

def get_training_data_for_Ciaran_dir(training_image_file_paths, bands,attribute="CODE"):
    learning_data = None
    classes = None
    for training_image_file_path in training_image_file_paths:
        print('working on ' + training_image_file_path)
        training_image_folder, training_image_name = os.path.split(training_image_file_path)
        training_image_name = training_image_name[:-4]  # Strip the file extension
        shape_path = os.path.join(training_image_folder, training_image_name, training_image_name + '.shp')
        this_training_data, this_classes = get_training_data(training_image_file_path, shape_path, attribute)
        if learning_data is None:
            learning_data = this_training_data
            classes = this_classes
        else:
            learning_data = np.append(learning_data, this_training_data, 0)
            classes = np.append(classes, this_classes)
    print(learning_data.shape)
    X_train, y_train = adjust_training_data_for_Ciaran(learning_data, classes,bands)
    return X_train,y_train

def train_cairan_model_dir(image_dir,outModel_path,bands,attribute = 'CODE'):
    image_paths = s2_functions.search_files_fulldir(input_path=image_dir,search_key='.tif',search_type='end')
    X_train,y_train = get_training_data_for_Ciaran_dir(training_image_file_paths=image_paths,attribute=attribute,bands=bands)

    export_training(y_train = y_train, out_dir = image_dir, summary_type='type')

    # Parameters for cross-validated exhaustive grid search
    paramsDict = {'n_estimators': [300],
                  'max_features': ['sqrt', 'log2'],
                  'min_samples_split': list(range(2, 13, 2)),
                  'min_samples_leaf': [5, 10, 20, 50, 100, 200, 500],
                  'max_depth': [10, None],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}

    resultList = learning_model.create_model(X_train, y_train, outModel=outModel_path, clf='rf',
                                             cores=6, params=paramsDict, scoring='accuracy')

def train_cairan_model_addNDVI_dir(image_dir,outModel_path,bands,attribute = 'CODE'):
    image_paths = s2_functions.search_files_fulldir(input_path=image_dir,search_key='.tif',search_type='end')
    X_train,y_train = get_training_data_for_Ciaran_dir(training_image_file_paths=image_paths,attribute=attribute,bands=bands)

    s0_before = X_train[:,0:4];s0_after = X_train[:,4:8]
    ndvi_diff = s2_functions.cal_vegIndex_diff(s0_before, s0_after)

    all = np.zeros((X_train.shape[0],X_train.shape[1]+1))

    for i in range(0,X_train.shape[1]):
        print(i)

        all[:,i] = X_train[:,i]
    all[:,-1] = ndvi_diff

    # Parameters for cross-validated exhaustive grid search
    paramsDict = {'n_estimators': [300],
                  'max_features': ['sqrt', 'log2'],
                  'min_samples_split': list(range(2, 13, 2)),
                  'min_samples_leaf': [5, 10, 20, 50, 100, 200, 500],
                  'max_depth': [10, None],
                  'bootstrap': [True, False],
                  'criterion': ['gini', 'entropy']}

    resultList = learning_model.create_model(all, y_train, outModel=outModel_path, clf='rf',
                                             cores=6, params=paramsDict, scoring='accuracy')


def get_local_top_left(raster1, raster2):
    """Gets the top-left corner of raster1 in the array of raster 2; WRITE A TEST FOR THIS"""
    inner_gt = raster2.GetGeoTransform()
    return point_to_pixel_coordinates(raster1, [inner_gt[0], inner_gt[3]])


def point_to_pixel_coordinates(raster, point, oob_fail=False):
    """Returns a tuple (x_pixel, y_pixel) in a georaster raster corresponding to the point.
    Point can be an ogr point object, a wkt string or an x, y tuple or list. Assumes north-up non rotated.
    Will floor() decimal output"""
    # Equation is rearrangement of section on affinine geotransform in http://www.gdal.org/gdal_datamodel.html
    if isinstance(point, str):
        point = ogr.CreateGeometryFromWkt(point)
        x_geo = point.GetX()
        y_geo = point.GetY()
    if isinstance(point, list) or isinstance(point, tuple):  # There is a more pythonic way to do this
        x_geo = point[0]
        y_geo = point[1]
    if isinstance(point, ogr.Geometry):
        x_geo = point.GetX()
        y_geo = point.GetY()
    gt = raster.GetGeoTransform()
    x_pixel = int(np.floor((x_geo - floor_to_resolution(gt[0], gt[1])) / gt[1]))
    y_pixel = int(np.floor((y_geo - floor_to_resolution(gt[3], gt[5] * -1)) / gt[5]))  # y resolution is -ve
    return x_pixel, y_pixel


def floor_to_resolution(input, resolution):
    """Returns input rounded DOWN to the nearest multiple of resolution."""
    return input - (input % resolution)


def reshape_ml_out_to_raster(classes, width, height):
    """Reshapes an output [x*y] to gdal order [y, x]"""
    # TODO: Test this.
    image_array = np.reshape(classes, (height, width))
    return image_array

def create_trained_model_path_TPOT(training_image_file_paths, model_out_path, generation = 5, population_size = 20, attribute="CODE"):
    """Returns a trained random forest model from the training data. This
    assumes that image and model are in the same directory, with a shapefile.
    Give training_image_path a path to a list of .tif files. See spec in the R drive for data structure.
    At present, the model is an ExtraTreesClassifier arrived at by tpot; see tpot_classifier_kenya -> tpot 1)"""
    # This could be optimised by pre-allocating the training array. but not now.
    learning_data = None
    classes = None
    for training_image_file_path in training_image_file_paths:
        training_image_folder, training_image_name = os.path.split(training_image_file_path)
        training_image_name = training_image_name[:-4]  # Strip the file extension
        shape_path = os.path.join(training_image_folder, training_image_name, training_image_name + '.shp')
        this_training_data, this_classes = get_training_data(training_image_file_path, shape_path, attribute)
        if learning_data is None:
            learning_data = this_training_data
            classes = this_classes
        else:
            learning_data = np.append(learning_data, this_training_data, 0)
            classes = np.append(classes, this_classes)

    X_train, X_test, y_train, y_test = train_test_split(learning_data.astype(np.float32),
                                                       classes.astype(np.float32), train_size=0.75,
                                                       test_size=0.25)
    model = TPOTClassifier(generations=generation, population_size=population_size, verbosity=2,n_jobs=-1)
    model.fit(X_train, y_train)
    scores = model.score(X_test, y_test)
    print(scores)
    model.export(model_out_path)
    return model, scores


def train_model_tpot(features, classes, outmodel, generations=5, population_size=20):
    model = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, n_jobs=-1)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, classes, train_size=0.75, test_size=0.25)

    model.fit(Xtrain, Ytrain)
    scores = model.score(Xtest, Ytest)
    print('TPOT model scores: ' + str(scores))
    print('saving model to: ' + outmodel)
    model.export(outmodel)
    return model, scores


def classify_image(in_image_path, model, out_image_path,num_chunks =10):
    print("Classifying image")
    image = gdal.Open(in_image_path)
    image_array = image.GetVirtualMemArray()
    features_to_classify = reshape_raster_for_ml(image_array)
    width = image.RasterXSize
    height = image.RasterYSize
    out_chunks = []
    for i, chunk in enumerate(np.array_split(features_to_classify, num_chunks)):
        print("Classifying {0}".format(i))
        chunk_copy = np.copy(chunk)
        chunk_copy = np.where(np.isfinite(chunk_copy), chunk_copy,0)  # this is a slower line
        out_chunks.append(model.predict(chunk_copy))
    out_classes = np.concatenate(out_chunks)

    image = gdal.Open(in_image_path)
    out_image = create_matching_dataset(image, out_image_path)
    image_array = None
    image = None

    out_image_array = out_image.GetVirtualMemArray(eAccess=gdal.GA_Update)
    out_image_array[...] = reshape_ml_out_to_raster(out_classes, width, height)
    out_image_array = None
    out_image = None


def save_model(model, filepath):
    with open(filepath, "wb") as fp:
        pickle.dump(model, fp)


def load_model(filepath):
    with open(filepath, "rb") as fp:
        return pickle.load(fp)

def summarise_training(in_classes,columns, out_csv, sumarise_type = 'count'):
    df = pd.DataFrame(in_classes, columns = [columns])
    if sumarise_type == 'count':
        training_summary = df.groupby([columns]).size()
    elif sumarise_type == 'mean':
        training_summary = df.groupby([columns]).mean()
    elif sumarise_type == 'median':
        training_summary = df.groupby([columns]).median
    else:
        print('Add more math func here, can only summarise in terms of count, mean or median now')
    training_summary.to_csv(out_csv)

def export_training(y_train, out_dir,summary_type = 'type'):
    out_csv = os.path.join(out_dir, 'training_data.csv')
    out_csv_summary = os.path.join(out_dir, 'training_data_summary.csv')
    df = pd.DataFrame(y_train, columns=[summary_type])
    df.to_csv(out_csv)
    summarise_training(y_train, columns=summary_type, out_csv=out_csv_summary, sumarise_type='count')

def train_model(features,classes, model_format,cross_val_repeats =5):
    model = model_format
    model.fit(features, classes)
    scores = cross_val_score(model, features, classes, cv=cross_val_repeats)
    return model, scores

def train_model_rf(features,classes):
    model = ens.ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.55, min_samples_leaf=2,
    min_samples_split=16, n_estimators=100, n_jobs=-1, class_weight='balanced')
    model.fit(features, classes)
    scores = cross_val_score(model, features, classes, cv=5)
    return model, scores


def train_model_tpot(features, classes, outmodel, generations=5, population_size=20):
    model = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, n_jobs=-1)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, classes, train_size=0.75, test_size=0.25)

    model.fit(Xtrain, Ytrain)
    scores = model.score(Xtest, Ytest)
    print('TPOT model scores: ' + str(scores))
    print('saving model to: ' + outmodel)
    model.export(outmodel)
    return model, scores




