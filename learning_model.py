
try:
    #import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    import xgboost as xgb
except ImportError:
    pass
    print('xgb not available')



from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import OrderedDict
#import os
import glob
from sklearn import svm
import gdal, ogr#,osr
import numpy as np
from sklearn.model_selection import StratifiedKFold
#from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
#from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
import joblib as jb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import site
lib_path = "/home/ubuntu/Documents/Code/geospatial-learn/geospatial_learn"
site.addsitedir(lib_path)


from scipy.stats import randint as sp_randint
from scipy.stats import expon
#from scipy.sparse import csr_matrix
from tpot import TPOTClassifier, TPOTRegressor


gdal.UseExceptions()
ogr.UseExceptions()


def create_model(X_train, y_train,outModel, clf='svc', random=False, cv=6, cores=-1,
                 strat=True, regress=False, params=None, scoring=None):
    """
    Brute force or random model creating using scikit learn. Either use the
    default params in this function or enter your own (recommended - see sklearn)

    Parameters
    ---------------

    X_train : np array
              numpy array of training data where the 1st column is labels

    outModel : string
               the output model path which is a gz file

    clf : string
          an sklearn or xgb classifier/regressor
          logit, sgd, linsvc, svc, svm, nusvm, erf, rf, gb, xgb

    random : bool
             if True, a random param search

    cv : int

    cores : int or -1 (default)
            the no of parallel jobs

    strat : bool
            a stratified grid search

    regress : bool
              a regression model if True, a classifier if False

    params : a dict of model params (see scikit learn)
             enter your own params dict rather than the range provided

    scoring : string
              a suitable sklearn scoring type (see notes)


    General Note:
    --------------------
        There are more sophisticated ways to tune a model, this greedily
        searches everything but can be computationally costly. Fine tuning
        in a more measured way is likely better. There are numerous books,
        guides etc...
        E.g. with gb- first tune no of trees for gb, then learning rate, then
        tree specific

    Notes on algorithms:
    ----------------------
        From my own experience and reading around


        sklearn svms tend to be not great on large training sets and are
        slower with these (i have tried on HPCs and they time out on multi fits)

        sklearn 'gb' is very slow to train, though quick to predict

        xgb is much faster, but rather different in algorithmic detail -
        ie won't produce same results as sklearn...

        xgb also uses the sklearn wrapper params which differ from those in
        xgb docs, hence they are commented next to the area of code

        Scoring types - there are a lot - some of which won't work for
        multi-class, regression etc - see the sklearn docs!

        'accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
        'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',
        'neg_mean_absolute_error', 'neg_mean_squared_error',
        'neg_median_absolute_error', 'precision', 'precision_macro',
        'precision_micro', 'precision_samples', 'precision_weighted',
        'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
        'recall_weighted', 'roc_auc'

    """
    # # t0 = time()
    #
    # min_max_scaler = preprocessing.MinMaxScaler()
    # print('Preparing data')
    # # TODO IMPORTANT add xgb boost functionality
    # # inputImage = gdal.Open(inputIm)
    #
    # """
    # Prep of data for model fitting
    # """
    #
    # bands = X_train.shape[1] - 1
    #
    # # X_train = X_train.transpose()
    #
    # X_train = X_train[X_train[:, 0] != 0]
    #
    # # Remove non-finite values
    # X_train = X_train[np.isfinite(X_train).all(axis=1)]
    # # y labels
    # y_train = X_train[:, 0]
    #
    # # remove labels from X_train
    # X_train = X_train[:, 1:bands + 1]
    if scoring is None and regress is False:
        scoring = 'accuracy'
    elif scoring is None and regress is True:
        scoring = 'r2'
    # Choose the classifier type
    # TODO this has become rather messy and inefficient - need to make it more
    # elegant
    if clf == 'erf':
        RF_clf = ExtraTreesClassifier(n_jobs=cores)
        if random == True:
            param_grid = {"max_depth": [10, None],
                          "n_estimators": [500],
                          "min_samples_split": sp_randint(1, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}

            # run randomized search
            grid = RandomizedSearchCV(RF_clf, param_distributions=param_grid,
                                      n_jobs=-1, n_iter=20, verbose=2)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel)
            # print("done in %0.3fs" % (time() - t0))
        else:
            if params is None:
                # currently simplified for processing speed
                param_grid = {"n_estimators": [500],
                              "max_features": ['sqrt', 'log2'],
                              "max_depth": [10, None],
                              "min_samples_split": [2, 3, 5],
                              "min_samples_leaf": [5, 10, 20, 50, 100],
                              "bootstrap": [True, False]}
            else:
                param_grid = params
            if strat is True and regress is False:
                grid = GridSearchCV(RF_clf, param_grid=param_grid,
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
            elif strat is False and regress is False:
                grid = GridSearchCV(RF_clf, param_grid=param_grid,
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)

        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)

    if clf is 'xgb' and regress is False:
        xgb_clf = XGBClassifier()
        if params is None:
            # This is based on the Tianqi Chen author of xgb
            # tips for data science as a starter
            # he recommends fixing trees - I haven't as default here...
            # crunch this first then fine tune rest
            #
            ntrees = 500
            param_grid = {'n_estimators': [ntrees],
                          'learning_rate': [0.1],  # fine tune last
                          'max_depth': [4, 6, 8, 10],
                          'colsample_bytree': [0.4, 0.6, 0.8, 1.0]}
        # total available...
        #            param_grid={['reg_lambda',
        #                         'max_delta_step',
        #                         'missing',
        #                         'objective',
        #                         'base_score',
        #                         'max_depth':[6, 8, 10],
        #                         'seed',
        #                         'subsample',
        #                         'gamma',
        #                         'scale_pos_weight',
        #                         'reg_alpha', 'learning_rate',
        #                         'colsample_bylevel', 'silent',
        #                         'colsample_bytree', 'nthread',
        #                         'n_estimators', 'min_child_weight']}
        else:
            param_grid = params
        grid = GridSearchCV(xgb_clf, param_grid=param_grid,
                            cv=StratifiedKFold(cv), n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
    if clf is 'gb' and regress is False:
        # Key parameter here is max depth
        gb_clf = GradientBoostingClassifier()
        if params is None:
            param_grid = {"n_estimators": [100],
                          "learning_rate": [0.1, 0.75, 0.05, 0.025, 0.01],
                          "max_features": ['sqrt', 'log2'],
                          "max_depth": [3, 5],
                          "min_samples_leaf": [5, 10, 20, 30]}
        else:
            param_grid = params
        #                       cut due to time
        if strat is True and regress is False:
            grid = GridSearchCV(gb_clf, param_grid=param_grid,
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is False and regress is False:
            grid = GridSearchCV(gb_clf, param_grid=param_grid,
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)

    if clf is 'gb' and regress is True:
        gb_clf = GradientBoostingRegressor(n_jobs=cores)
        if params is None:
            param_grid = {"n_estimators": [500],
                          "loss": ['ls', 'lad', 'huber', 'quantile'],
                          "learning_rate": [0.1, 0.75, 0.05, 0.025, 0.01],
                          "max_features": ['sqrt', 'log2'],
                          "max_depth": [3, 5],
                          "min_samples_leaf": [5, 10, 20, 30]}
        else:
            param_grid = params

        grid = GridSearchCV(gb_clf, param_grid=param_grid,
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)

        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)

        # Find best params----------------------------------------------------------
    if clf == 'rf' and regress is False:
        RF_clf = RandomForestClassifier(n_jobs=cores, random_state=123)
        if random == True:
            param_grid = {"max_depth": [10, None],
                          "n_estimators": [500],
                          "min_samples_split": sp_randint(1, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}

            # run randomized search
            grid = RandomizedSearchCV(RF_clf, param_distributions=param_grid,
                                      n_jobs=-1, n_iter=20, verbose=2)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel)
            # print("done in %0.3fs" % (time() - t0))
        else:
            if params is None:
                # currently simplified for processing speed
                param_grid = {"n_estimators": [500],
                              "max_features": ['sqrt', 'log2'],
                              "max_depth": [10, None],
                              "min_samples_split": [2, 3, 5],
                              "min_samples_leaf": [5, 10, 20, 50, 100, 200, 500],
                              "bootstrap": [True, False]}
            else:
                param_grid = params
            if strat is True and regress is False:
                grid = GridSearchCV(RF_clf, param_grid=param_grid,
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
            elif strat is False and regress is False:
                grid = GridSearchCV(RF_clf, param_grid=param_grid,
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)

        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)

    if clf is 'rf' and regress is True:
        RF_clf = RandomForestRegressor(n_jobs=cores, random_state=123)
        if params is None:
            param_grid = {"n_estimators": [500],
                          "max_features": ['sqrt', 'log2'],
                          "max_depth": [10, None],
                          "min_samples_split": [2, 3, 5],
                          "min_samples_leaf": [5, 10, 20, 50, 100, 200, 500],
                          "bootstrap": [True, False]}
        else:
            param_grid = params
        grid = GridSearchCV(RF_clf, param_grid=param_grid,
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)

        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
        # print("done in %0.3fs" % (time() - t0))

    # Random can be quicker and more often than not produces close to
    # exaustive results
    if clf == 'linsvc' and regress is False:
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.LinearSVC()
        if random == True:
            param_grid = [{'C': [expon(scale=100)], 'class_weight': ['auto', None]}]
            # param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel)
            # print("done in %0.3fs" % (time() - t0))
        else:
            param_grid = [{'C': [1, 10, 100, 1000], 'class_weight': ['auto', None]}]
            # param_grid = [{'kernel':['rbf', 'linear']}]
            if strat is True:
                grid = GridSearchCV(svm_clf, param_grid=param_grid,
                                    cv=StratifiedKFold(cv), n_jobs=cores,
                                    scoring=scoring, verbose=2)
            elif strat is False and regress is False:
                grid = GridSearchCV(svm_clf, param_grid=param_grid,
                                    cv=cv, n_jobs=cores,
                                    scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
    if clf is 'linsvc' and regress is True:
        svm_clf = svm.LinearSVR()
        if params is None:
            param_grid = [{'C': [1, 10, 100, 1000]},
                          {'loss': ['epsilon_insensitive',
                                    'squared_epsilon_insensitive']}]
        else:
            param_grid = params
        grid = GridSearchCV(svm_clf, param_grid=param_grid,
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
        # print("done in %0.3fs" % (time() - t0))
    if clf == 'svc':  # Far too bloody slow
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.SVC(probability=False)
        if random == True:
            if params is None:

                param_grid = [{'C': [expon(scale=100)], 'gamma': [expon(scale=.1).astype(float)],
                               'kernel': ['rbf'], 'class_weight': ['auto', None]}]
            else:
                param_grid = params
            # param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel)
            # print("done in %0.3fs" % (time() - t0))

        if params is None:

            param_grid = [{'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4],
                           'kernel': ['rbf'], 'class_weight': ['auto', None]}]
        else:
            param_grid = params
            # param_grid = [{'kernel':['rbf', 'linear']}]
        if strat is True and regress is False:
            grid = GridSearchCV(svm_clf, param_grid=param_grid,
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is False and regress is False:
            grid = GridSearchCV(svm_clf, param_grid=param_grid,
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
        # print("done in %0.3fs" % (time() - t0))

    if clf == 'nusvc' and regress is False:
        X_train = min_max_scaler.fit_transform(X_train)
        svm_clf = svm.NuSVC(probability=False)
        if random == True:
            if params is None:
                param_grid = [{'nu': [0.25, 0.5, 0.75, 1], 'gamma': [expon(scale=.1).astype(float)],
                               'class_weight': ['auto']}]
            else:
                param_grid = params
            # param_grid = [{'kernel':['rbf', 'linear']}]
            grid = GridSearchCV(svm_clf, param_grid=param_grid, cv=cv,
                                scoring=scoring, verbose=1, n_jobs=cores)
            grid.fit(X_train, y_train)
            joblib.dump(grid.best_estimator_, outModel)
            # print("done in %0.3fs" % (time() - t0))
        else:
            if params is None:
                param_grid = [{'nu': [0.25, 0.5, 0.75, 1], 'gamma': [1e-3, 1e-4],
                               'class_weight': ['auto']}]
            else:
                param_grid = params
            # param_grid = [{'kernel':['rbf', 'linear']}]
        if strat is True and regress is False:
            grid = GridSearchCV(svm_clf, param_grid=param_grid,
                                cv=StratifiedKFold(cv), n_jobs=cores,
                                scoring=scoring, verbose=2)
        elif strat is True and regress is False:
            grid = GridSearchCV(svm_clf, param_grid=param_grid,
                                cv=cv, n_jobs=cores,
                                scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
    if clf is 'nusvc' and regress is True:
        svm_clf = svm.NuSVR()
        param_grid = [{'nu': [0.25, 0.5, 0.75, 1], 'gamma': [1e-3, 1e-4]}]
        grid = GridSearchCV(svm_clf, param_grid=param_grid,
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)
        # print("done in %0.3fs" % (time() - t0))
    if clf is 'logit':
        logit_clf = LogisticRegression()
        if params is None:
            param_grid = [{'C': [1, 10, 100, 1000], 'penalty': ['l1', 'l2', ],
                           'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                           'multi_class': ['ovr', 'multinomial']}]
        else:
            param_grid = params
        grid = GridSearchCV(logit_clf, param_grid=param_grid,
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)

    if clf is 'sgd':
        logit_clf = SGDClassifier()
        if params is None:
            param_grid = [{'loss': ['hinge, log', 'modified_huber',
                                    'squared_hinge', 'perceptron'],
                           'penalty': ['l1', 'l2', 'elasticnet'],
                           'learning_rate': ['constant', 'optimal', 'invscaling'],
                           'multi_class': ['ovr', 'multinomial']}]
        else:
            param_grid = params
        grid = GridSearchCV(logit_clf, param_grid=param_grid,
                            cv=cv, n_jobs=cores,
                            scoring=scoring, verbose=2)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, outModel)

    return [grid.cv_results_, grid.best_score_, grid.best_params_, grid.best_estimator_]
#    print(grid.best_params_)
#    print(grid.best_estimator_)
#    print(grid.oob_score_)
#
#    plt.plot(est_range, grid_mean_scores)
#    plt.xlabel('no of estimators')
#    plt.ylabel('Cross validated accuracy')


# Save the model
#    joblib.dump(grid.best_estimator_, outModel+'.pkl')
#    print("done in %0.3fs" % (time() - t0))

def _bbox_to_pixel_offsets(rgt, geom):
    """
    Internal function to get pixel geo-locations of bbox of a polygon

    Parameters
    ----------

    rgt : array
          List of points defining polygon (?)

    geom : shapely.geometry
           Structure defining geometry

    Returns
    -------
    int
       x offset

    int
       y offset

    xcount : int
             rows of bounding box

    ycount : int
             columns of bounding box
    """

    xOrigin = rgt[0]
    yOrigin = rgt[3]
    pixelWidth = rgt[1]
    pixelHeight = rgt[5]
    ring = geom.GetGeometryRef(0)
    numpoints = ring.GetPointCount()
    pointsX = [];
    pointsY = []

    if (geom.GetGeometryName() == 'MULTIPOLYGON'):
        count = 0
        pointsX = [];
        pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
            count += 1
    elif (geom.GetGeometryName() == 'POLYGON'):
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = [];
        pointsY = []
        for p in range(numpoints):
            lon, lat, z = ring.GetPoint(p)
            pointsX.append(lon)
            pointsY.append(lat)

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin) / pixelWidth)
    yoff = int((yOrigin - ymax) / pixelWidth)
    xcount = int((xmax - xmin) / pixelWidth) + 1
    ycount = int((ymax - ymin) / pixelWidth) + 1
    #    originX = gt[0]
    #    originY = gt[3]
    #    pixel_width = gt[1]
    #    pixel_height = gt[5]
    #    x1 = int((bbox[0] - originX) / pixel_width)
    #    x2 = int((bbox[1] - originX) / pixel_width) + 1
    #
    #    y1 = int((bbox[3] - originY) / pixel_height)
    #    y2 = int((bbox[2] - originY) / pixel_height) + 1
    #
    #    xsize = x2 - x1
    #    ysize = y2 - y1
    #    return (x1, y1, xsize, ysize)
    return (xoff, yoff, xcount, ycount)


def get_training(inShape, inRas, bands, field, outFile=None):
    """
    Collect training as an np array for use with create model function

    Parameters
    --------------

    inShape : string
              the input shapefile - must be esri .shp at present

    inRas : string
            the input raster from which the training is extracted

    bands : int
            no of bands

    field : string
            the attribute field containing the training labels

    outFile : string (optional)
              path to the training file saved as joblib format (eg - 'training.gz')

    Returns
    ---------------------

    A tuple containing:
    -np array of training data
    -list of polygons with invalid geometry that were not collected

    """
    # t0 = time()
    outData = list()
    print('Loading & prepping data')
    raster = gdal.Open(inRas)
    shp = ogr.Open(inShape)
    lyr = shp.GetLayer()
    labels = np.arange(lyr.GetFeatureCount())
    rb = raster.GetRasterBand(1)
    rgt = raster.GetGeoTransform()
    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')
    rejects = []

    print('calculating stats')
    for label in tqdm(labels):
        # print(label)
        feat = lyr.GetFeature(label)
        if feat == None:
            print('no geometry for feature ' + str(label))
            continue
        iD = feat.GetField(field)
        geom = feat.GetGeometryRef()

        # Get raster georeference info

        src_offset = _bbox_to_pixel_offsets(rgt, geom)

        # calculate new geotransform of the feature subset
        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5])

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # Use the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly

        rb = raster.GetRasterBand(1)
        src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                                   src_offset[3])
        if np.shape(src_array) is ():
            rejects.append(label)
            continue
        # Read raster as arrays
        for band in range(1, bands + 1):

            rb = raster.GetRasterBand(band)
            src_array = rb.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                                       src_offset[3])
            if src_array is None:
                src_array = rb.ReadAsArray(src_offset[0] - 1, src_offset[1], src_offset[2],
                                           src_offset[3])

            masked = np.ma.MaskedArray(src_array,
                                       mask=np.logical_or(src_array == 0,
                                                          np.logical_not(rv_array)))

            datafinal = masked.flatten()

            if band == 1:
                X = np.zeros(shape=(datafinal.shape[0], bands + 1))
            X[:, 0] = iD

            X[:, band] = datafinal

        outData.append(X)
    outData = np.asarray(outData)
    outData = np.concatenate(outData).astype(None)

    if outFile != None:
        jb.dump(outData, outFile, compress=2)

    return outData, rejects


def _copy_dataset_config(inDataset, FMT='Gtiff', outMap='copy',
                         dtype=gdal.GDT_Int32, bands=1):
    """Copies a dataset without the associated rasters.

    """
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'

    x_pixels = inDataset.RasterXSize  # number of pixels in x
    y_pixels = inDataset.RasterYSize  # number of pixels in y
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  # size of the pixel...they are square so thats ok.
    # if not would need w x h
    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    # dtype=gdal.GDT_Int32
    driver = gdal.GetDriverByName(FMT)

    # Set params for output raster
    outDataset = driver.Create(
        outMap + fmt,
        x_pixels,
        y_pixels,
        bands,
        dtype)

    outDataset.SetGeoTransform((
        x_min,  # 0
        PIXEL_SIZE,  # 1
        0,  # 2
        y_max,  # 3
        0,  # 4
        -PIXEL_SIZE))

    outDataset.SetProjection(projection)

    return outDataset


def classify_pixel_bloc(model, inputImage, bands, outMap, blocksize=None,
                        FMT=None, ndvi=None, dtype=gdal.GDT_Int32):
    """
    A block processing classifier for large rasters, supports KEA, HFA, & Gtiff
    formats. KEA is recommended, Gtiff is the default

    Parameters
    ------------------

    model : sklearn model
            a path to a scikit learn model that has been saved

    inputImage : string
                 path to image including the file fmt 'Myimage.tif'

    bands : band
            the no of image bands eg 8

    outMap : string
             path to output image excluding the file format 'pathto/mymap'

    FMT : string
          optional parameter - gdal readable fmt

    blocksize : int (optional)
                size of raster chunck in pixels 256 tends to be quickest
                if you put None it will read size from gdal (this doesn't always pay off!)

    dtype : int (optional - gdal syntax gdal.GDT_Int32)
            a gdal dataype - default is int32


    Notes
    -------------------------------------------------

    Block processing is sequential, but quite a few sklearn models are parallel
    so that has been prioritised rather than raster IO
    """
    if FMT == None:
        FMT = 'Gtiff'
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'

    inDataset = gdal.Open(inputImage)

    outDataset = _copy_dataset_config(inDataset, outMap=outMap,
                                      bands=bands)
    band = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize
    outBand = outDataset.GetRasterBand(1)
    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = band.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize

    # For either option below making a block index is FAR slower than the
    # code used below - don't be tempted - likely cython is the solution to
    # performance gain here or para processing (but the model is already multi-core)

    # TODO 1- find an efficient way of classifying only non-zero values
    # issue is extracting them, then writing back to the output array
    # e.g
    # [[02456812000002567], ]02456812000002567], ]02456812000002567]]
    # predict [[24568122567, 24568122567, 24568122567]]
    # then write back to original positions
    # scipy sparse doesn't seem to work....
    # TODO 2- thread or parallelise block/line processing
    # Pressumably writing to different parts of raster should be ok....

    model1 = joblib.load(model)
    if blocksizeY == 1:
        rows = np.arange(cols, dtype=np.int)
        for row in tqdm(rows):
            i = int(row)
            j = 0
            # X = np.zeros(shape = (bands, blocksizeX))
            # for band in range(1,bands+1):

            X = inDataset.ReadAsArray(j, i, xsize=blocksizeX, ysize=blocksizeY)
            X.shape = ((bands, blocksizeX))

            if X.max() == 0:
                predictClass = np.zeros_like(rows, dtype=np.int32)
            else:
                X = np.where(np.isfinite(X), X, 0)  # this is a slower line
                X = X.transpose()
                # Xs = csr_matrix(X)
                predictClass = model1.predict(X)
                predictClass[X[:, 0] == 0] = 0

            outBand.WriteArray(predictClass.reshape(1, blocksizeX), j, i)
            # print(i,j)

    # else it is a block
    else:
        for i in tqdm(range(0, rows, blocksizeY)):
            if i + blocksizeY < rows:
                numRows = blocksizeY
            else:
                numRows = rows - i

            for j in range(0, cols, blocksizeX):
                if j + blocksizeX < cols:
                    numCols = blocksizeX
                else:
                    numCols = cols - j

                X = inDataset.ReadAsArray(j, i, xsize=numCols, ysize=numRows)
                #                X = np.zeros(shape = (bands, numCols*numRows))
                #                for band in range(1,bands+1):
                #                    band1 = inDataset.GetRasterBand(band)
                #                    data = band1.ReadAsArray(j, i, numCols, numRows)
                if X.max() == 0:
                    continue
                else:
                    X.shape = ((bands, numRows * numCols))
                    X = X.transpose()
                    X = np.where(np.isfinite(X), X, 0)
                    # this is a slower line
                    # Xs= csr_matrix(X)

                    # YUCK!!!!!! This is a repulsive solution
                    if ndvi != None:
                        ndvi1 = (X[:, 3] - X[:, 2]) / (X[:, 3] + X[:, 2])
                        ndvi1.shape = (len(ndvi1), 1)
                        ndvi1 = np.where(np.isfinite(ndvi1), ndvi1, 0)
                        ndvi2 = (X[:, 7] - X[:, 6]) / (X[:, 7] + X[:, 6])
                        ndvi2.shape = (len(ndvi2), 1)
                        ndvi2 = np.where(np.isfinite(ndvi2), ndvi2, 0)

                        X = np.hstack((X[:, 0:4], ndvi1, X[:, 4:8], ndvi2))

                    predictClass = model1.predict(X)
                    predictClass[X[:, 0] == 0] = 0
                    predictClass = np.reshape(predictClass, (numRows, numCols))
                    outBand.WriteArray(predictClass, j, i)
                # print(i,j)
    outDataset.FlushCache()
    outDataset = None
