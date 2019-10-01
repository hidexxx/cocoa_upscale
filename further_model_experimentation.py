import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyeo import classification as cls
from pyeo.filesystem_utilities import init_log
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

init_log("experimentation.log")

band_labels = ["band_{}".format(number) for number in range(1,13)]

training_data = pd.read_csv("data/training_sigs_12_bands.csv", names=
                           ["class"]+band_labels)


# Average CV score on the training set was:0.8129780700079303
#model =  = make_pipeline(
#    make_union(
#        StackingEstimator(estimator=GaussianNB()),
#        FunctionTransformer(copy)
#    ),
#    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.4, min_samples_leaf=3, min_samples_split=2, n_estimators=100)
#)

model = SVC(kernel='rbf', gamma=0.0769, C=512)

# Removing class 5 - it's swamping classification. Also removing segmentation
#training_data = training_data[training_data['class'] != 5]
labels = training_data['class']
features = (training_data.loc[:,'band_1':]
    .fillna(0)
    .astype(np.uint32)
    )

fitted_model = model.fit(features, labels)
joblib.dump(fitted_model, "svc_no_builtup.pkl")

image_path = "data/s2_20180219_testsite_vegIndex_s1_clipped.tif"
output_path = "outputs/svc_no_builtup_test.tif"
model_path = "svc_no_builtup.pkl"
cls.classify_image(image_path, model_path, output_path, apply_mask=False)
