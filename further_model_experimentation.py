import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyeo import classification as cls
from pyeo.filesystem_utilities import init_log
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from copy import copy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

init_log("experimentation.log")

band_labels = ["ndvi","ci","psri","gndvi","s2_rep","ireci","s2_b", "s2_g",
 "s2_r", "s2_nir", "hv", "vv"]

class_labels = ["Closed canopy forest", "Open canopy forest", "Full sun cocoa",
                "Agroforestry cocoa", "Agricultural lands", "Oil-palm",
                "Built-up/bare"]

training_data = pd.read_csv("data/training_sigs_12_bands.csv", names=
                           ["class"]+band_labels).astype(np.uint32)

def sample_class(class_df):
    return class_df.sample(150)

sample_data = training_data.groupby('class').apply(sample_class).reset_index(1)

#model = make_pipeline(
#    make_union(
#        StackingEstimator(estimator=GaussianNB()),
#        FunctionTransformer(copy)
#    ),
#    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.4, min_samples_leaf=3, min_samples_split=2, n_estimators=100)
#)


# Average CV score on the training set was:0.6910391809871432
model = make_pipeline(
    ZeroCount(),
    RobustScaler(),
    StackingEstimator(estimator=LogisticRegression(C=15.0, dual=True, penalty="l2")),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.9500000000000001, min_samples_leaf=7, min_samples_split=3, n_estimators=100)
)


#model = SVC(kernel='rbf', gamma=0.0769, C=1)

# Removing class 5 - it's swamping classification. Also removing segmentation
#training_data = training_data[training_data['class'] != 5]
labels = sample_data['class']
features = (sample_data.loc[:,'ndvi':]
    .fillna(0)
    .astype(np.uint32)
    )

f_train, f_test, l_train, l_test = train_test_split(features, labels, train_size = 0.8)

experiment_name = "rf_object_150_samples"

fitted_model = model.fit(f_train, l_train)
print("\n{} Score: {}\n".format(experiment_name, model.score(f_test, l_test)))
image_path = "data/s2_20180219_testsite_vegIndex_s1_clipped.tif"
output_path = "outputs/{}.tif".format(experiment_name)
model_path = "models/{}.pkl".format(experiment_name)

joblib.dump(fitted_model, model_path)
cls.classify_image(image_path, model_path, output_path, apply_mask=False)

for_confusion = fitted_model.predict(f_test)
cm = confusion_matrix(l_test, for_confusion)
print(cm)
