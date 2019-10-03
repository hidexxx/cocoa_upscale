import pandas as pd
import numpy as np
from pyeo import classification as cls
from pyeo.filesystem_utilities import init_log
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from copy import copy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

init_log("experimentation.log")

n_samples_per_class = 150

band_labels = ["ndvi","ci","psri","gndvi","s2_rep","ireci","s2_b", "s2_g",
 "s2_r", "s2_nir", "hv", "vv"]

class_labels = ["Closed canopy forest", "Open canopy forest", "Full sun cocoa",
                "Agroforestry cocoa", "Agricultural lands", "Oil-palm",
                "Built-up/bare"]

training_data = pd.read_csv("data/training_sigs_12_bands.csv", names=
                           ["class"]+band_labels).astype(np.uint32)

def sample_class(class_df):
    return class_df.sample(n_samples_per_class)

sample_data = training_data.groupby('class').apply(sample_class).reset_index(1)

labels = sample_data['class']
features = (sample_data.loc[:,'ndvi':]
    .fillna(0)
    .astype(np.uint32)
    )

f_train, f_test, l_train, l_test = train_test_split(features, labels, train_size = 0.8)

model = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        PCA(iterated_power=9, svd_solver="randomized")
    ),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.2, min_samples_leaf=1, min_samples_split=8, n_estimators=100)
)

experiment_name = "most_consistent_model"

image_path = "data/s2_20180219_testsite_vegIndex_s1_clipped.tif"
output_path = "outputs/{}.tif".format(experiment_name)
model_path = "models/{}.pkl".format(experiment_name)

fitted_model = model.fit(f_train, l_train)
print("\n{} Score: {}\n".format(experiment_name, model.score(f_test, l_test)))

joblib.dump(fitted_model, model_path)
cls.classify_image(image_path, model_path, output_path, apply_mask=False)

for_confusion = fitted_model.predict(f_test)
cm = confusion_matrix(l_test, for_confusion)
print(cm)
