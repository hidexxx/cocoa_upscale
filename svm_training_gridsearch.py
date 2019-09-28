import sys

from sklearn import svm
from sklearn.svm import SVC

sys.path.append(r"/home/ubuntu/Documents/Code/pyeo")
from pyeo.classification import get_training_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
import numpy as np


def do_grid_search(training_shape, training_image, image_out_path, model_out_path, ):

    features, classes = get_training_data(training_image, training_shape, attribute="Id")
    print(features.shape)
    print(classes.shape)

    X_train, X_test, y_train, y_test = train_test_split(features.astype(np.uint8),
                                                        classes.astype(np.uint8), train_size=0.7,
                                                        test_size=0.3)

    # grid search for the best parameters for SVM
    # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    # C_range = np.logspace(0, 1, 2, base=10)  # base = 2 for a fine tuning
    # gamma_range = np.logspace(0, 128, 2, base=10)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=6)
    # grid.fit(X_train, y_train)
    #
    # print("The best parameters are %s with a score of %0.2f"
    #       % (grid.best_params_, grid.best_score_))
    # c, gamma = grid.best_params_
  #  c=
  #  gamma =
    model = svm.SVC(kernel='rbf')
#    model = TPOTClassifier(generations=10, population_size=20, verbosity=2,n_jobs=-1)
    model.fit(features, classes)
    print(model.score(X_test, y_test))
    model.export(model_out_path)

def do_classify(model_path,image_in_path,image_out_path):

    print('here')


    # s2simp_veg_s1_bands_255 = s2simp_veg_s1_bands_out[:-4] +'_255.tif'
    # training_image_255 = scale_to_255(intif=s2simp_veg_s1_bands_out,outtif= s2simp_veg_s1_bands_255)
    # training_image = s2simp_veg_s1_bands_255
   # image_to_classify = training_image


def test_do_grid_search():
    training_shape = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/shp/field_data_clip.shp"
    training_image = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_vegIndex_s1.tif"
    out_image_path = "output/ghana_classified_SVM.tif"
    out_model_path = "output/ghana_svm_grid.pkl"

    do_grid_search(training_shape, training_image, out_image_path, out_model_path)


if __name__ == "__main__":
    test_do_grid_search()