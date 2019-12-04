from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC

import PYEO_model
from sklearn import preprocessing
import general_functions
import pandas as pd
import numpy as np
import os
# def do_TPOT():
#
#    if not does_model_exist:
#        features, classes = get_training_data(training_image, training_shape, attribute="NUMBER")
#        print(features.shape)
#        print(classes.shape)
#
#        X_train, X_test, y_train, y_test = train_test_split(features.astype(np.float32),
#                                                            classes.astype(np.float32), train_size=0.75,
#                                                            test_size=0.25)
#
#        #model = TPOTClassifier(generations=10, population_size=20, verbosity=2,n_jobs=-1)
#        model = TPOTClassifier(generations=100, population_size=100, verbosity=2, n_jobs = -1) #suggested
#        model.fit(features, classes)
#        print(model.score(X_test, y_test))
#        model.export(model_out_path)

def train_SVM(csv):
    band_labels = ["Id","ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
                   "s2_r", "s2_nir", "hv", "vv","s2_20m_1","s2_20m_2","s2_20m_3","s2_20m_4","s2_20m_5","s2_20m_6","seg"]

    class_labels = ["Closed canopy forest", "Open canopy forest", "Full sun cocoa",
                    "Agroforestry cocoa", "Agricultural lands", "Oil-palm",
                    "Built-up/bare"]

    training_data = pd.read_csv(csv)

    training_data.columns = band_labels
    training_data.to_csv(csv[:-5]+'3.csv', index= False)


 #   print(training_data)

  #  print(str(training_data[['b3_all_19b']].mean()))

    #training_data.replace(-9999,np.nan)

    #training_data.dropna(axis = 0)
    #print(str(training_data[['b3_all_19b']].mean()))

  #   def sample_class(class_df,n_samples_per_class):
  #       return class_df.sample(n_samples_per_class)
  #
  #
  #
  #
  # #  sample_data = training_data.groupby('class').apply(sample_class).reset_index(1)
  #  # sample_data = training_data.groupby('Id')
  #
  #   labels = training_data['Id']
  #   features = (training_data.loc[:,"b1_all_19b":])
  #
  #   print(labels)
  #   print(features)
  #
  #   f_train, f_test, l_train, l_test = train_test_split(features, labels, train_size=0.7)
  #
  #
  #   # grid search for the best parameters for SVM
  #   # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
  #   C_range = np.logspace(0, 128, 4, base=10)  # base = 2 for a fine tuning
  #   gamma_range = np.logspace(0, 1, 5, base=10)
  #   param_grid = dict(gamma=gamma_range, C=C_range)
  #   cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
  #   grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=6)
  #   grid.fit(f_train, l_train)
  #   #
  #   print("The best parameters are %s with a score of %0.2f"
  #          % (grid.best_params_, grid.best_score_))
  #   # c, gamma = grid.best_params_
  #   #  c=
  #   #  gamma =
  #   # model = svm.SVC(kernel='rbf')
  #   # #    model = TPOTClassifier(generations=10, population_size=20, verbosity=2,n_jobs=-1)
  #   # model.fit(features, classes)
  #   # print(model.score(X_test, y_test))
  #   # model.export(model_out_path)

def cleaning_training_points():
    training_points_shp ="/media/ubuntu/Data/Ghana/cocoa_upscale_test/shp/Training points_clip.shp"
    general_functions.clean_shp(shape_path=training_points_shp, id_field = 'Id', name_field="Class")



def do_gridSearch(training_shp, training_image, model_out_path, bands ):
    PYEO_model.train_cairan_model(image_path = training_image,shp_path=training_shp, outModel_path = model_out_path,
                                  bands =bands ,attribute = 'Id', shape_projection_id = 32630)

def test_do_grid_search():
    training_shp = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/shp/field_data_clip.shp"
    training_image = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_19bands_stack.tif"
    out_model_path = training_image[:-4] +'.pkl'
    bands = 19

    do_gridSearch(training_shp = training_shp, training_image=training_image, model_out_path = out_model_path, bands= bands)

def test_do_classify_image():
    image_to_classify = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_19bands_stack.tif"
    out_image_path = image_to_classify[:-4] + '_classified_v2.tif'
    model_path = image_to_classify[:-4] + '.pkl'

    PYEO_model.classify_image(model_path=model_path,in_image_path=image_to_classify,out_image_path=out_image_path)


def train_RS():
    george_points_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/shp/Training_points_clip_dropTransition_add.tif"
    # training_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_19bands_stack.tif"
    # outmodel = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/george_data_19bands_add.pkl"
    # training_summary = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/george_data_19bands_add.csv"
    # band_labels = ["ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
    #                "s2_r", "s2_nir", "hv", "vv","s2_20m_1","s2_20m_2","s2_20m_3","s2_20m_4","s2_20m_5","s2_20m_6","seg"]
    #bands = 19

    training_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_13bands_stack_v2.tif"
    outmodel = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/george_data_13bands_add_255_eRF_v2.pkl"
    training_summary = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/george_data_13bands_add_255_v2.csv"
    band_labels = ["ndvi", "ci", "psri", "gndvi", "s2_rep", "ireci", "s2_b", "s2_g",
                   "s2_r", "s2_nir", "hv", "vv","seg"]
    bands = 13

    features, labels = PYEO_model.get_training_data_tif(image_path=training_tif,training_tif_path=george_points_tif)

    features_df = pd.DataFrame(data= features,columns= band_labels).astype(np.int32)

    scale = preprocessing.MinMaxScaler(feature_range=(0,255)).fit(features)
    features_trans =  scale.transform(features)
    features_trans_df = pd.DataFrame(features_trans, index=features_df.index, columns=features_df.columns)

    features_df.describe()
    features_trans_df.describe()

    labels_df = pd.DataFrame(data= labels, columns= ['class_name']).astype(np.int)

    features_df["class_name"] = labels_df["class_name"]

    features_df.describe().to_csv(training_summary)

    #PYEO_model.train_rs_simple(X_train_input=features_trans,y_train_input=labels,outModel_path=outmodel, bands = bands)
    PYEO_model.train_eRF_gs_simple(X_train_input=features_trans, y_train_input=labels, outModel_path=outmodel, bands=bands)
    image_tobe_classified = training_tif

    PYEO_model.classify_image(model_path=outmodel,
                      in_image_path= image_tobe_classified,
                      out_image_path=image_tobe_classified[:-4] +"_classified_13bands_255_eRF2_v2.tif",
                      rescale_predict_image = scale)
    image_tobe_classified = "/media/ubuntu/Data/Ghana/north_region/s2_NWN/images/stacked/with_s1_seg/composite_20180122T102321_T30NWN.tif"

    PYEO_model.classify_image(model_path=outmodel,
                              in_image_path=image_tobe_classified,
                              out_image_path=image_tobe_classified[:-4] + "_classified_13bands_255_eRF2_v2.tif",
                              rescale_predict_image=scale)


def test_histmatching_classify():
    training_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_13bands_stack_v2.tif"
    george_points_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/shp/Training_points_clip_dropTransition_add.tif"
    input_model = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/george_data_13bands_add_255_eRF_v2.pkl"
    image_tobe_classified = "/media/ubuntu/Data/Ghana/north_region/s2/images/stacked/with_s1_seg/composite_20180122T102321_T30NWN.tif"

    features, labels = PYEO_model.get_training_data_tif(image_path=training_tif, training_tif_path=george_points_tif)

    scale = preprocessing.MinMaxScaler(feature_range=(0, 255)).fit(features)

    PYEO_model.classify_image(model_path=input_model,
                              in_image_path=image_tobe_classified,
                              out_image_path=image_tobe_classified[:-4] + "_classified_13bands_255_eRF2_v4.tif",
                              rescale_predict_image=scale,
                              ref_img_for_linear_shift=training_tif,
                              generate_mask=False)

def do_classify(working_dir):
    training_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/all_13bands_stack_v2.tif"
    george_points_tif = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/shp/Training_points_clip_dropTransition_add.tif"
    input_model = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/george_data_13bands_add_255_eRF_v2.pkl"

    features, labels = PYEO_model.get_training_data_tif(image_path=training_tif, training_tif_path=george_points_tif)
    scale = preprocessing.MinMaxScaler(feature_range=(0, 255)).fit(features)

    os.chdir(working_dir)

    for image in os.listdir("images/stacked/with_s1_seg"):
        if image.endswith(".tif"):
            image_tobe_classified = os.path.join("images/stacked/with_s1_seg",image)
            out_image_path = os.path.join("output/", image[:-4]+"_13bands_255_eRF.tif")

            PYEO_model.classify_image(model_path=input_model,
                                      in_image_path=image_tobe_classified,
                                      out_image_path=out_image_path,
                                      rescale_predict_image=scale,
                                      ref_img_for_linear_shift=training_tif,
                                      generate_mask= False)


if __name__ == "__main__":
   # cleaning_training_points()
   # test_do_grid_search()
    #test_do_classify_image()
  # train_SVM(csv = "/media/ubuntu/Data/Ghana/cocoa_upscale_test/Training_points_clip_dropTransition_point_clean2.csv")
   #train_RS()
   #test_histmatching_classify()
   #do_classify(working_dir="/media/ubuntu/Data/Ghana/north_region/s2/")
   #do_classify(working_dir="/media/ubuntu/Data/Ghana/cocoa_big/s2_batch2/")
   #do_classify(working_dir="/media/ubuntu/Data/Ghana/cocoa_big/s2/")
   #test_histmatching_classify()

   #general_functions.do_mask(working_dir="/media/ubuntu/Data/Ghana/cocoa_big/s2_batch2/")
   #general_functions.do_mask(working_dir="/media/ubuntu/Data/Ghana/cocoa_big/s2/")
   general_functions.do_mask(working_dir="/media/ubuntu/Data/Ghana/north_region/s2/", generate_mask=False)








