import PYEO_model

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

def do_gridSearch(training_shp, training_image, model_out_path, ):
    PYEO_model.train_cairan_model(image_path = training_image,shp_path=training_shp, outModel_path = model_out_path,
                                  bands =12 ,attribute = 'Id', shape_projection_id = 32630)
def do_classify_image(model_path,in_image_path,out_image_path):
    model = PYEO_model.load_model(model_path)
    PYEO_model.classify_image(in_image_path=in_image_path,out_image_path=out_image_path,model=model)

def test_do_grid_search():
    training_shape = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/shp/field_data_clip.shp"
    training_image = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/s2/s2_20180219_testsite_vegIndex_s1.tif"
    out_image_path = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/ghana_classified_rs.tif"
    out_model_path = "/media/ubuntu/storage/Ghana/cocoa_upscale_test/ghana_gridsearch.pkl"

    do_gridSearch(training_shp = training_shape, training_image=training_image, model_out_path = out_model_path)
    do_classify_image(model_path=out_model_path,in_image_path=training_image,out_model_path=out_image_path)


if __name__ == "__main__":
    test_do_grid_search()




