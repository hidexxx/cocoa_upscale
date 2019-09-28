if __name__ == "__main__":

 # ###############################################################################
 #    image_to_classify = "/media/ubuntu/storage/Ghana/s2_ALOSstat_ini.tif"
 #    training_image = image_to_classify
 #    training_shape = "//media/ubuntu/storage/Ghana/field_data/cocoa_filed_data_clean_v2/merged_V2_UTM.shp"
 #
 #    out_path = image_to_classify[:-4]+'_classified.tif'
 #
 #    model_out_path = "/media/ubuntu/storage/Ghana/ALOS_s2_model_V1_TPOT.py"
 #
 #    does_model_exist = True
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
 #        model = TPOTClassifier(generations=5, population_size=20, verbosity=2,n_jobs=-1)  # suggested;
 #       # model = TPOTClassifier(generations=100, population_size=100, verbosity=2) #suggested
 #        model.fit(features, classes)
 #        print(model.score(X_test, y_test))
 #        model.export(model_out_path)
 #
 #
 #       # save_model(model, model_out_path)
 #    # else:
 #    #     model = load_model(model_out_path)
 #    print('here.. classify image!')
 #
 #    features, classes = get_training_data(training_image, training_shape, attribute="NUMBER")
 #    exported_pipeline = make_pipeline(
 #        StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.55,
 #                                                         min_samples_leaf=4, min_samples_split=15, n_estimators=100)),
 #        LogisticRegression(C=25.0, dual=False, penalty="l2")
 #    ) # 0.82
 #
 #    exported_pipeline.fit(features, classes)
 #   # results = exported_pipeline.predict(testing_features)
 #
 #    classify_image(image_to_classify, exported_pipeline, out_path,num_chunks=10)
 #
#####################################################
# ###############################################################################
   #image_to_classify = "/media/ubuntu/storage/Ghana/Planet/Planet_cocoa3_outtif.tif"
   image_to_classify = "/media/ubuntu/storage/Ghana/Planet/cocoa_planet_merge04.tif"
   training_image = image_to_classify
   training_shape = "/media/ubuntu/storage/Ghana/field_data/cocoa_filed_data_clean_v2/merged_V2_UTM.shp"

   out_path = image_to_classify[:-4]+'_classified_TOP_smooth.tif'

   model_out_path = "/media/ubuntu/storage/Ghana/Planet_model_V2_TPOT.py"

   does_model_exist = False

   if not does_model_exist:
       features, classes = get_training_data(training_image, training_shape, attribute="NUMBER")
       print(features.shape)
       print(classes.shape)

       X_train, X_test, y_train, y_test = train_test_split(features.astype(np.float32),
                                                           classes.astype(np.float32), train_size=0.75,
                                                           test_size=0.25)

       #model = TPOTClassifier(generations=10, population_size=20, verbosity=2,n_jobs=-1)
       model = TPOTClassifier(generations=100, population_size=100, verbosity=2, n_jobs = -1) #suggested
       model.fit(features, classes)
       print(model.score(X_test, y_test))
       model.export(model_out_path)


      # save_model(model, model_out_path)
   # else:
   #     model = load_model(model_out_path)
   print('here.. classify image!')

   features, classes = get_training_data(training_image, training_shape, attribute="NUMBER")
#Best pipeline: ExtraTreesClassifier(VarianceThreshold(BernoulliNB(input_matrix, alpha=100.0, fit_prior=True), threshold=0.05), bootstrap=True, criterion=gini, max_features=0.15000000000000002, min_samples_leaf=9, min_samples_split=15, n_estimators=100)
#0.7925977762226888

   exported_pipeline = make_pipeline(
        StackingEstimator(estimator=BernoulliNB(alpha=100.0, fit_prior=True)),
        VarianceThreshold(threshold=0.05),
        ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.15000000000000002, min_samples_leaf=9,
                             min_samples_split=15, n_estimators=100)
   )


 # 0.79

   exported_pipeline.fit(features, classes)
  # results = exported_pipeline.predict(testing_features)

   classify_image(image_to_classify, exported_pipeline, out_path,num_chunks=10)
