Part of the code for data extraction from MIMIC III is adapted from Dr Niranjani Prasad at https://github.com/bee-hive/VentRL/tree/master/utils. After building the time Frame of the included patients on an hour basis, blood pressure and SBT are then added in add_bloodpre.ipynb and add_sbt.ipynb. Then the data is processed in data_process.ipynb to generate the train, validation and test dataset for this work. Extubation_failure cohort is generated in extubaition_failure_data_process.ipynb.
PCA_visu gives a PCA visuliation of the data. The CNN model training is in CNN_model.ipynb. The final CNN model is saved in the folder final_model. 
The feature importance along with the counterfactual examples are produced in feature_explanation.ipynb using the final model trained in the previous step. 
A Dnn_model is also trained in dnn_methods.ipynb. It is used in the explanability paper for comparison with the CNN model. Feature_importance_dnn.ipynb compared the dnn model with the cnn model in terms of the feature importance in the test dataset. 
Other methods in the paper for comparision purpose are provided in other_methods_inpaper.ipynb. 
