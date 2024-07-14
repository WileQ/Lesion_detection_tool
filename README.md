# final_product
03_evaluate_classifier.py contains the code needed to run the classify on 
a single image. As long as the image and mask is the same size and loaded 
with the plt.imread command, the function will work correctly. It 
returns the predicted label, where True is cancer and False is non-cancer. 
It also returns an 1x2 array of probabilities. The first probability is 
the probability of being non-cancer and the second probability is the 
probability of cancer. Threshold is set to 0.5. Extract features will 
return 0.5 for all features if an error occurs, but that has not yet 
happened with the function being run on over 2000 images.

trainingtest2.py is for the boxplots in the report.
evaluate_metrics.py is for the evaluation metrics for the classifier seen 
in the final report.

01_process_images.py extracts features for the training images located in 
training_data. The features are saved in features/features.csv

process_test.py functions identically to 01_process_images.py except it 
extracts feature for the images in the test set folder.  The features are 
saved in features_test.csv.

The classifier is saved in groupXY_classifier.sav, and uses a k value of 
9 as described in the report.
