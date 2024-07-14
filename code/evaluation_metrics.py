# THIS DOCUMENT IS FOR THE EVALUATION METRICS FOR THE CLASSIFIER IN THE FINAL REPORT

import os
import pandas as pd
import numpy as np
import pickle

# Default packages for the minimum example
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



import pickle #for saving/loading trained classifiers


#Where are the files
file_data = '..' + os.sep + 'training_data' + os.sep +'metadata.csv'
df = pd.read_csv(file_data)

images_folder = '..' + os.sep + 'test' + os.sep + 'images'
masks_folder = '..' + os.sep + 'test' + os.sep + 'masks'

# Load metadata
df = pd.read_csv(file_data)

#Where did we store the features?
file_features = 'features/features_test.csv'
feature_names = ['Color variability', 'Streaks', 'asymmetry']

#The following is to make sure we only get the images for which we actually have measured features.
# List all files in the images directory
image_files = os.listdir(images_folder)
image_files = set(image_files)  

mask_files = os.listdir(masks_folder)
mask_files = set(mask_files)  

# Create a mask where each element is True if the corresponding image file exists
mask = df['img_id'].apply(lambda img_id: img_id in image_files)

# Filter the DataFrame based on the existence of image files
df_filtered = df[mask]


df_features = pd.read_csv(file_features)

label = np.array(df_filtered['diagnostic'])

# Create a mask where each row has all zero values
mask_all_zeros = (df_features == 0).all(axis=1)

# Use the mask to select rows where not all values are zero
df_features = df_features[~mask_all_zeros]

# Make the dataset, you can select different classes (see task 0)
x = np.array(df_features[feature_names])
y = np.isin(label, ['BCC', 'SCC', 'MEL'])

classifier = pickle.load(open('groupXY_classifier.sav', 'rb'))

predictions = classifier.predict(x)

conf_matrix = confusion_matrix(y, predictions)
print("Confusion Matrix on Test Set:")
print(conf_matrix)

# Calculate and print the accuracy
accuracy = accuracy_score(y, predictions)
print("Overall Accuracy on Test Set:", accuracy)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix on Test Set')
plt.show()


report = classification_report(y, predictions, target_names=["non-cancer", "cancer"])
print("Classification Report:")
print(report)
