# THIS DOCUMENT IS FOR GENERATING THE SET OF BOXPLOTS FOR CROSS-VALIDATION PERFORMANCE AS SEEN IN THE REPORT.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import pickle

# Where are the files
file_data = '..' + os.sep + 'training_data' + os.sep + 'metadata.csv'
df = pd.read_csv(file_data)

images_folder = '..' + os.sep + 'training_data' + os.sep + 'images'

# Load metadata
image_files = os.listdir(images_folder)
image_files = set(image_files)  # Convert to a set for faster membership testing

# Filter DataFrame based on the existence of image files
mask = df['img_id'].apply(lambda img_id: img_id in image_files)
df_filtered = df[mask]

label = np.array(df_filtered['diagnostic'])

# Where did we store the features?
file_features = 'features/features.csv'
feature_names = ['Color variability', 'Streaks', 'asymmetry']
df_features = pd.read_csv(file_features)

# Exclude rows where all feature values are zero
mask_all_zeros = (df_features == 0).all(axis=1)
df_features = df_features[~mask_all_zeros]

x = np.array(df_features[feature_names])
y = np.isin(label, ['BCC', 'SCC', 'MEL'])
patient_id = df_filtered['patient_id']

# Prepare cross-validation - images from the same patient must always stay together
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)

# Define classifiers for all odd k from 1 to 15
classifiers = [KNeighborsClassifier(k) for k in range(1, 16, 2)]
num_classifiers = len(classifiers)
acc_val = np.empty([num_folds, num_classifiers])

for i, (train_index, val_index) in enumerate(group_kfold.split(x, y, patient_id)):
    x_train, y_train = x[train_index], y[train_index]
    x_val, y_val = x[val_index], y[val_index]
    
    for j, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)
        acc_val[i, j] = accuracy_score(y_val, clf.predict(x_val))

# Plot boxplots for each classifier
plt.figure(figsize=(10, 6))
plt.boxplot(acc_val, labels=[f'k={k}' for k in range(1, 16, 2)])
plt.title('Boxplot of Classifier Accuracies for KNN with Different k Values')
plt.ylabel('Accuracy')
plt.xlabel('Classifier (k Value)')
plt.grid(True)
plt.show()

