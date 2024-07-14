"""
FYP project imaging
"""

import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our own file that has the feature extraction functions
from extract_features import extract_features



#-------------------
# Main script
#-------------------



#Where is the raw data
file_data = '..' + os.sep + 'training_data' + os.sep +'metadata.csv'
path_image = '..' + os.sep + 'test' + os.sep + 'images' 
path_mask = '..' + os.sep + 'test' + os.sep + 'masks'

print(os.getcwd())

print(file_data, path_image, path_mask)
    
  
#Where we will store the features
file_features = 'features/features_test.csv'


#Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data. 
image_id = list(df['img_id'])
label = np.array(df['diagnostic'])

num_images = len(image_id)


#Make array to store features
feature_names = ['Color variability', 'Streaks', 'asymmetry']
num_features = len(feature_names)
features = np.zeros([num_images,num_features], dtype=np.float16)  

count = 0

training_set = []

#Loop through all images (now just 10 for demonstration)
for i in np.arange(num_images):
    
    # Define filenames related to this image
    file_image = path_image + os.sep + image_id[i] 



    mask_filename = image_id[i].replace('.png', '_mask.png')
    mask_image = path_mask + os.sep + mask_filename

    if exists(file_image) and exists(mask_image):
        # Read the image
        im = plt.imread(file_image)
        im = np.float16(im)  
        immask = plt.imread(mask_image)
        training_set.append(image_id[i])
        print(file_image)
        count += 1
    
        # Measure features - this does not do anything useful yet!
        x = extract_features(im, immask)
           
        # Store in the variable we created before
        features[i,:] = x
    
    if not exists(file_image) and exists(mask_image):
        print("ERROR FOUND")
        print(file_image)
       
print("Measured features for ", count, " images and added them to features.csv")
        
#Save the image_id used + features to a file   
df_features = pd.DataFrame(features, columns=feature_names)     
df_features.to_csv(file_features, index=False)  

