import os

# Define paths to the directories
training_images_dir = os.path.join('..', 'data', 'images')
validation_images_dir = os.path.join('..', 'validation_data', 'images')
training_masks_dir = os.path.join('..', 'data', 'masks')
validation_masks_dir = os.path.join('..', 'validation_data', 'masks')

def list_files(directory):
    """ Return a set of file names in the given directory """
    return set(os.listdir(directory))

# Get the lists of files
training_images = list_files(training_images_dir)
validation_images = list_files(validation_images_dir)
training_masks = list_files(training_masks_dir)
validation_masks = list_files(validation_masks_dir)

# Check for duplicates between training and validation datasets
duplicate_images = training_images.intersection(validation_images)
duplicate_masks = training_masks.intersection(validation_masks)

# Output the results
if duplicate_images:
    print(f"Duplicate images found: {duplicate_images}")
else:
    print("No duplicate images found between training and validation datasets.")

if duplicate_masks:
    print(f"Duplicate masks found: {duplicate_masks}")
else:
    print("No duplicate masks found between training and validation datasets.")
