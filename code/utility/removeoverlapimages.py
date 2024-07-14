import os
import shutil

def remove_overlapping_files(source_dir, validation_dir):
    """
    Remove files in the validation directory that also exist in the source directory.
    
    Args:
    - source_dir (str): Directory where the training files are located.
    - validation_dir (str): Directory where the validation files are located.
    """
    # List all files in both directories
    source_files = set(os.listdir(source_dir))
    validation_files = set(os.listdir(validation_dir))

    # Find overlapping files
    overlapping_files = source_files.intersection(validation_files)

    # Delete overlapping files from the validation directory
    for file_name in overlapping_files:
        file_path = os.path.join(validation_dir, file_name)
        try:
            os.remove(file_path)
            print(f"Deleted {file_name} from validation set.")
        except OSError as e:
            print(f"Error deleting {file_name}: {e}")

# Define the directories for images and masks
training_images_dir = os.path.join('..', 'data', 'images')
validation_images_dir = os.path.join('..',  'validation_data', 'images')
training_masks_dir = os.path.join('..', 'data', 'masks')
validation_masks_dir = os.path.join('..', 'validation_data', 'masks')

print(training_images_dir)
print(validation_images_dir)
# Remove overlapping images
remove_overlapping_files(training_images_dir, validation_images_dir)

# Remove overlapping masks
remove_overlapping_files(training_masks_dir, validation_masks_dir)
