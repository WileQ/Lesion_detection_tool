#imports
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, morphology, measure, transform
from PIL import Image

def detection(image,mask):
    image = image[:, :, :3]
# Convert image to grayscale
    image_gray = color.rgb2gray(image)

# Resize mask to match the dimensions of the grayscale image (used because some images did not have the same dimensions)
    mask_resized = transform.resize(mask, image_gray.shape, anti_aliasing=True)

# Apply mask to the grayscale image
    masked_image = image_gray * mask_resized

# Apply Gaussian blur for noise reduction (does a good job actually)
    smoothed_image = filters.gaussian(masked_image, sigma=1) # sigma = 1 and threshold = 0.3 performed best.

# Use edge detection (Sobel operator) on the smoothed image to detect edges (duh)
    edges = filters.sobel(smoothed_image)

# Threshold edge image (to see what pixels stand out)
    threshold = 0.3  # Adjust threshold as needed (I tried different thresholds and 0.3 with sigma = 1 works best during the tests.)
    binary_edges = edges > threshold
    
# Find contours of the edges in the binary edge image 
    contours = measure.find_contours(binary_edges, 0.5) #Find contours on the image with the set threshold 0.5. Contours will be used to visualize irregular streaks.

# Calculate the percentage of irregular streaks
    image_area = np.prod(image.shape[:2])  # Total number of pixels in the image
    contour_area = sum(len(contour) for contour in contours)  # Total length of all contours
    percentage_irregular_streaks = (contour_area / image_area) * 100
    
# Mark irregular streaks on the original image with blue color
    fig, ax = plt.subplots() #You can delete this line if you don't need to see the image
     #You can delete this line if you don't need to see the image

    irregular_streaks_found = False  # Flag to check if irregular streaks are not found

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='blue')
        irregular_streaks_found = True  # Set flag to True if any contours are found

    plt.close()

# Print 1 if irregular streaks are found where more than 0.2% of pixels are marked as irregular streaks, otherwise print 0
    if irregular_streaks_found and percentage_irregular_streaks > 0.2:
        return 1
    else:
        return 0