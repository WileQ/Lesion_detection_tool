#This script will hopefully segment dots/globules, and give them a score on a countinuous scale from 0 to 1. 
#This is a part of a larger evaluation algorithm that defines whether the skin lesion presented is malignant.

#Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt



def dots_and_globules(image, mask, threshold = 0.1):

    image = image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)


    # Load the mask as grayscale
    mask_og = mask

    if mask_og.dtype != np.uint8:
        mask_og = (mask_og * 255).astype(np.uint8)



    #DoG - to detect edges in the picture
    def DoG(img):

        # Apply 9x9 and 11x11 Gaussian blur
        low_sigma = cv2.GaussianBlur(img,(9,9),0)
        high_sigma = cv2.GaussianBlur(img,(11,11),0)
        
        # Calculate the DoG by subtracting
        dog = low_sigma - high_sigma
        return dog


    #Detecting edges
    image = DoG(image)

    #Setting a threshold for creating a mask
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    threshold = 245
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _ ,mask = cv2.threshold(image, 215, 255, 0)

    # Apply the original mask to the generated mask
    image = cv2.bitwise_and(image, image, mask=mask)

    kernel = np.ones((3, 3), np.uint8)  # Adjust the kernel size as needed

    # Apply dilation to connect parts of hairs or close small gaps
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Erosion to potentially disconnect hairs
    mask_eroded = cv2.erode(mask, kernel, iterations=2)

    # Find contours on the eroded mask
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate perimeter and area for each contour
    perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
    areas = [cv2.contourArea(cnt) for cnt in contours]


    # Compute compactness for each contour
    compactness_values = [(perimeter ** 2) / area if area != 0 else 0 for perimeter, area in zip(perimeters, areas)]

    # Define a threshold for compactness
    compactness_threshold = 0.80  # Adjust as needed

    # Filter contours based on compactness
    filtered_contours = [cnt for cnt, compactness in zip(contours, compactness_values) if compactness > compactness_threshold]

    # Create a new mask with the filtered contours
    mask_filtered = np.zeros_like(mask)
    cv2.drawContours(mask_filtered, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Fill the mask fully
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Detect edges in the filled mask
    edges_filled_mask = cv2.Canny(filled_mask, 30, 150)

    # Subtract the detected edges from the original mask
    mask_without_edges = cv2.bitwise_and(mask, cv2.bitwise_not(edges_filled_mask))
    mask_without_edges = cv2.bitwise_and(image, image, mask=mask_og)

    return np.sum(mask_without_edges)/np.sum(mask_og)
      
