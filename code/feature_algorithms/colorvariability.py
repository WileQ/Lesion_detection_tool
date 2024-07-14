#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:53:20 2023

@author: vech
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import special 

# Import packages for image processing
from skimage import morphology #for measuring things in the masks - Not sure why this is here


#-------------------
# Help functions - THESE ARE THE 3 FUNCTIONS I USE TO CALCULATE COLOR VARIABILITY
#-------------------

def compute_blocks(image, size, h, w):
    #Me trying to make sure as much of the image as possible is being used:
    image_copy = image.copy()
    remainder_h = h % 20
    remainder_w = w % 20
    block_space_available_height = (h - remainder_h) // 20
    block_space_available_width = (w - remainder_w) // 20


    block_array = np.zeros((block_space_available_height, block_space_available_width, 3), dtype=float)
    for y in range(0, block_space_available_height*size, size):
        for x in range(0, block_space_available_width*size, size):
            block = image_copy[y:y+size,x:x+size]
            # Computes the overall mean of the elements in a row element along a column.
            # Then, computes the mean of the elements in a row element along a column from the result. 
            # The result is the overall mean of the values in the three different positions, in the current block.
            mask = (block[:, :, 0] != 0) | (block[:, :, 1] != 0) | (block[:, :, 2] != 0) #Apparantly black is 0, 128, 128 in openCV? - Fixed this so it's 0,0,0.
            filtered_block = block[mask]
            #print("Filter", filtered_block)
            #block = block.reshape(-1,3)
            #print("Before", block)
            
            #The mask turns it back into a 2d array with 400 rows, and 3 columns (L, a, b), assuming no black pixels - the amount of black pixels decides the amount of rows.
            #We do not take the mean of blocks where over 50 percent of pixels are black:
            if filtered_block.shape[0] <= 200:
                continue
            if filtered_block.size > 0:
                mean_lab = np.mean(filtered_block, axis=0)
                block_array[int(y/size)][int(x/size)] = mean_lab
    return block_array


def euclidian_distance(color1, color2):
    distance = np.linalg.norm(color1 - color2)
    return distance



def computeColorDistances(block_matrix):
    # We compute the distances of a pair of blocks.
    # We do what's equivalent to order not mattering, without replacement - order doesn't influence euclidian distance, and we shouldn't compare a block to itself.
    distances = []
    block_2d = block_matrix.reshape(-1,3) #Get all the means in a single row for ease of calculation
    block_2d = block_2d[~(block_2d == 0).all(axis=1)]
    #print("Calculating color variability based on ", special.binom(block_2d.shape[0], 2), "distinct pairs of boxes")
    #Adjusting the step for the boxes as the algorithm can get very slow for large resolution images. Done a bit of testing, and doesn't seem to effect accuracy.
    step = 3
    if special.binom(block_2d.shape[0], 2) <= 50:
        step = 1
    if special.binom(block_2d.shape[0], 2) > 10000:
        step = 10
    if special.binom(block_2d.shape[0], 2) > 100000:
        step = 20
    if special.binom(block_2d.shape[0], 2) > 300000:
        step = 100
    for i in range(block_2d.shape[0]):
        for j in range(i+1, block_2d.shape[0], step):
            distance = euclidian_distance(block_2d[i], block_2d[j])
            distances.append(distance)
    return distances


def computeTotalVariabilityScore(ds, t): 
    distances = np.array(ds)
    significant_distances1 = distances[distances > t]
    if distances.size == 0:
        score = 0
        return score
    score = significant_distances1.size/distances.size
    return score

def colorvariability(image, mask):

    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # convert to grayscale
    gray = mask

    # threshold
    thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)[1]

    # get contours
    result = mask.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours) > 1:
        x,y,w,h = 0,0,0,0
    else:
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            #print("x,y,w,h:",x,y,w,h)   

    # LOAD AND PROCESS IMAGE:
    # Load color image

    # Load binary mask and convert to boolean
    binary_mask = mask  # 0 to read as grayscale
    boolean_mask = binary_mask > 0

    # Apply the mask
    segmented_image = image.copy()
    if segmented_image.dtype != np.uint8:
        segmented_image = (segmented_image * 255).astype(np.uint8)
    segmented_image[~boolean_mask] = 0  # Set non-masked pixels to black

    image_processed = cv2.medianBlur(segmented_image, 9)

    #Conversion based on this thread: https://stackoverflow.com/questions/26992476/cv-color-bgr2lab-gives-wrong-range TO GET PROPER LAB RANGES!
    image = image_processed.astype("float32") / 255

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    if x == 0 and y == 0 and w == 0 and h == 0:
        cropped_image_processed = lab_image
    else:
        cropped_image_processed = lab_image[y:y+h, x:x+w] #We crop the image based on the bounding rectangle.

    block_size = 20 #20x20 blocks of pixels

    threshold = 10 #Can be changed freely. Might need adjustment but seems to work alright for now, given how much the image quality varies.

    # Image dimensions
    height, width, channel = cropped_image_processed.shape
    #print("Height: ", height, "Width: ", width, "No. of channel: ", channel)


    #RUNNING THE FUNCTIONS:
    blocks = compute_blocks(cropped_image_processed, block_size, height, width)

    height_blocks, width_blocks, channels_blocks = blocks.shape



    distances = computeColorDistances(blocks)

    variability_score = computeTotalVariabilityScore(distances, threshold)

    if variability_score == 0:
        variability_score = 0.01

    return variability_score