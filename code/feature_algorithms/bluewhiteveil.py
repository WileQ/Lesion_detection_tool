import numpy as np
from skimage import io, color, filters, morphology, exposure
import matplotlib.pyplot as plt

def evaluate_blue_white_veil(image, mask):
    '''Function that highlights potential blue-white veil in an image'''
    
    #removing alpha channel (otherwise it throws an error later)
    img = image[:, :, :3]
    
    #loading mask
    mask = mask

    #cropping the image to only show the masked part
    object_coords = np.where(mask != 0)
    min_x = min(object_coords[0])
    max_x = max(object_coords[0])
    min_y = min(object_coords[1])
    max_y = max(object_coords[1])

    img = img[min_x:max_x, min_y:max_y]

    #converting the image to grayscale to identify white areas
    gray_img = color.rgb2gray(img)
    
    #enhancing the contrast in the grayscale image to help identifying white areas
    #from the documentation: 
    #"An algorithm for local contrast enhancement, 
    #that uses histograms computed over different tile regions of the image. 
    #Local details can therefore be enhanced even in regions that are darker or lighter than most of the image."
    enhanced_gray_img = exposure.equalize_adapthist(gray_img)

    #convert the original image to Lab color space to extract the blue channel
    #(I tried to do this directly in RGB, but Lab proved to be better at identifying blue areas)
    lab_img = color.rgb2lab(img)
    blue_channel = lab_img[:, :, 2]

    #extract blue areas by taking the pixels that are more blue than the average
    #probably not the best method but I tried having a minimum blue value as well, but for some reason it did not really help
    blue_mask = blue_channel > np.mean(blue_channel)

    #extract white areas using the Otsu method
    #which minimises the intra-class variance of the black and white pixels in the image
    #to find a threshold that separates foreground (white in this case) from background
    #by selecting pixels with higher intensity than the threshold
    white_mask = enhanced_gray_img > filters.threshold_otsu(enhanced_gray_img)

    #combine the blue and white masks to have the first draft of potential blue-white veil
    #although at this point way too much has been masked
    combined_mask = np.logical_or(blue_mask, white_mask)
    
    #invert the combined mask to make blue/white areas black insteaed of the other way around
    inverted_mask = np.invert(combined_mask)

    #morphological opening to first do erosion and remove isolated pixels which seem to be likely to be errors
    #and then dilation to recover the lost pixels on the edges
    #this removes a lot of the errors, although some actual blue-white veil might also be removed
    opened_mask = morphology.opening(inverted_mask, morphology.disk(10))
    
    #highlight potential blue-white veil in the original image
    highlighted_img = img.copy()
    highlighted_img[opened_mask] = [0, 0, 0]  #set potential blue-white veil areas to black

    #compute the percentage of the image covered by the blue/white mask and the corresponding (selfmade) blue-white veil value
    percentage_blue_white_mask = (np.count_nonzero(opened_mask) / (img.shape[0] * img.shape[1])) * 100
    if percentage_blue_white_mask >= 1:
        blue_white_veil_value = 1 
    else:
        blue_white_veil_value = 0
        
    return blue_white_veil_value
