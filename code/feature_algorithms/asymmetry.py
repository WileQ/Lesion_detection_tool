import numpy as np
import matplotlib.pyplot as plt
import cv2


def asymmetry(mask):

    #find the mean of all rows and collumns to find the middle for both splits
    rows, cols = np.nonzero(mask)
    middle_row = int(np.mean(rows))
    middle_col = int(np.mean(cols))

    #spliting mask into 4 halves with the previously aquired values of borders
    left_half = mask[:, :middle_col]
    right_half = mask[:, middle_col:]
    top_half = mask[:middle_row, :]
    bottom_half = mask[middle_row:, :]

    #flip right and bottom half so they are on the same level as the left and top one
    rotated_right_half = cv2.flip(right_half, 1)
    rotated_bottom_half = cv2.flip(bottom_half, 0)

    #change right and left half to booleans
    left_half = left_half.astype(bool)
    rotated_right_half = rotated_right_half.astype(bool)

    #check which half is smaller and concatenate the smaller one with a boolean numpy array of zeros
    #we do it so we can later compare the two halfs
    if left_half.shape[1] > rotated_right_half.shape[1]:
        diff_x = left_half.shape[1]-rotated_right_half.shape[1]

        addi = np.zeros((left_half.shape[0],diff_x))
        addi_bool = addi.astype(bool)

        rotated_right_half = np.concatenate((addi_bool,rotated_right_half),axis=1)
    elif left_half.shape[1] < rotated_right_half.shape[1]:
        diff_x = rotated_right_half.shape[1]-left_half.shape[1]

        addi = np.zeros((left_half.shape[0], diff_x))
        addi_bool = addi.astype(bool)

        left_half = np.concatenate((addi_bool,left_half), axis=1)

    #we check the intersection and union of the two half and calculate asymmetry based on them
    intersection_lr = np.logical_and(left_half, rotated_right_half)
    union_lr = np.logical_or(left_half, rotated_right_half)
    asym_lr = np.sum(intersection_lr)/np.sum(union_lr)

    #we repeat the same process for the top and bottom half
    top_half = top_half.astype(bool)
    rotated_bottom_half = rotated_bottom_half.astype(bool)

    if top_half.shape[0] > rotated_bottom_half.shape[0]:
        diff_y = top_half.shape[0]-rotated_bottom_half.shape[0]

        addi = np.zeros((diff_y, top_half.shape[1],))
        addi_bool = addi.astype(bool)

        rotated_bottom_half = np.concatenate((addi_bool,rotated_bottom_half), axis=0)

    elif top_half.shape[0] < rotated_bottom_half.shape[0]:
        diff_y = rotated_bottom_half.shape[0]-top_half.shape[0]

        addi = np.zeros((diff_y, top_half.shape[1]))
        addi_bool = addi.astype(bool)

        top_half = np.concatenate((addi_bool, top_half), axis=0)

    intersection_tb = np.logical_and(top_half, rotated_bottom_half)
    union_tb = np.logical_or(top_half, rotated_bottom_half)
    asym_tb = np.sum(intersection_tb) / np.sum(union_tb)

    #we take both asymmetry values, add them and divide by 2, then multiply by 100 to get the value in percentage
    return round((asym_lr+asym_tb)/2,3)