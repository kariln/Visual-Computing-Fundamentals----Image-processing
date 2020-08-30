import utils
import numpy as np


def position_check(im,x,y):
    #finds the size of the image, height - H, width -W
    (H,W) = im.shape
    
    #checks if the seed_point is within the image
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            x_bol = (i < W) and (i >= 0)
            y_bol = (j < H) and (j >= 0)
            centre = (i == x) and (j == y)
    if (x_bol and y_bol and not centre):
        return True
    else:
        return False

def neighbourhood(im, segmented,x,y,T):
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if position_check(im,i,j):
                #threshold, T, as homogeneity criteria (defines the maximum absolute difference in intensty between seed point and pixel)
                intensity_bol = (np.abs(im[i,j] - im[x,y]) <= T)
                if not segmented[i,j] and intensity_bol:
                    segmented[i,j] = True
                    #segments recursively
                    neighbourhood(im,segmented,i,j,T)

def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness). 
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row, col). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    segmented = np.zeros_like(im).astype(bool)
    for row, col in seed_points:
        segmented[row, col] = True
        neighbourhood(im, segmented,row,col,T)
        
    return segmented
    ### END YOUR CODE HERE ### 




if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [ # (row, column)
        [254, 138], # Seed point 1
        [253, 296], # Seed point 2
        [233, 436], # Seed point 3
        [232, 417], # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, \
        "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, \
        "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented.png", segmented_image)

