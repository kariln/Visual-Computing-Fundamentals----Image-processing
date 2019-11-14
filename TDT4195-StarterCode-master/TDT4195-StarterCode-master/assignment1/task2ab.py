import matplotlib.pyplot as plt
import numpy as np
import os

image_output_dir = "image_processed"
os.makedirs(image_output_dir, exist_ok=True)


def save_im(imname, im, cmap=None):
    impath = os.path.join(image_output_dir, imname)
    plt.imsave(impath, im, cmap=cmap)


def greyscale(im):
    """ Converts an RGB image to greyscale
    
    Args:
        im ([type]): [np.array of shape [H, W, 3]]
    
    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
	
    # YOUR CODE HERE
    rgb_weights= np.array([0.212, 0.7152, 0.0722])
    x,y,_ = np.shape(im)
    img = np.zeros([x,y])
	
    for i in range(x):
        for j in range(y):
            img[i,j] = im[i,j].dot(rgb_weights)
			
    return img


def inverse(im):
    """ Finds the inverse of the greyscale image
    
    Args:
        im ([type]): [np.array of shape [H, W]]
    
    Returns:
        im ([type]): [np.array of shape [H, W]]
    """    	   
    x,y = np.shape(im)
    img = np.zeros([x,y])
	
    for i in range(x):
        for j in range(y):
            img[i,j] = 255 - im[i,j]
    return img


if __name__ == "__main__":
    im = plt.imread("images/lake.jpg")
    im = greyscale(im)
    inverse_im = inverse(im)
    save_im("lake_greyscale.jpg", im, cmap="gray")
    save_im("lake_inverse.jpg", inverse_im, cmap="gray")
