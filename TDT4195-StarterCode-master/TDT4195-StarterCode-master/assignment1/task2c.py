import numpy as np
import os
import matplotlib.pyplot as plt
from task2ab import save_im
import numpy as np


def convolve_im(im, kernel):
    """ A function that convolves im with kernel
    
    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, L]]
    
    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    y,x,_ = im.shape 
    m,n   = kernel.shape

    new_img = np.zeros_like(im)
    
    if (m == n):
        n = m//2
  
        for i in range(n,y-n):
            for j in range(n,x-n):
                for c in range(3):
                    new_img[i][j][c] = np.sum(im[i-n:i+n+1, j-n:j+n+1, c]*kernel)	 
    return new_img
   




if __name__ == "__main__":
    # Read image
    impath = os.path.join("images", "lake.jpg")
    im = plt.imread(impath)

    # Define the convolutional kernels
    h_a = np.ones((3, 3)) / 9
    h_b = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    # Convolve images
    smoothed_im1 = convolve_im(im.copy(), h_a)
    smoothed_im2 = convolve_im(im, h_b)

    # DO NOT CHANGE
    assert isinstance(smoothed_im1, np.ndarray), \
        f"Your convolve function has to return a np.array. " +\
        f"Was: {type(smoothed_im1)}"
    assert smoothed_im1.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert smoothed_im2.shape == im.shape, \
        f"Expected smoothed im ({smoothed_im1.shape}" + \
        f"to have same shape as im ({im.shape})"

    save_im("convolved_im_h_a.jpg", smoothed_im1)
    save_im("convolved_im_h_b.jpg", smoothed_im2)
