import skimage
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import numpy as np
import utils
from task4b import convolve_im



def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [h, w] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
	
    ### START YOUR CODE HERE ### (You can change anything inside this block) 
	
    H,W = np.shape(im)
    h,w = np.shape(kernel)
    t_b = (H-h)//2
    l_r = (W-w)//2
    kernel_padded = np.pad(kernel, ((t_b, t_b+1),(l_r, l_r+1)), 'constant')
    kernel_padded = np.pad(kernel, ((0, 2*t_b),(0, 2*l_r)), 'constant')
    fft_kernel = np.fft.fft2(kernel_padded, s=None, axes=(-2, -1), norm=None)
    
       
    im_fft = np.fft.fft2(im, s=None, axes=(-2, -1), norm=None)    
    im_filt = im_fft*fft_kernel    
    conv_result = np.fft.ifft2(im_filt, s=None, axes=(-2, -1), norm=None).real    

    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(12, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 2, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 2, 2) 
        plt.imshow(conv_result, cmap="gray")

    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    # DO NOT CHANGE

    impath = os.path.join("images", "clown.jpg")
    im = skimage.io.imread(impath)
    im = utils.uint8_to_float(im)
    kernel = np.load("images/notch_filter.npy")

    ### START YOUR CODE HERE ### (You can change anything inside this block)
    im_filtered = convolve_im(im, kernel, verbose = True)

    ### END YOUR CODE HERE ###

    utils.save_im("clown_filtered.png", im_filtered)
	
    H,W = np.shape(im)
    h,w = np.shape(kernel)
    t_b = (H-h)//2
    l_r = (W-w)//2
    kernel_padded = np.pad(kernel, ((0, 2*t_b),(0, 2*l_r)), 'constant')
	
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    f_im = np.fft.fftshift(np.fft.fft2(im, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('Image')
	
    plt.subplot(1, 3, 2)
    f_im = np.fft.fftshift(np.fft.fft2(im_filtered, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('Filtered')
	
    plt.subplot(1, 3, 3)
    f_im = np.fft.fftshift(np.fft.fft2(kernel_padded, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=0.01, vmax = 10),  aspect='auto')
    plt.colorbar()   
    plt.title('Kernel')
	

