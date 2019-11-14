import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import skimage
import utils



def convolve_im(im: np.array,
                fft_kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for visualizing different parts of the 
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    
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
    verbose = True

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass,
                                 verbose=verbose) 
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass,
                                  verbose=verbose)
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    f_im = np.fft.fftshift(np.fft.fft2(im, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('Image')
	
    plt.subplot(1, 3, 2)
    f_im = np.fft.fftshift(np.fft.fft2(image_high_pass, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('High Pass')
	
    plt.subplot(1, 3, 3)
    f_im = np.fft.fftshift(np.fft.fft2(image_low_pass, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('Low Pass')
	
	
    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)
