import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import skimage
import utils



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
    kernel_padded = np.pad(kernel, ((0, 2*t_b+1),(0, 2*l_r+1)), 'constant')
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
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)


    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    f_im = np.fft.fftshift(np.fft.fft2(im, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('Image')
	
    plt.subplot(1, 3, 2)
    f_im = np.fft.fftshift(np.fft.fft2(image_gaussian, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('Gaussian')
	
    plt.subplot(1, 3, 3)
    f_im = np.fft.fftshift(np.fft.fft2(image_sobelx, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 1000),  aspect='auto')
    plt.colorbar()   
    plt.title('Sobel')
	