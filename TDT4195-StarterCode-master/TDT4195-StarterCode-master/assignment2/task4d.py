import numpy as np
import skimage
import utils
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from task4b import convolve_im

def sharpen(im: np.array):

    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
	
    ### START YOUR CODE HERE ### (You can change anything inside this block)
    im_filt = convolve_im(im, laplacian, verbose=True)
# JUST FOR TESTING	
#    im_filt = im_filt-im_filt.min()
#    im_filt = im_filt/im_filt.max()
#	
#    im = im-im.min()
#    im = im/im.max()
	
#    im_filt = np.where(im+im_filt > im.max(), im.max(), im_filt)
#    im_filt = np.where(im+im_filt < im.min(), im.min(), im_filt)
#    
    
    im_sharp = im+im_filt   
    
    
    #im_sharp = np.where(im_sharp > im.max(), im.max(), im_sharp)
    plt.imshow(im_sharp,cmap="gray")
	### END YOUR CODE HERE ###
    return im_sharp


if __name__ == "__main__":
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
	])
	
    # DO NOT CHANGE
    im = skimage.data.moon()  
    im = utils.uint8_to_float(im)
    sharpen_im = sharpen(im)

    sharpen_im = utils.to_uint8(sharpen_im)
    im = utils.to_uint8(im)
    # Concatenate the image, such that we get
    # the original on the left side, and the sharpened on the right side
    im = np.concatenate((im, sharpen_im), axis=1)
    utils.save_im("moon_sharpened.png", im)
	
	
    H,W = np.shape(im)
    h,w = np.shape(laplacian)
    t_b = (H-h)//2
    l_r = (W-w)//2
    kernel_padded = np.pad(laplacian, ((0, 2*t_b),(0, 2*l_r)), 'constant')
	
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    f_im = np.fft.fftshift(np.fft.fft2(im, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 10000),  aspect='auto')
    plt.colorbar()   
    plt.title('Image')
	
    plt.subplot(1, 3, 2)
    f_im = np.fft.fftshift(np.fft.fft2(sharpen_im, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=1, vmax = 10000),  aspect='auto')
    plt.colorbar()   
    plt.title('sharpened')
	
    plt.subplot(1, 3, 3)
    f_im = np.fft.fftshift(np.fft.fft2(kernel_padded, s=None, axes=(-2, -1), norm=None))
    plt.imshow(np.abs(f_im), norm=LogNorm(vmin=0.01, vmax = 10),  aspect='auto')
    plt.colorbar()   
    plt.title('Laplacian')
