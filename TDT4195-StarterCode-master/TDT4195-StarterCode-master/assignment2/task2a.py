import skimage
import numpy as np
import utils


def MaxPool2d(im: np.array,
              kernel_size: int):
    """ A function that max pools an image with size kernel size.
    Assume that the stride is equal to the kernel size, and that the kernel size is even.

    Args:
        im: [np.array of shape [H, W, 3]]
        kernel_size: integer
    Returns:
        im: [np.array of shape [H/kernel_size, W/kernel_size, 3]].
    """
    stride = kernel_size
    ### START YOUR CODE HERE ### (You can change anything inside this block) 
    H,W,_ = np.shape(im)
    h,w = int(H/kernel_size), int(W/kernel_size)
	
    new_im = np.zeros([h,w,3])
	
    for c in range(3):
        for i in range(h):
            for j in range(w):
                new_im[i,j,c] = np.max(im[i*stride:(i+1)*stride, j*stride:(j+1)*stride, c])			

    return new_im
    ### END YOUR CODE HERE ### 


if __name__ == "__main__":

    # DO NOT CHANGE
    im = skimage.data.chelsea()
    im = utils.uint8_to_float(im)
    max_pooled_image = MaxPool2d(im, 4)

    utils.save_im("chelsea.png", im)
    utils.save_im("chelsea_maxpooled.png", max_pooled_image)

    im = utils.create_checkerboard()
    im = utils.uint8_to_float(im)
    utils.save_im("checkerboard.png", im)
    max_pooled_image = MaxPool2d(im, 2)
    utils.save_im("checkerboard_maxpooled.png", max_pooled_image)