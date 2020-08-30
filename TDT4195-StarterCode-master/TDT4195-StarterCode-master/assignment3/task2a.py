import numpy as np
import skimage
import utils
import pathlib


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    ### START YOUR CODE HERE ### (You can change anything inside this block) 
    #the steps and formulas are taken from the otsu_thresholding.pdf handout
    
    # You can also define other helper functions
    L = 256 #as the range is from 0 to 255
    
    # 1. Compute normalized histogram of the input image 
    # The components of the histogram is denoted p_i, i = 0,1,2,..., L-1
    hist, bin_edges = np.histogram(im, L, (0,(L-1)))
    sum_hist = np.sum(hist)
    p = hist/sum_hist
    
    
    # 2. Compute the cumulative sums, P_1(k), k = 0,1,2,..., L-1
    # P_1 is the sum of p_i, i = 0,1,2,..., k
    # equation (10-49)
    
    P_1 = np.zeros_like(p)
    for k in range(0, len(p)):
        if (k == 0):
            P_1[0] = p[0]
        else:
            P_1[k] = P_1[k-1] + p[k]
    
    # 3. Compute the cumulative mean (average intensity) , m(k), up to level k 
    # equation (10-53)
    m = np.zeros_like(p)
    for k in range(0,len(p)):
        if(k == 0):
            m[0] = 0
        else:
            m[k] = m[k-1] + p[k]*k
    
    # 4.Compute the global mean (average intensity of the entire image), m_g
    # equation (10-54)
    m_g = 0
    for i in range(0,L):
        m_g += p[i]*i
    
    # 5. Compute the between class variance, sigma_B_2, for k = 0,1,2,...L-1
    # equation (10-62)
    sigma_B_2 = np.divide((m_g*P_1-m)**2,(P_1*(1-P_1)))
    sigma_B_2[np.isnan(sigma_B_2)] = 0 #exchanges nan with 0

    
    # 6. Obtain Otsu threshold k_star as the value of k for which sigma_B_2 is maximum, this is the optimal threshold
    # equation (10-63)
    # If no unique maximum exists, it is customary to average the values of k in which sigma_B_2 is maximum
    
    sigma_B_2_max = np.max(sigma_B_2)
    k_star_list = np.argwhere(sigma_B_2 == sigma_B_2_max)
    if(len(k_star_list) == 1):
        k_star = np.argmax(sigma_B_2)
    else:
        k_star = np.divide(np.sum(k_star_list,len(k_star_list)))
         
    # 7. Compute the global variance sigma_G_2
    # equation (10-58)
    
    """
    Skal det være m_g eller m_g[i]
    hvordan setter jeg i så fall m_g?
    """
    
    sigma_G_2 = np.empty(len(p))
    for i in range(0,L):
        if (i == 0):
            sigma_G_2[0] = (0-m_g)**2*p[0]
        else:
            sigma_G_2[i] = ((i-m_g)**2*p[i])
    
    #Obtain the seperability measure eta_star with k = k_star
    # equation (10-61)
    eta = sigma_B_2 / sigma_G_2
    eta_star = eta[k_star]
    
    threshold = k_star
    return threshold
    ### END YOUR CODE HERE ### 


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, \
            "Expected image shape ({}) to be same as thresholded image shape ({})".format(
                im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, \
            "Expected thresholded image dtype to be np.bool. Was: {}".format(
                segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)


