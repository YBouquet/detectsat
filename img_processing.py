import cv2
import numpy as np
from canny import cannyEdgeDetector
import math
from scipy.signal import convolve2d


def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    """
    Generate gabor filters
    """
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    gabor = gauss * sinusoid
    return gabor

def mask_bad_pixels(crop, unscaled_crop):
    """
    Retrieve mask with bad pixels (bad columns, persistence effect, saturated 'inliers' and 'outliers' pixels)
    """
    filterSize =(8, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,filterSize)
    tophat_img = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, kernel)
    return (((crop < 40) | (unscaled_crop > 1000) |(tophat_img > 125))*1).astype(np.uint8)

def morphological_reconstruction(mask, bin_img, kernel_size):

    seed = mask * bin_img
    disk_mask =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    for i in range(100):
        prev_seed = seed
        seed = cv2.dilate(seed,disk_mask, iterations = 1) * bin_img
        if (seed == prev_seed).all(): #check if tshe seed has changed 
            break #if not we stop the process
    return seed

def process_crop(params, gabor_k_size = 16):

    id_, crop, unscaled_crop, h_threshold = params


    # BAD COLUMNS DETECTION
    bad_pix_mask = mask_bad_pixels(crop, unscaled_crop)
    h,w = bad_pix_mask.shape
    for _ in range(10): # applying the process 5 times systematically removes all noise in the mask
        tmp_old = bad_pix_mask.copy()
        bad_pix_mask_trans = np.concatenate((bad_pix_mask, np.zeros((50,w)).astype(np.uint8)))[50:] # apply a vertical translation
        tmp_pix_mask = bad_pix_mask * bad_pix_mask_trans # intersection between the mask and its translation
        bad_pix_mask = morphological_reconstruction(tmp_pix_mask, bad_pix_mask,5) # reconstruction the object that didn't disapear during the intersection
        if (tmp_old == bad_pix_mask).all():
            break
    mask_dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    bad_pix_mask=cv2.dilate(bad_pix_mask.astype(np.uint8), mask_dil, iterations=2)

    # GABOR FILTERING
    thetas = [k * math.pi / 4 for k in range(1,5)]
    print('Start thread (%d,%d)'%id_)
    result = []
    for theta in thetas :
        g = genGabor((16,16), 0.35, theta)
        tmp = convolve2d(crop, g)
        result.append(tmp)
        res_mean = np.zeros(result[0].shape)
    for conv in result:
        res_mean += conv
    res_mean = res_mean / len(result)

    # TOP-HAT TRANSFORM
    filterSize =(10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)
    tophat_img = cv2.morphologyEx(res_mean, cv2.MORPH_TOPHAT, kernel)

    # THRESHOLDING AND BAD COLUMNS REMOVAL
    (retVal, img_gseuil)=cv2.threshold(tophat_img, 80, 1, cv2.THRESH_BINARY)#70
    sortie = img_gseuil[16:-16,16:-16] # gabor filters generate a padding effect on the image due to the convolutions
    final_mask = morphological_reconstruction(bad_pix_mask[8:-8,8:-8], sortie,10) * bad_pix_mask[8:-8,8:-8]
    sortie = ((sortie * (1-final_mask))*255).astype(np.uint8)

    # CANNY FILTERING AND HOUGH TRANSFORM
    h,w = sortie.shape
    post_process = np.zeros((h,w,3)).astype(int) + crop[8:-8, 8:-8].reshape(h,w,1).astype(int)
    detector = cannyEdgeDetector([sortie], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
    gauss, nonmax, th, imgs_final = detector.detect()
    lines = cv2.HoughLines(np.uint8(imgs_final[0]), 1, np.pi / 180, h_threshold)

    print('End thread (%d,%d)'%id_)
    return (id_, (post_process, bad_pix_mask, tophat_img, img_gseuil, final_mask, sortie, imgs_final[0], lines))
