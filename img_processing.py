import cv2
import numpy as np
from canny import cannyEdgeDetector
import math
from scipy.signal import convolve2d


def saturated_stars(unscaled_img):
    sigma = np.std(unscaled_img)
    mean = np.mean(unscaled_img)
    indices = np.argwhere(unscaled_img > mean + 3*sigma)
    mask = np.zeros(unscaled_img.shape).astype(np.uint8)
    mask[indices[:,0], indices[:,1]] = 1
    mask_1 = np.zeros(unscaled_img.shape).astype(np.uint8)
    indices = np.argwhere( unscaled_img > mean)
    mask_1[indices[:,0], indices[:,1]] = 1
    seed = mask * mask_1
    disk_mask =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    for i in range(100):
        prev_seed = seed
        seed = cv2.dilate(seed, disk_mask, iterations = 1) * mask_1
        if (seed == prev_seed).all():
            break
    return seed#, boxes

def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    gabor = gauss * sinusoid
    return gabor

def remove_bad_pixels(crop):
    mean_ = np.mean(crop)
    std_ = np.std(crop)
    filterSize =(10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,filterSize)
    tophat_img = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, kernel)
    indices = np.argwhere(tophat_img > 125)
    xs, ys = indices[:,0], indices[:,1]

    new_values = std_ * np.random.randn(len(xs)) + mean_
    min_ = np.min(new_values)
    new_values = (new_values - min_)/(np.max(new_values) - min_)

    crop[xs, ys] = new_values
    return tophat_img,crop

def process_crop(params, gabor_k_size = 16):
    id_, crop, unscaled_crop, h_threshold = params
    fth , tcrop = remove_bad_pixels(crop)
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
    filterSize =(10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,filterSize)
    tophat_img = cv2.morphologyEx(res_mean, cv2.MORPH_TOPHAT, kernel)
    #print(np.max(tophat_img), np.min(tophat_img), np.mean(tophat_img), np.std(tophat_img))
    (retVal, img_gseuil)=cv2.threshold(tophat_img, 80, 255, cv2.THRESH_BINARY)
    sortie = img_gseuil[16:-16,16:-16]
    #print(np.unique(sortie))
    h,w = sortie.shape
    post_process = np.zeros((h,w,3)).astype(int) + crop[8:-8, 8:-8].reshape(h,w,1).astype(int)
    detector = cannyEdgeDetector([sortie], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
    gauss, nonmax, th, imgs_final = detector.detect()
    lines = cv2.HoughLines(np.uint8(imgs_final[0]),1, np.pi / 180, h_threshold)

    print('End thread (%d,%d)'%id_)
    return (id_, (post_process, fth, tophat_img, img_gseuil, gauss[0], nonmax[0], th[0], imgs_final[0],lines))


def get_points(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * (a))
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * (a))
    return (x0,y0),(x1,y1),(x2,y2)

"""def band_from_line(mask, th_crop, line):
    h,w = th_crop.shape
    new = np.zeros((h,w,3)).astype(np.uint8())
    vectors = []
    for rho, theta in line:
        for i in range(40,0,-1):
            _, p1,p2 = get_points(rho, theta)
            tmp = np.concatenate((np.zeros((i,w,3)), new.copy()), axis = 0)
            np_line = cv2.line(tmp, p1, p2, (0, 255, 0), 1)[i:,:,1]
            result_line = th_crop[np_line>0]
            vectors.append(result_line)
        for i in range(40):
            _, p1,p2 = get_points(rho, theta)
            np_line = cv2.line(new[i:].copy(), p1, p2, (0, 255, 0), 1)
            np_line = np.concatenate((new[:i], np_line), axis = 0)[:,:,1]
            result_line = th_crop[np_line>0]
            vectors.append(result_line)
    analyse = np.full((len(vectors), np.max([len(v) for v in vectors])), np.nan)
    for i, row in enumerate(vectors):
        for j, pix in enumerate(row):
            analyse[i,j] = pix
    y = np.nanmedian(np.array(vectors).reshape(-1,w), axis = 1)
    bande = np.argwhere(y>0).reshape(-1)
    i_s = np.arange(-40,40)[bande]
    for rho, theta in line:
        for i in i_s :
            if i >= 0:
                _,p1,p2 = get_points(rho, theta)
                np_line = cv2.line(mask[i:], p1, p2, (1, 1, 1), 1)
                mask = np.concatenate((mask[:i], np_line), axis = 0)
            else:
                _,p1,p2 = get_points(rho, theta)
                tmp = np.concatenate((np.zeros((-i,w,3)), mask), axis = 0)
                mask = cv2.line(tmp, p1, p2, (1, 1, 1), 1)[-i:]
    return mask

def get_satellites_from_crop(parameters):
    id_, th_crop, lines = parameters
    print("Analysing (%d,%d)..." % id_)
    h,w = th_crop.shape
    mask = np.zeros((h,w,3)).astype(np.uint8())
    if lines is not None:
        for line in lines:
            mask = band_from_line(mask, th_crop, line)
    print("Ending (%d,%d)..." % id_)
    return (id_, mask[:,:,0])"""
