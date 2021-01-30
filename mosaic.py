from astropy.visualization import ZScaleInterval
from astropy.io import fits

import cv2
import numpy as np

import math
import time



def get_block(img, x_addresses, y_addresses):
    """
    Rescale pixels intensity according to the iraf's ZScale Algorithm

    Parameters
    ----------
    img : list[int]
        full mosaic
    x_addresses : numpy.array(np.float32)
        [min_x, max_x]
    y_addresses : list[int]
        [min_y, min_y]
    Return
    ----------
    result : numpy.array(np.float32)
        corresponding image block
    """
    min_x, max_x = x_addresses
    min_y, max_y = y_addresses
    result = img[min_x:max_x, min_y : max_y]
    if np.any(np.isnan(result)):
        raise Exception('Some values are NAN')
    return result

def scale_image(raw_img):
    """
    Rescale pixels intensity according to the iraf's ZScale Algorithm

    Parameters
    ----------
    raw_img : numpy.array(np.float32)
        Mosaic with raw intensities
    Return
    ----------
    raw_img : numpy.array(np.float32)
        Rescaled mosaic
    """
    s = ZScaleInterval()
    z1,z2 = s.get_limits(raw_img)
    raw_img[raw_img > z2] = z2
    raw_img[raw_img < z1] = z1
    return raw_img

def get_raw_image(filename):
    """
    Retrieve the mosaic from the fits file and apply a ZScale on it

    Parameters
    ----------
    filename : string

    Return
    ----------
    raw_img, data : (numpy.array(np.float32), numpy.array(np.float32))
        Rescaled mosaic, unscaled mosaic
    """
    hdul = fits.open(filename)
    data = hdul[1].data
    raw_img = scale_image(data[::-1].copy())
    return raw_img, data[::-1]

def get_blocks_addresses(raw_img):
    """
    Isolating the 32 blocks of the mosaic by registering for each block
    the indices of their corners

    Parameters
    ----------
    raw_img : numpy.array(np.float32)
        Zscaled mosaic
    Return
    ----------
    crops_addresses : dict
    """
    nans = np.argwhere(np.isnan(raw_img))
    bunches = nans[(nans[:,1] == 0) | (nans[:,0] == 0)]
    cuts_y = []
    cuts_x = []
    temp_y = -1
    temp_x = -1
    for i in bunches :
        x,y = tuple(i)
        if x == 0 :
            if abs(y - temp_y) > 1:
                cuts_y.append([temp_y+1, y])
            temp_y = y
        if y == 0:
            if abs(x - temp_x) > 1:
                cuts_x.append([temp_x+1, x])
            temp_x = x
    cuts_x.append([temp_x+1, raw_img.shape[0]])
    cuts_y.append([temp_y+1, raw_img.shape[1]])
    cuts_y = np.array(cuts_y)
    cuts_x = np.array(cuts_x)
    crops_addresses = {}
    current_study_x = cuts_x
    current_study_y = cuts_y
    for t_x in current_study_x:
        min_x, max_x = tuple(t_x)
        list_tx = [tuple(t_x)]
        if max_x - min_x > 7000:
            tmp_middle = int((max_x + min_x) / 2)
            list_tx = [(min_x, tmp_middle), (tmp_middle, max_x)]
        for sub_tx in list_tx:
            crops_addresses[sub_tx] = []
            for t_y in current_study_y:
                 crops_addresses[sub_tx].append(tuple(t_y))

    return crops_addresses
