from astropy.visualization import ZScaleInterval
from astropy.io import fits

import cv2
import numpy as np

import math
import time

def get_raw_image(filename):
    hdul = fits.open(filename)
    data = hdul[1].data
    s = ZScaleInterval()
    z1,z2 = s.get_limits(data)
    raw_img = data[::-1].copy()
    raw_img[raw_img > z2] = z2
    raw_img[raw_img < z1] = z1
    return raw_img, data[::-1]

def get_crops_addresses(raw_img):
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
    current_study_x = cuts_x[:]
    current_study_y = cuts_y[:]
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
