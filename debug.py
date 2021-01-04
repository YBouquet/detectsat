import matplotlib.pyplot as plt
import multiprocessing as mp

import numpy as np
from img_processing import process_crop
import math
from mosaic import *
from post_processing import *
import prologue


def get_hough_results(input_file):
    with open(input_file, 'rb') as f:
        result = np.load(f, allow_pickle = True)
    return result

def get_crop(img, x_addresses, y_addresses):
    min_x, max_x = x_addresses
    min_y, max_y = y_addresses
    result = img[min_x:max_x, min_y : max_y]
    if np.any(np.isnan(result)):
        raise Exception('Some values are NAN')
    return result

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

def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
#     myimshow(gauss)
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
#     myimshow(sinusoid)
    gabor = gauss * sinusoid
    return gabor


def get_lines(img, output_file, h_threshold = 200):
    outputs = []
    mm_crop = ((img - np.min(img) )/ (np.max(img) - np.min(img)) ) * 255
    mm_crop = mm_crop.astype(np.uint8())
    print('Process...')
    dict_lines = process_crop(((0,0), mm_crop, h_threshold))

    lines = dict_lines[(0,0)][-1]
    post_process = dict_lines[(0,0)][0]
    new = post_process.copy()
    print('Draw lines...')
    if lines is not None:
        lines = lines[lines[:,0,1] != 0]
        for line in lines:
            for rho, theta in line:
                _,p1,p2 = get_points(rho, theta)
                new = cv2.line(new, p1, p2, (255, 0, 0), 2)
    else:
        lines = []
    return dict_lines[(0,0)], new

def main(args):
    print('Retrieving fits file...')
    raw_img, _ = get_raw_image(args.i)#"OMEGA.2020-03-18T00_21_55.912_fullfield_binned.fits")
    crops_addresses = get_crops_addresses(raw_img)
    m_row = list(crops_addresses.keys())[args.subcrop_i]
    crop = get_crop(raw_img, m_row, crops_addresses[m_row][args.subcrop_j])#[8:-8,8:-8]
    print('Crop Retrieved')
    print('Starting Hough Process')
    imgs, final_img = get_lines(crop, args.h)#"image04.npy")

    f, axes = plt.subplots(1,len(imgs), figsize = (64,64))
    for i in range(len(imgs)):
        if i < len(imgs) -1:
            crop_img = imgs[i]
            axes[i].imshow(crop_img)
        else:
            axes[i].imshow(new)
    plt.show()
    plt.savefig(args.o)#'raw_img_full.png')


if __name__ == '__main__':
    main(prologue.get_args())
