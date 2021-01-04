%matplotlib inline
import matplotlib.pyplot as plt

import prologue

from post_processing import *
from hough import *
from img_processing import *
import time
import multiprocessing as mp
from matplotlib import pyplot as plt


def get_raw_image(filename):
    hdul = fits.open(filename)
    data = hdul[1].data
    s = ZScaleInterval()
    z1,z2 = s.get_limits(data)
    raw_img = data[::-1].copy()
    raw_img[raw_img > z2] = z2
    raw_img[raw_img < z1] = z1
    return raw_img, data[::-1]

def main(args):
    raw_img, _ = get_raw_image(args['i'])#"OMEGA.2020-03-18T00_21_55.912_fullfield_binned.fits")
    crops_addresses = get_crops_addresses(raw_img)
    #hough_result = get_lines(raw_img, crops_addresses, len(crops_addresses.keys()), len(list(crops_addresses.items())[0][1]), args['h'])#"image04.npy")
    for i, row in enumerate(sorted(parameters.keys())):
        for j, column in enumerate(parameters[row]):
            crop = get_crop(img, row, column)
            mm_crop = ((crop - np.min(crop) )/ (np.max(crop) - np.min(crop)) ) * 255
            mm_crop = mm_crop.astype(np.uint8())
            crops.append(((i,j), mm_crop, h_threshold))
            
    pool = mp.Pool(8)
    with pool:
        dict_lines = dict(pool.map(process_crop, crops))

    params_tuples = []

    for i in range(4):
        for j in range(8):
            m_row = list(crops_addresses.keys())[i]
            crop = get_crop(raw_img, m_row, crops_addresses[m_row][j])[8:-8,8:-8]
            params_tuples.append(tuple([crop, dict_lines[(i,j)][-1], i,j]))

    with pool:
        dict_lines = dict(pool.map(retrieve_raw_satellites, params_tuples))

    f, axes = plt.subplots(4,8, figsize = (64,64))
    for i in range(4):
        for j in range(8):
            crop_img, crop_results = dict_lines[(i,j)]
             axes[i, j].imshow(crop_img)

    plt.show()
    plt.savefig(args['0'])#'raw_img_full.png')


if __name__ == '__main__':
    main(prologue.get_args())
