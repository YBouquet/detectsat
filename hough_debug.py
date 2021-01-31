import matplotlib.pyplot as plt
import multiprocessing as mp

import numpy as np
import math
import os

from utils.mosaic import *
from utils.lines import *
from utils.img_processing import process_block
from utils.post_processing import *
import utils.prologue as prologue

DATAPATH = './debug_results/'
def get_lines(img, unscaled_img, output_file, tmp,  h_threshold = 200):

    outputs = []
    print('Process...')
    _, results_tuple = process_block(((0,0), img, unscaled_img, h_threshold, None , False ,False))
    dict_lines, post_dict_lines = results_tuple

    lines = dict_lines[-1]
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)

    np.save(DATAPATH+tmp[0][:-4]+'_'+str(tmp[1])+str(tmp[2])+".npy", lines)
    post_process = dict_lines[0]
    new = post_process.copy()
    print('Draw lines...')
    if lines is not None:
        print(lines)
        for line in lines:
            for rho, theta in line:
                _,p1,p2 = get_points(rho, theta)
                new = cv2.line(new, p1, p2, (255, 0, 0), 2)
    else:
        lines = []
    return dict_lines, new, post_dict_lines

def main(args):
    print('Retrieving fits file...')
    raw_img, unscaled_img = get_raw_image(args.i)#"OMEGA.2020-03-18T00_21_55.912_fullfield_binned.fits")
    crops_addresses = get_blocks_addresses(raw_img)
    m_row = list(crops_addresses.keys())[args.subcrop_i]
    crop = get_block(raw_img, m_row, crops_addresses[m_row][args.subcrop_j])#[8:-8,8:-8]
    unscaled_crop = get_block(unscaled_img, m_row, crops_addresses[m_row][args.subcrop_j])

    print('Block Retrieved')
    print('Starting Debug Process')
    imgs, final_img, post_dict_lines= get_lines(crop,unscaled_crop, args.h, (args.o, args.subcrop_i, args.subcrop_j), h_threshold = args.hough)#"image04.npy")

    f, axes = plt.subplots(1,len(imgs)+1+3, figsize = (64,64))
    for i in range(len(imgs)):
        if i < len(imgs) -1:
            crop_img = imgs[i]
            axes[i].imshow(crop_img)
        else:
            axes[i].imshow(final_img)
    print('Debug process done')

    thds, imgsds, fth, mmc,  tuple_= post_dict_lines

    axes[-1].imshow(mmc)
    axes[-2].imshow(fth)
    axes[-3].imshow(imgsds)
    axes[-4].imshow(thds)

    plt.savefig(args.o)
    plt.show()
    #'raw_img_full.png')


if __name__ == '__main__':
    main(prologue.get_args())
