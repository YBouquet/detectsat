import matplotlib.pyplot as plt
import multiprocessing as mp

import numpy as np
from img_processing import process_crop
import math
from mosaic import *
from post_processing import *
import prologue
from lines import *



def get_lines(img, unscaled_img, output_file, tmp,  h_threshold = 200):

    outputs = []
    mm_crop = ((img - np.min(img) )/ (np.max(img) - np.min(img)) ) * 255
    mm_crop = mm_crop.astype(np.uint8)
    print('Process...')
    _,dict_lines = process_crop(((0,0), mm_crop, unscaled_img, h_threshold))

    lines = dict_lines[-1]
    np.save('./np_results/'+tmp[0][-4]+'_'+str(tmp[1])+str(tmp[2])+".npy", lines)
    post_process = dict_lines[0]
    new = post_process.copy()
    print('Draw lines...')
    if lines is not None:
        #lines = lines[lines[:,0,1] != 0]
        print(lines)
        for line in lines:
            for rho, theta in line:
                _,p1,p2 = get_points(rho, theta)
                new = cv2.line(new, p1, p2, (255, 0, 0), 2)
    else:
        lines = []
    return dict_lines, new

def main(args):
    print('Retrieving fits file...')
    raw_img, unscaled_img = get_raw_image(args.i)#"OMEGA.2020-03-18T00_21_55.912_fullfield_binned.fits")
    crops_addresses = get_crops_addresses(raw_img)
    m_row = list(crops_addresses.keys())[args.subcrop_i]
    crop = get_crop(raw_img, m_row, crops_addresses[m_row][args.subcrop_j])#[8:-8,8:-8]
    unscaled_crop = get_crop(unscaled_img, m_row, crops_addresses[m_row][args.subcrop_j])

    print('Crop Retrieved')
    print('Starting Hough Process')
    imgs, final_img = get_lines(crop,unscaled_crop, args.h, (args.o, args.subcrop_i, args.subcrop_j), h_threshold = args.hough)#"image04.npy")

    f, axes = plt.subplots(1,len(imgs)+1+3, figsize = (64,64))
    for i in range(len(imgs)):
        if i < len(imgs) -1:
            crop_img = imgs[i]
            axes[i].imshow(crop_img)
        else:
            axes[i].imshow(final_img)
    print('Hough process done, start distinguish_satellites')

    parameters = tuple([crop, imgs[-1], 0,0])

    _, results = retrieve_raw_satellites(parameters)

    mmc, thds, imgsds, fth, tuple_= results

    axes[-1].imshow(mmc)
    axes[-2].imshow(fth)
    axes[-3].imshow(imgsds)
    axes[-3].imshow(thds)

    plt.savefig(args.o)
    plt.show()
    #'raw_img_full.png')


if __name__ == '__main__':
    main(prologue.get_args())
