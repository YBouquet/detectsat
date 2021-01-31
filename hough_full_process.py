from matplotlib import pyplot as plt
import time
import os
import multiprocessing as mp


import utils.prologue as prologue

from utils.mosaic import *
from utils.img_processing import process_block

DATAPATH = './lines/'

def main(args):
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    raw_img, unscaled_img = get_raw_image(args.i)
    crops_addresses = get_blocks_addresses(raw_img)
    crops = []
    for i, row in enumerate(sorted(crops_addresses.keys())):
        for j, column in enumerate(crops_addresses[row]):
            crop = get_block(raw_img, row, column)
            unscaled_crop = get_block(unscaled_img, row, column)
            filename = DATAPATH + args.o[:-4] + '_%d%d'%(i,j) + '.npy'

            crops.append(((i,j), crop, unscaled_crop, args.hough,  filename, args.load_lines, args.save_lines))

    print('Full Processing...')
    global_start = time.time()
    pool = mp.Pool(8)
    with pool:
        dict_lines = dict(pool.map(process_block, crops))
    pool.join()
    pool.close()

    global_end = time.time()
    seconds = (global_end - global_start)
    print('... Ending Full Processing after %d min %d sec' % (seconds // 60, seconds % 60))

    f, axes = plt.subplots(4,8, figsize = (64,64))
    for i in range(4):
        for j in range(8):
            _,post_results = dict_lines[(i,j)]
            _, _, _, crop_img,crop_results = post_results
            axes[i, j].imshow(crop_img)

    plt.savefig(args.o)
    plt.show()


if __name__ == '__main__':
    main(prologue.get_args())
