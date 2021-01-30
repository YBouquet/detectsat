from matplotlib import pyplot as plt
import time
import multiprocessing as mp


import utils.prologue

from utils.post_processing import *
from utils.mosaic import *
from utils.img_processing import *


def main(args):
    raw_img, unscaled_img = get_raw_image(args.i)
    crops_addresses = get_blocks_addresses(raw_img)
    crops = []
    for i, row in enumerate(sorted(crops_addresses.keys())):
        for j, column in enumerate(crops_addresses[row]):
            crop = get_block(raw_img, row, column)
            unscaled_crop = get_block(unscaled_img, row, column)
            mm_crop = ((crop - np.min(crop) )/ (np.max(crop) - np.min(crop)) ) * 255
            mm_crop = mm_crop.astype(np.uint8())
            crops.append(((i,j), mm_crop, unscaled_crop, args.hough))

    print('Start Hough Processing...')
    start = time.time()
    if not(args.load_lines) :
        pool = mp.Pool(6)
        with pool:
            dict_lines = dict(pool.map(process_crop, crops))
        pool.join()
        pool.close()

    end = time.time()
    seconds = (end - start)
    print('... Ending Hough Processing after %d min %d sec' % (seconds // 60, seconds % 60))
    params_tuples = []

    for i in range(4):
        for j in range(8):
            print(i,j)
            if args.load_lines :
                lines = np.load('./lines/' + args.o[:-4] + '_%d%d'%(i,j) + '.npy', allow_pickle = True)
            else :
                lines = dict_lines[(i,j)][-1]
                if args.save_lines :
                    if lines is None:
                        lines = np.array([])
                    np.save('./lines/' + args.o[:-4] + '_%d%d'%(i,j) + '.npy', lines)

            m_row = list(crops_addresses.keys())[i]
            crop = get_block(raw_img, m_row, crops_addresses[m_row][j])[8:-8,8:-8]
            params_tuples.append(tuple([crop, lines, i,j]))

    print('Start Post-Processing...')

    start = time.time()
    pool = mp.Pool(6)
    with pool:
        dict_lines = dict(pool.map(retrieve_raw_satellites, params_tuples))
    pool.join()
    pool.close()
    end = time.time()
    seconds = (end - start)

    print('... Ending Post-Processing after %d min %d sec' % (seconds // 60, seconds % 60))

    f, axes = plt.subplots(4,8, figsize = (64,64))
    for i in range(4):
        for j in range(8):
            crop_img, _,_,_, crop_results = dict_lines[(i,j)]
            axes[i, j].imshow(crop_img)


    plt.savefig(args.o)
    plt.show()
#'raw_img_full.png')


if __name__ == '__main__':
    main(prologue.get_args())
