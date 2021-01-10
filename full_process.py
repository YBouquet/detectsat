import matplotlib.pyplot as plt

import prologue

from post_processing import *
from mosaic import *
from img_processing import *
import time
import multiprocessing as mp
from matplotlib import pyplot as plt


def main(args):
    raw_img, unscaled_img = get_raw_image(args.i)#"OMEGA.2020-03-18T00_21_55.912_fullfield_binned.fits")
    crops_addresses = get_crops_addresses(raw_img)
    #hough_result = get_lines(raw_img, crops_addresses, len(crops_addresses.keys()), len(list(crops_addresses.items())[0][1]), args['h'])#"image04.npy")
    crops = []
    for i, row in enumerate(sorted(crops_addresses.keys())):
        for j, column in enumerate(crops_addresses[row]):
            crop = get_crop(raw_img, row, column)
            unscaled_crop = get_crop(unscaled_img, row, column)
            mm_crop = ((crop - np.min(crop) )/ (np.max(crop) - np.min(crop)) ) * 255
            mm_crop = mm_crop.astype(np.uint8())
            crops.append(((i,j), mm_crop, unscaled_crop, args.hough))

    print('Start Hough Processing...')
    start = time.time()
    if not(args.load_lines) :
        pool = mp.Pool(8)
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
            crop = get_crop(raw_img, m_row, crops_addresses[m_row][j])[8:-8,8:-8]
            params_tuples.append(tuple([crop, lines, i,j]))

    dict_lines = dict([((i,j), (crop,lines)) for crop, lines, i,j in params_tuples])
    """print('Start Post-Processing...')

    start = time.time()
    pool = mp.Pool(8)
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
            crop_img, crop_results = dict_lines[(i,j)]
            axes[i, j].imshow(crop_img)"""

    f, axes = plt.subplots(4,8, figsize=(64,64))
    for i in range(4):
        for j in range(8):
            lines = dict_lines[(i,j)][-1]
            post_process = dict_lines[(i,j)][0]
            new = post_process.copy()
            print('Draw lines for (%d,%d)'%(i,j))
            print(lines is not None, lines, lines.size )
            if lines is not None and len(lines) > 0:
                #lines = lines[lines[:,0,1] != 0]
                for line in lines:
                    for rho, theta in line:
                        _,p1,p2 = get_points(rho, theta)
                        new = cv2.line(new, p1, p2, (255, 0, 0), 2)
            axes[i,j].imshow(new)
    plt.savefig(args.o)
    plt.show()
#'raw_img_full.png')


if __name__ == '__main__':
    main(prologue.get_args())
