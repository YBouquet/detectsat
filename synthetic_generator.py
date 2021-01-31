from utils.mosaic import *
from utils.lines import *
from utils.img_processing.py import = morphological_reconstruction
import utils.prologue as prologue
import numpy as np
import cv2
import os

import operator
import random
import gc

DATAPATH = "trainset/"

def saturated_stars(unscaled_img):
    """
    Get a mask with all saturated light blob by thresholding and morphological reconstruction
    """
    sigma = math.sqrt(stats.describe(unscaled_img.flatten()).variance)
    indices = np.argwhere(unscaled_img > 3*sigma)
    mask = np.zeros(unscaled_img.shape).astype(np.uint8)
    mask[indices[:,0], indices[:,1]] = 1
    mask_1 = np.zeros(unscaled_img.shape).astype(np.uint8)
    indices = np.argwhere( unscaled_img > stats.describe(unscaled_img.flatten()).mean)
    mask_1[indices[:,0], indices[:,1]] = 1
    final_mask = morphological_reconstruction(mask, mask_1, 2)
    return final_mask #, boxes



def main(args, seed = 123456789):
    raw_image,_ = get_raw_image(args.i)#"OMEGA.2020-01-29T03_51_46.345_fullfield_binned.fits")
    crops_addresses = get_blocks_addresses(raw_image)

    x_train = []
    y_train = []
    has_satellite = []
    tmp_mask = []

    random.seed(seed)

    for i in range(4):
        for j in range(8):
            x_ = list(crops_addresses.keys())[i]
            crop = get_block(raw_image, x_, crops_addresses[x_][j])
            final_mask = saturated_stars(unscaled_crop) # detect saturated light blob in the image
            mask_dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            dilation=cv2.dilate(final_mask,mask_dil, iterations=1)

            h,w = crop.shape
            for k in range(args.n): # number of samples generated from a single 64x64 patch
                subh, subw = 64, 64
                for alpha in range(0,h, subh):
                    for beta in range(0,w, subw):
                        if (alpha + subh) <= h and (beta+subw) <= w :
                            subcrop = crop[alpha:alpha+subh, beta:beta+subw].copy()
                            star_indices = np.argwhere(dilation[alpha:alpha+subh, beta:beta+subw] == 1)
                            replacement_value = np.median(subcrop)
                            subcrop[star_indices[:,0], star_indices[:,1]] = replacement_value # remove outlier pixels
                            max_ = np.max(subcrop)
                            min_ = np.min(subcrop)
                            if max_ != min_ : # white patch
                                x_mirror = random.randint(0,1)
                                y_mirror = random.randint(0,1)
                                if x_mirror == 1:
                                    subcrop = subcrop[::-1]
                                if y_mirror == 1:
                                    subcrop = subcrop[:,::-1]
                                subcrop = (subcrop - min_) / (max_ - min_) # rescaling
                                y_true = np.zeros(subcrop.shape)
                                mask = np.full(subcrop.shape, 0.)
                                decision = random.random()
                                hs = 0
                                if decision < 0.5:
                                    hs = 1

                                    # STREAK PARAMETRIZATION
                                    s_length = random.randint(20,40) # chose the length of the streak
                                    s_width = random.randint(1,3) # chose the width of the streak
                                    theta = random.randint(0,179) * math.pi / 180. # chose the direction of the streak
                                    x_where, y_where = random.randint(0, subh-s_length-1), random.randint(0,subw-s_length-1) # chose the position of the streak

                                    # DRAWING
                                    sat_line = np.zeros((s_length,s_length,3)).astype(np.uint8)
                                    _,p1,p2 = get_points(0,theta)
                                    sat_line = cv2.line(sat_line, tuple(map(operator.add, p1,(int(s_length/2),int(s_length/2)))),tuple(map(operator.add, p2,(int(s_length/2),int(s_length/2)))), (255,255,255), s_width)

                                    # APPLY THE STREAK IN THE PATCH
                                    h_sat,w_sat,_ = sat_line.shape
                                    cx,cy = float(h_sat//2), float(w_sat//2)
                                    r = min(cx,cy)
                                    for a in range(h_sat):
                                        for b in range(w_sat):
                                            if math.sqrt((a - cx)**2 + (b-cy)**2) > r:
                                                sat_line[a,b] = 0
                                    final_synth = cv2.GaussianBlur(sat_line[:,:,0]/255,(s_width*2-1,s_width*2-1),0)
                                    alpha_trans = random.randint(40,90)/100. # opacity of the streak
                                    final_synth = (final_synth / np.max(final_synth))
                                    tmp_mask.append(final_synth)
                                    mask[x_where:x_where+s_length, y_where:y_where+s_length] = final_synth
                                    indices = np.argwhere(mask > 0.)
                                    for subx, suby in indices :
                                        subcrop[subx,suby] = max(alpha_trans * mask[subx,suby] + (1-alpha_trans) * subcrop[subx,suby], subcrop[subx,suby])
                                    y_true[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line[:,:,0] / 255).astype(int)
                                    del sat_line
                                    gc.collect()
                                subcrop = subcrop * (max_ - min_) + min_
                                subcrop[star_indices[:,0], star_indices[:,1]] = ccrop[alpha:alpha+subh, beta:beta+subw][star_indices[:,0], star_indices[:,1]] # put the blobs in the image back
                                sub_max = np.max(subcrop)
                                sub_min = np.min(subcrop)
                                subcrop = (subcrop - sub_min) / (sub_max - sub_min)

                                # SAVING THE SAMPLES
                                has_satellite.append(hs)
                                x_train.append([subcrop])
                                y_train.append(y_true)
                del y_true
                del mask
                del ccrop
                gc.collect()
            del crop
            gc.collect()



    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    np.save(datapath + args.o + "_samples.npy", np.array(x_train))
    np.save(datapath + args.o + "_targets.npy", np.array(y_train))
    np.save(datapath + args.o + "_patch_targets.npy", np.array(has_satellite))


if __name__ == '__main__':
    main(prologue.get_args())
