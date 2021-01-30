from utils.mosaic import *
from utils.lines import *
import numpy as np
import cv2
import operator
import random
import gc

raw_image,_ = get_raw_image("OMEGA.2020-01-29T03_51_46.345_fullfield_binned.fits")
crops_addresses = get_blocks_addresses(raw_image)

x_train = []
y_train = []
has_satellite = []
tmp_mask = []

random.seed(123456789)

for i in range(4):
    for j in range(8):
        x_ = list(crops_addresses.keys())[i]
        crop = get_block(raw_image, x_, crops_addresses[x_][j])
        final_mask = saturated_stars(unscaled_crop)
        mask_dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation=cv2.dilate(1-final_mask,mask_dil, iterations=1)
        h,w = crop.shape
        for k in range(1):
            subh, subw = 64, 64
            for alpha in range(0,h, subh):
                for beta in range(0,w, subw):
                    if (alpha + subh) <= h and (beta+subw) <= w :
                        subcrop = crop[alpha:alpha+subh, beta:beta+subw].copy()
                        x_mirror = random.randint(0,1)
                        y_mirror = random.randint(0,1)
                        if x_mirror == 1:
                            subcrop = subcrop[::-1]
                        if y_mirror == 1:
                            subcrop = subcrop[:,::-1]
                        star_indices = np.argwhere(dilation[alpha:alpha+subh, beta:beta+subw] == 1)
                        replacement_value = np.median(subcrop)
                        subcrop[star_indices[:,0], star_indices[:,1]] = replacement_value
                        max_ = np.max(subcrop)
                        min_ = np.min(subcrop)
                        subcrop = (subcrop - min_) / (max_ - min_)
                        y_true = np.zeros(subcrop.shape)
                        mask = np.full(subcrop.shape, 0.)
                        decision = random.random()
                        hs = 0
                        if decision < 0.5:
                            hs = 1
                            s_length = random.randint(20,40)
                            s_width = random.randint(1,3)
                            theta = random.randint(0,179) * math.pi / 180.
                            x_where, y_where = random.randint(0, subh-s_length-1), random.randint(0,subw-s_length-1)
                            sat_line = np.zeros((s_length,s_length,3)).astype(np.uint8)
                            _,p1,p2 = get_points(0,theta)
                            sat_line = cv2.line(sat_line, tuple(map(operator.add, p1,(int(s_length/2),int(s_length/2)))),tuple(map(operator.add, p2,(int(s_length/2),int(s_length/2)))), (255,255,255), s_width)
                            h_sat,w_sat,_ = sat_line.shape
                            cx,cy = float(h_sat//2), float(w_sat//2)
                            r = min(cx,cy)
                            for a in range(h_sat):
                                for b in range(w_sat):
                                    if math.sqrt((a - cx)**2 + (b-cy)**2) > r:
                                        sat_line[a,b] = 0
                            final_synth = cv2.GaussianBlur(sat_line[:,:,0]/255,(s_width*2-1,s_width*2-1),0)
                            alpha_trans = random.randint(40,90)/100.
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
                        subcrop[star_indices[:,0], star_indices[:,1]] = ccrop[alpha:alpha+subh, beta:beta+subw][star_indices[:,0], star_indices[:,1]]
                        sub_max = np.max(subcrop)
                        sub_min = np.min(subcrop)
                        subcrop = (subcrop - sub_min) / (sub_max - sub_min)
                        has_satellite.append(hs)
                        x_train.append([subcrop])
                        y_train.append(y_true)
            del y_true
            del mask
            del ccrop
            gc.collect()
        del crop
        gc.collect()

datapath = "trainset/"
np.save(datapath + "2k_64_x.npy", np.array(x_train))
np.save(datapath + "2k_64_y.npy", np.array(y_train))
np.save(datapath + "2k_64_z.npy", np.array(has_satellite))
