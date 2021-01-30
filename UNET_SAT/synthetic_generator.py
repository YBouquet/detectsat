


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

def get_slope(rho, theta, x_offset = 0, y_offset = 0):
    p0,p1,p2 = get_points(rho, theta)
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    return p0, (lambda x : y0 - (y2-y1)/(x2-x1) * (x0 + y_offset) + x_offset +  (y2-y1)/(x2-x1)*x)

def get_slope_parameters(rho, theta, x_offset = 0, y_offset = 0):
    p0,p1,p2 = get_points(rho, theta)
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    return (x0+x_offset, y0+y_offset), y0+y_offset - ((y2-y1)/(x2-x1)) * (x0+x_offset), ((y2-y1)/(x2-x1))

def scale_image(unscaled_image):
    s = ZScaleInterval()
    z1,z2 = s.get_limits(unscaled_image)
    unscaled_image[unscaled_image > z2] = z2
    unscaled_image[unscaled_image < z1] = z1
    return unscaled_image


def get_raw_image(filename):
    hdul = fits.open(filename)
    data = hdul[1].data
    raw_img = data[::-1].copy()
    raw_img = scale_image(raw_img)
    return raw_img, data[::-1]




random.seed(123456789)
x_train = []
y_train = []
has_satellite = []
tmp_mask = []
for i in range(4):
    for j in range(8):
        x_ = list(crops_addresses.keys())[i]
        #unscaled_crop = get_crop(unscaled_img, x_, crops_addresses[x_][j])[:4000,:2000]
        unscaled_crop = get_crop(unscaled_img, x_, crops_addresses[x_][j])[:4000,:2000]
        crop = get_crop(raw_image, x_, crops_addresses[x_][j])[:4000,:2000]
        final_mask = saturated_stars(unscaled_crop)
        #min_kernel = max(1, np.min(np.array([[x,y] for _,_,x,y in boxes]).ravel()) //2)
        mask_dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation=cv2.dilate(1-final_mask,mask_dil, iterations=1)
        h,w = crop.shape
        for k in range(1):
            ccrop = crop.copy()
            sccrop = unscaled_crop.copy()
            n_satellites = random.randint(10,20)
            x_mirror = random.randint(0,1)
            y_mirror = random.randint(0,1)

            if x_mirror == 1:
                ccrop = ccrop[::-1]
                sccrop = sccrop[::-1]
            if y_mirror == 1:
                ccrop = ccrop[:,::-1]
                sccrop = sccrop[:,::-1]
            subh, subw = 64, 64
            for alpha in range(0,h, subh):
                for beta in range(0,w, subw):
                    if (alpha + subh) <= h and (beta+subw) <= w :
                        subcrop = ccrop[alpha:alpha+subh, beta:beta+subw].copy()
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
                            s_width = random.randint(2,3)
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
                            alpha_trans = random.randint(40,50)/100.
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
                        x_train.append([subcrop, sccrop[alpha:alpha+subh, beta:beta+subw]])
                        y_train.append(y_true)
            del y_true
            del mask
            del ccrop
            del sccrop
            gc.collect()
        del crop
        del unscaled_crop
        gc.collect()


sub_train = np.array(x_train)[np.argwhere(np.array(has_satellite) > 0).reshape(-1)]
sub_target = np.array(y_train)[np.argwhere(np.array(has_satellite) > 0).reshape(-1)]
ns_train = np.array(x_train)[np.argwhere(np.array(has_satellite) == 0).reshape(-1)]
ns_target = np.array(y_train)[np.argwhere(np.array(has_satellite) == 0).reshape(-1)]
print(len(sub_train), len(ns_train))

adds = np.arange(len(x_train))
np.random.shuffle(adds)

datapath = "trainset/"
np.save(datapath + "2k_64_x.npy", np.array(x_train[adds[:2000]]))
np.save(datapath + "2k_64_y.npy", np.array(y_train[adds[:2000]]))
np.save(datapath + "2k_64_z.npy", np.array(has_satellite[adds[:2000]]))

np.save(datapath + "64k_64_x.npy", np.array(x_train))
np.save(datapath + "64k_64_y.npy", np.array(y_train))
np.save(datapath + "64k_64_z.npy", np.array(has_satellit))
