import cv2
import numpy as np
import math
import networkx as nx
from lines import *

datapath = 'graphs/'


def get_window_from_line(img, line, theta_step = 0.1*math.pi/180., theta_midrange = 10, axis_midrange = 30):

    h,w = img.shape
    for rho, theta in line:
        rot_bands = []
        max_means = []
        db_s = []
        if theta >= 10e-1 :#and rho>0:
            #fig, axes = plt.subplots(theta_midrange + 1, 2, figsize = (20,20))
            for k, j in enumerate(range(-theta_midrange,theta_midrange+1)):
                band = []
                tmp_theta = j*theta_step + theta
                for i in range(-axis_midrange,axis_midrange):

                    m_final_line = build_line(rho, tmp_theta, i, h, w)
                    tmp_img = img[m_final_line[:,0], m_final_line[:,1]]
                    tmp_band = np.full(h, np.nan)
                    tmp_band[:len(tmp_img)] = tmp_img
                    band.append(tmp_band)
                y = np.nanmean(band, axis = 1)
                #axes[k//2, k%2].imshow(np.array(band)[:,:w])
                if len(y)>0:
                    min_pos = np.min(y[y>0])
                    db_y= 10*np.log(y/min_pos)
                    max_means.append((round(np.max(y)), np.sum(y), len(db_y[db_y >= np.max(db_y) + 10*math.log(1/2)])))
                    db_s.append(db_y)
                else:
                    max_means.append(0)
                    db_s.append(None)
                rot_bands.append(band)

            np_max_means = np.array(max_means)
            maxx_means = np_max_means[np_max_means[:,0] == np.max(np_max_means[:,0])]
            maxxz_means = maxx_means[maxx_means[:,2] == np.min(maxx_means[:,2])]
            maxxyz_means = maxxz_means[maxxz_means[:,1] == np.max(maxxz_means[:,1])]
            index = np.argwhere(np.all(np_max_means == maxxyz_means, axis = 1)).reshape(-1)[0]
            final_band = np.array(rot_bands)[index]
            final_db = db_s[index]
            band_3db = np.argwhere(final_db >= np.max(final_db) + 10 * math.log(1/2)).reshape(-1)

            final_j =  np.arange(-theta_midrange,theta_midrange+1)[index]
            result_band = np.arange(-axis_midrange,axis_midrange)[band_3db]
            return rho, line[0][1] + theta_step *final_j, result_band
        else:
            return rho, 0, None

def distinguish_satellites(h,w, h_results, id_, threshold = 200000 ):

    if h_results is not None and len(h_results) > 0:
        print(h_results, h_results.shape)
        filtered_lines = h_results
        n = len(filtered_lines)
        m_lines = []
        for i, line in enumerate(filtered_lines):
            for rho, theta in line :
                if theta != 0 :
                    (x0,y0), b, a = get_slope_parameters(rho, theta)
                    F = lambda x : (a*(x**2))/2 + b*x
                    m_lines.append(F(w-1) - F(0))
                else:
                    print('vertical line !!! (%d,%d)' % id_)
        dist_mat = np.zeros((n,n))
        for i,x in enumerate(m_lines) :
            for j,y in enumerate(m_lines):
                dist_mat[i,j] = abs(x - y)
        adj_mat = (dist_mat < threshold).astype(float) - np.eye(dist_mat.shape[0])
        G = nx.from_numpy_matrix(adj_mat)
        print("Number of satellites : %d" % nx.number_connected_components(G))
        return [np.median(filtered_lines[list(c)], axis = 0) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    #print("Number of satellites : 0")
    return []

def get_satellites_blocs(img, h_result, id_):

    rs = []
    ts = []
    bs = []
    h,w = img.shape
    lines = distinguish_satellites(h,w,h_result, id_)
    for i, line in enumerate(lines):
        #print(i,line)
        r,t,b= get_window_from_line(img, line, theta_step = 0.05*math.pi/180., theta_midrange = 10, axis_midrange = 30)
        if b is not None:
            rs.append(r)
            ts.append(t)
            bs.append(b)
    return lines, rs, ts, bs

palette = [(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255)] #red, green, yellow, majenta, cyan

def retrieve_raw_satellites(params):#raw_img, crops_addresses, h_results, i=0, j=0):

    crop, h_result, i,j = params
    id = (i,j)
    print('Start thread (%d,%d)'%id)
    mm_crop = ((crop - np.min(crop) )/ (np.max(crop) - np.min(crop)) ) * 255
    mm_crop = mm_crop.astype(np.uint8())
    filterSize =(30, 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize)
    tophat_img = cv2.morphologyEx(mm_crop, cv2.MORPH_TOPHAT, kernel)
    (retVal, img_gseuil)=cv2.threshold(tophat_img, 120, 1., cv2.THRESH_BINARY)
    th_crop = np.multiply(crop, img_gseuil)
    lines, rs, ts, bs = get_satellites_blocs(th_crop, h_result, (i,j))
    h,w = th_crop.shape
    new = np.zeros((h,w,3)).astype(int) + mm_crop.reshape(h,w,1).astype(int)
    for i, r in enumerate(rs):
        for j in bs[i]:
            bresen_line = build_line(r, ts[i], j, h,w)
            new[bresen_line[:,0], bresen_line[:,1]] = palette[i%5]

    print('End thread (%d,%d)'%id)
    return id, (new, tophat_img, img_gseuil, th_crop, (lines, rs,ts,bs))
