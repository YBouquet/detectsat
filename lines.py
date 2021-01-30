import numpy as np
import math
from bresenham import bresenham


def get_points(rho, theta):
    """
    Retrieve the mosaic from the fits file and apply a ZScale on it

    Parameters
    ----------
    filename : string

    Return
    ----------
    raw_img, data : (numpy.array(np.float32), numpy.array(np.float32))
        Rescaled mosaic, unscaled mosaic
    """
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
    """
    Retrieve the mosaic from the fits file and apply a ZScale on it

    Parameters
    ----------
    filename : string

    Return
    ----------
    raw_img, data : (numpy.array(np.float32), numpy.array(np.float32))
        Rescaled mosaic, unscaled mosaic
    """
    p0,p1,p2 = get_points(rho, theta)
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    return p0, (lambda x : y0 - (y2-y1)/(x2-x1) * (x0 + y_offset) + x_offset +  (y2-y1)/(x2-x1)*x)

def get_slope_parameters(rho, theta, x_offset = 0, y_offset = 0):
    """
    Retrieve the mosaic from the fits file and apply a ZScale on it

    Parameters
    ----------
    filename : string

    Return
    ----------
    raw_img, data : (numpy.array(np.float32), numpy.array(np.float32))
        Rescaled mosaic, unscaled mosaic
    """
    p0,p1,p2 = get_points(rho, theta)
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    return (x0+x_offset, y0+y_offset), y0+y_offset - ((y2-y1)/(x2-x1)) * (x0+x_offset), ((y2-y1)/(x2-x1))

def build_line(rho, theta,i,h,w, x_offset = 0, y_offset = 0):
    (x0,y0), f = get_slope(rho, theta, x_offset = x_offset, y_offset = y_offset)
    bresen_line = bresenham(int(np.floor(f(0)))+i, 0, int(np.floor(f(w-1)))+i, w-1)
    m_lits = list(bresen_line)
    f_array = np.array([[int(math.floor(x)), int(math.floor(y))] for x,y in m_lits])
    #print((f_array[:,0] < h) & (f_array[:,0] >=0) & (f_array[:,1] < w) & (f_array[:,1] >=0))
    f_array = f_array[(f_array[:,0] < h) & (f_array[:,0] >=0) & (f_array[:,1] < w) & (f_array[:,1] >=0)]
    return f_array
