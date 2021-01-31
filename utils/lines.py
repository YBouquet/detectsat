import numpy as np
import math
from bresenham import bresenham


def get_points(rho, theta):
    """
    Generate points belonging to the line with (rho, theta) parameters

    Parameters
    ----------
    rho : float
    theta : float
        radian
    Return
    ----------
    (x0,y0),(x2,y2),(x2,y2): tuple(float), tuple(float), tuple(float)
        x,y coordinates of 3 points belonging to the (rho, theta) line
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


def get_slope(rho, theta):
    """
    Generate a point of (rho, theta) line and a linear function defining this line
    along the x axis
    Parameters
    ----------
    rho : float
    theta : float
        radian
    x_offset : int
        translation on x axis
    y_offset : int
        translation on y axis
    Return
    ----------
    p0, f : tuple(float), function
    """
    p0,p1,p2 = get_points(rho, theta)
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    return p0, (lambda x : y0 - (y2-y1)/(x2-x1) * x0 +  (y2-y1)/(x2-x1)*x)

def get_slope_parameters(rho, theta):
    """
    from the linear function x -> a*x+b, get a and b and a point from the line

    Parameters
    ----------
    rho : float
    theta : float
        radian
    Return
    ----------
    (x0,y0), b, a : tuple(float), float, float
    """
    p0,p1,p2 = get_points(rho, theta)
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    return (x0, y0), y0 - ((y2-y1)/(x2-x1)) * x0, ((y2-y1)/(x2-x1))

def build_line(rho, theta,i,h,w):
    """
    projection of a line onto pixels of a 2D image of size (h,w)
    Parameters
    ----------
    rho : float
    theta : float
        radian
    i: int
        translation on x axis
    Return
    ----------
    f_array : np.array(int)
        pixels coordinates belonging to the line
    """
    (x0,y0), f = get_slope(rho, theta)
    bresen_line = bresenham(int(np.floor(f(0)))+i, 0, int(np.floor(f(w-1)))+i, w-1)
    m_lits = list(bresen_line)
    f_array = np.array([[int(math.floor(x)), int(math.floor(y))] for x,y in m_lits])
    #print((f_array[:,0] < h) & (f_array[:,0] >=0) & (f_array[:,1] < w) & (f_array[:,1] >=0))
    f_array = f_array[(f_array[:,0] < h) & (f_array[:,0] >=0) & (f_array[:,1] < w) & (f_array[:,1] >=0)]
    return f_array
