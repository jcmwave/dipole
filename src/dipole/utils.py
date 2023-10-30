import numpy as np
from typing import Tuple

def cart2spherical(x: np.ndarray, y:np.ndarray,
                   z:np.ndarray)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy_sq = x**2 + y**2
    r = np.sqrt(xy_sq +z**2)
    theta = np.arctan2(np.sqrt(xy_sq), z)
    phi = np.arctan2(y, x)
    return r, theta, phi
    
def spherical2cart(r: np.ndarray, theta:np.ndarray,
                   phi:np.ndarray)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return x, y, z
