#  FUNCTION File: Radial Basis Function
#  Author: Arshad Afzal, IIT Kanpur, India
#  For Questions/ Comments, please email to arshad.afzal@gmail.com

import numpy as np
from math import *


def rbf(x, c, theta):
    f = np.exp(- pow(theta, 2) * np.sum(np.square(x - c)))
    return f
