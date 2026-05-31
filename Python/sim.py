#  FUNCTION File: Simulate the Network
#  Author: Arshad Afzal, IIT Kanpur, India
#  For Questions/ Comments, please email to arshad.afzal@gmail.com
from rbf import *


def sim(x_test, theta, c, lw):
    q = (np.shape(x_test))[0]
    r = (np.shape(x_test))[1]
    t = (np.shape(c))[0]
    phi_test = np.zeros([q, t])
    #  Create Design Matrix
    for i in range(t):
        for j in range(q):
            phi_test[j][i] = rbf(x_test[j:j + 1, 0:r], c[i:i + 1, 0:r], theta)
    bias = np.ones([q, 1])  # Add bias term
    phi_test = np.concatenate((bias, phi_test), axis=1)
    yp = np.dot(phi_test, lw)
    return yp
