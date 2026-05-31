#  Regression Using Radial-Basis Neural Networks
#  -----------------------------------------------------------------------------------------
#  Algorithm:Orthogonal Least Squares Learning Algorithm for Radial Basis Function Networks
#  S . Chen, C. F. N. Cowan, and P. M. Grant
#  IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 2, NO. 2, MARCH 1991
#  -----------------------------------------------------------------------------------------
#  =========================================================================================
#  INPUTS:
#  =========================================================================================
#  'x_train' Matrix of Inputs/Predictors/features (N-by-D)
#  'y_train' Vector of Response (N-by-1)
#  'theta'   Hyperparameter of the Network
#  'funtol'  Tolerance for Mean-squared Error
#  =========================================================================================
#  OUTPUT:
#  =========================================================================================
#  'lw' Weights in the Linear Layer(1st Entry in the Array is Bias Term)
#  =========================================================================================
#  Tip: Run the Code with Different Values of Hyperparameter, Theta  to Find the Best Value
#       OR use cross-validation instead
#  =========================================================================================
#  Version 1.0
#  Author: Arshad Afzal, IIT Kanpur, India
#  https://www.researchgate.net/profile/Arshad_Afzal
#  For Questions/ Comments, please email to arshad.afzal@gmail.com
#  ========================================================================================

import numpy as np
import matplotlib.pyplot as plt
from rbf import rbf


def rbnn(x_train, y_train, theta, funtol):
    #  Initialization
    q = (np.shape(x_train))[0]
    r = (np.shape(x_train))[1]
    x = x_train.copy()  # Create a copy of Training Data
    p = np.zeros([q, q])  # Design Matrix
    err = np.zeros([q, 1])  # Store Errors associated with columns of p
    c = np.zeros([q, r])  # Store neurons centers
    phi_col = np.zeros([q, 1])  # Array to update design matrix for actual error calculations
    showiterinfo = True  # Display iterations
    #  Create Lists for Plots
    x_data = []  # Epochs
    y_data = []  # MSE
    #  Create Design Matrix
    for i in range(q):
        for j in range(q):
            p[j][i] = rbf(x[j:j + 1, 0:r], x[i:i + 1, 0:r], theta)  # RBF function call
    #  Find column vector in p with maximum error
    for k in range(q):
        a = p[0:q, k:k + 1]
        g = np.dot(np.transpose(a), y_train) / np.dot(np.transpose(a), a)
        err[k] = pow(g, 2) * np.dot(np.transpose(a), a) / np.dot(np.transpose(y_train), y_train)
    maxerr = err.max()
    j = err.argmax()
    wj = p[0:q, j:j + 1]
    p = np.delete(p, j, 1)
    err = np.delete(err, j, 0)
    c[0:1, 0:r] = x[j:j + 1, 0:r]
    x = np.delete(x, j, 0)
    #  Calculate mean-squared error of the network
    for i in range(q):
        # Update Design Matrix for Training Data
        phi_col[i][0] = rbf(x_train[i:i + 1, 0:r], c[0:1, 0:r], theta)
    bias = np.ones([q, 1])  # Add bias term
    phi = np.concatenate((bias, phi_col), axis=1)
    lw = np.dot(np.linalg.pinv(phi[0:q, 0:2]), y_train)
    mse = np.mean(np.square(np.dot(phi[0:q, 0:2], lw) - y_train))
    x_data.append(1)
    y_data.append(mse)
    # Display iteration info
    if showiterinfo:
        print("Epoch: " + str(1) + " MSE: " + str(mse))
    #  Main loop
    for it in range(q - 1):
        alpha = np.dot(np.transpose(wj), p) / np.dot(np.transpose(wj), wj)
        p = p - np.dot(wj, alpha)
        e = (np.shape(p))[1]
        #  Find column vector in p with maximum error
        for k in range(e):
            a = p[0:q, k:k + 1]
            g = np.dot(np.transpose(a), y_train) / np.dot(np.transpose(a), a)
            err[k] = pow(g, 2) * np.dot(np.transpose(a), a) / np.dot(np.transpose(y_train), y_train)
        maxerr = err.max()
        j = err.argmax()
        wj = p[0:q, j:j + 1]
        p = np.delete(p, j, 1)
        err = np.delete(err, j, 0)
        c[it + 1:it + 2, 0:r] = x[j:j + 1, 0:r]
        x = np.delete(x, j, 0)
        #  Calculate mean-squared error of the network
        for i in range(q):
            phi_col[i][0] = rbf(x_train[i:i + 1, 0:q - 1], c[it + 1:it + 2, 0:r], theta)
        # Update Design Matrix for Training Data
        phi = np.concatenate((phi, phi_col), axis=1)
        lw = np.dot(np.linalg.pinv(phi[0:q, 0:it + 3]), y_train)
        mse = np.mean(np.square(np.dot(phi[0:q, 0:it + 3], lw) - y_train))
        x_data.append(it + 2)
        y_data.append(mse)
        # Display iteration info
        if showiterinfo:
            print("Epoch: " + str(it + 2) + " MSE: " + str(mse))
        #  Check for convergence
        if mse <= funtol:
            print("Algorithm Stopped:Mean-squared error less than specified tolerance")
            for j in range(q-1, it+1, -1):
                c = np.delete(c, j, 0)
            break
        if it == (q - 2):
            print("Algorithm Stopped: Maximum number of neurons reached")
            break
    #  Plot for Best and Average Cost Function with Iterations
    plt.plot(x_data, y_data, '-b')
    plt.xlabel('Epochs')
    plt.ylabel("MSE")
    plt.grid()
    plt.legend(["Training"], loc="upper right", frameon=False)
    plt.show()
    return c, lw
