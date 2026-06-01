import numpy as np
import matplotlib.pyplot as plt


def rbf(x, c, theta):
    """
    Gaussian radial basis function.

    Parameters
    ----------
    x : Input data point.
    c : Center of the radial basis function.
    theta : float
        Spread/width parameter of the RBF.

    Returns
    -------
    value : float
        RBF value between x and c.
    """

    # Squared Euclidean distance between input point and center
    dist_sq = np.sum((x - c) ** 2)

    # Gaussian RBF
    value = np.exp(-(theta ** 2) * dist_sq)

    return value


def rbnn(x_train, y_train, theta, funtol):
    """
    Train a Radial Basis Function Neural Network using an incremental
    center selection strategy.

    The algorithm selects centers from the training data one by one.
    At each step, the center that gives the largest contribution to
    reducing the output error is selected.

    Parameters
    ----------
    x_train : ndarray, shape (q, r)
        Training input data.
        q = number of training samples
        r = number of input variables/features

    y_train : ndarray, shape (q, 1)
        Training output data.

    theta : float
        RBF spread parameter.

    funtol : float
        Stopping tolerance for mean squared error.

    Returns
    -------
    c : ndarray
        Selected RBF centers.

    lw : ndarray
        Linear weights of the trained RBF network.
        The first weight corresponds to the bias term.
    """
    # q = number of training samples
    # r = number of input features
    q = x_train.shape[0]
    r = x_train.shape[1]

    # Make a copy of the training inputs
    # This copy will be reduced as centers are selected
    x = x_train.copy()
    # ---------------------------------------------------------
    # Initialize arrays
    # ---------------------------------------------------------

    # Design matrix containing RBF values for all candidate centers
    p = np.zeros((q, q))

    # Error contribution associated with each candidate column
    err = np.zeros((q, 1))

    # Array to store selected centers
    c = np.zeros((q, r))

    # Temporary column used to update the final design matrix
    phi_col = np.zeros((q, 1))

    # Whether to print iteration information
    showiterinfo = True

    # Lists for plotting MSE history
    x_data = []
    y_data = []
    #  Build the initial RBF design matrix
    for i in range(q):
        for j in range(q):
            p[j][i] = rbf(x[j:j + 1, 0:r], x[i:i + 1, 0:r], theta)  # RBF function call
    #  Select the first RBF center
    for k in range(q):
        a = p[0:q, k:k + 1]
        g = np.dot(np.transpose(a), y_train) / np.dot(np.transpose(a), a)
        err[k] = pow(g, 2) * np.dot(np.transpose(a), a) / np.dot(np.transpose(y_train), y_train)
    # Find the column with maximum error contribution
    # maxerr = err.max()
    j = err.argmax()
    wj = p[0:q, j:j + 1]
    # Remove selected column from candidate matrix
    p = np.delete(p, j, 1)
    err = np.delete(err, j, 0)
    # Store the selected center
    c[0:1, 0:r] = x[j:j + 1, 0:r]
    x = np.delete(x, j, 0)
    # ---------------------------------------------------------
    # Calculate MSE using first selected center
    # ---------------------------------------------------------
    for i in range(q):
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

    # ---------------------------------------------------------
    # Main loop for adding more RBF neurons
    # ---------------------------------------------------------
    for it in range(q - 1):
        alpha = np.dot(np.transpose(wj), p) / np.dot(np.transpose(wj), wj)
        p = p - np.dot(wj, alpha)
        e = (np.shape(p))[1]
        # -----------------------------------------------------
        # Find next column with maximum error contribution
        # -----------------------------------------------------
        for k in range(e):
            a = p[0:q, k:k + 1]
            g = np.dot(np.transpose(a), y_train) / np.dot(np.transpose(a), a)
            err[k] = pow(g, 2) * np.dot(np.transpose(a), a) / np.dot(np.transpose(y_train), y_train)
        # maxerr = err.max()
        j = err.argmax()
        wj = p[0:q, j:j + 1]
        p = np.delete(p, j, 1)
        err = np.delete(err, j, 0)
        c[it + 1:it + 2, 0:r] = x[j:j + 1, 0:r]
        x = np.delete(x, j, 0)
        # -----------------------------------------------------
        # Update final RBF design matrix
        # -----------------------------------------------------
        for i in range(q):
            phi_col[i][0] = rbf(x_train[i:i + 1, 0:q - 1], c[it + 1:it + 2, 0:r], theta)
        phi = np.concatenate((phi, phi_col), axis=1)

        lw = np.dot(np.linalg.pinv(phi[0:q, 0:it + 3], rcond=1e-12), y_train)
        mse = np.mean(np.square(np.dot(phi[0:q, 0:it + 3], lw) - y_train))

        x_data.append(it + 2)
        y_data.append(mse)

        # Display iteration info
        if showiterinfo:
            print("Epoch: " + str(it + 2) + " MSE: " + str(mse))
        #  Check for convergence
        if mse <= funtol:
            print("Algorithm Stopped:Mean-squared error less than specified tolerance")
            # Keep only selected centers
            c = c[0:it + 2, :]

            break
        if it == (q - 2):
            print("Algorithm Stopped: Maximum number of neurons reached")
            break
    # ---------------------------------------------------------
    # Plot training MSE history
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(x_data, y_data, 'o')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("RBF Neural Network Training Error")
    plt.grid(True)
    plt.show()

    return c, lw
