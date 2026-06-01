import numpy as np
from rbnn import rbf


def sim(x_test, theta, c, lw):
    """
    Simulate a trained RBF neural network.

    This function predicts the output for new input data using
    the selected RBF centers and trained linear weights.

    Parameters
    ----------
    x_test : ndarray, shape (q, r)
        Test input data.
        q = number of test samples
        r = number of input variables/features

    theta : float
        Shape parameter of the Gaussian RBF.

    c : ndarray, shape (t, r)
        Selected RBF centers obtained from the training function.
        t = number of selected RBF neurons

    lw : Linear weights of the trained RBF network.
         The first weight corresponds to the bias term.

    Returns
    -------
    yp : ndarray, shape (q, 1)
        Predicted output values.
    """

    # Number of test samples
    q = x_test.shape[0]

    # Number of input variables/features
    r = x_test.shape[1]

    # Number of selected RBF centers / neurons
    t = c.shape[0]

    # -----------------------------------------------------
    # Create the RBF design matrix for test data
    # -----------------------------------------------------
    # Each column corresponds to one RBF neuron.
    # Each row corresponds to one test sample.
    phi_test = np.zeros((q, t))

    for i in range(t):          # loop over RBF centers
        for j in range(q):      # loop over test samples

            phi_test[j, i] = rbf(
                x_test[j:j + 1, 0:r],
                c[i:i + 1, 0:r],
                theta
            )

    # -----------------------------------------------------
    # Add bias column
    # -----------------------------------------------------
    bias = np.ones((q, 1))

    # Final test design matrix:
    phi_test = np.concatenate((bias, phi_test), axis=1)

    # -----------------------------------------------------
    # Predict output
    # -----------------------------------------------------
    yp = np.dot(phi_test, lw)

    return yp