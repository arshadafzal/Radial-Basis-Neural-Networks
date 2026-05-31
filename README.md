# OLS-RBNN

This repository provides a simple and educational implementation of Orthogonal Least Squares learning for Radial Basis Function Neural Networks.

The main purpose of this repository is to demonstrate how Orthogonal Least Squares can be used to select important radial basis functions and construct a compact RBF neural network for regression and function approximation problems.

## Highlights

Key highlights include:

- Implementation of Radial Basis Function Neural Networks
- Orthogonal Least Squares learning for basis function selection
- Simple and readable Python code structure
- Suitable for regression and function approximation problems
- Useful for academic learning, classroom teaching, and research purposes
- Easy to modify for different datasets and engineering applications
- External data can be imported using libraries such as Pandas or NumPy

## Background

Radial Basis Function Neural Networks are feed-forward neural networks that use radial basis functions as activation functions in the hidden layer.

The network output can be written as:

$$
y(x) = \sum_{i=1}^{M} w_i \phi_i(x)
$$

where:

- $y(x)$ is the predicted output
- $w_i$ are the output weights
- $\phi_i(x)$ are radial basis functions
- $M$ is the number of selected basis functions

A commonly used radial basis function is the Gaussian function:

$$
\phi_i(x) =
\exp\left(
-\frac{\|x-c_i\|^2}{\theta_i^2}
\right)
$$

where:

- $c_i$ is the center of the radial basis function
- $\theta_i$ is the width or spread parameter

## Orthogonal Least Squares Learning

Orthogonal Least Squares is used to select the most significant basis functions from a candidate set.

Instead of using all possible basis functions, OLS selects the basis functions that contribute most to reducing the prediction error. This helps create a compact and efficient RBF neural network.

The method is useful because it can:

- reduce the number of hidden neurons,
- improve model interpretability,
- reduce unnecessary complexity,
- improve computational efficiency.

## Algorithm Reference

The Orthogonal Least Squares learning algorithm used in this repository is based on the original work by Chen, Cowan, and Grant:

```bibtex
@article{chen1991orthogonal,
  author  = {Chen, S. and Cowan, C. F. N. and Grant, P. M.},
  title   = {Orthogonal Least Squares Learning Algorithm for Radial Basis Function Networks},
  journal = {IEEE Transactions on Neural Networks},
  volume  = {2},
  number  = {2},
  pages   = {302--309},
  year    = {1991},
  doi     = {10.1109/72.80341}
}
```

Please cite the original paper if you use the Orthogonal Least Squares learning algorithm for Radial Basis Function Neural Networks.

## Applications

This implementation can be useful for:

- Function approximation
- Regression problems
- Surrogate modeling
- Engineering design optimization
- Scientific machine learning
- Data-driven modeling of physical systems


## Citation

If you use this repository for academic, teaching, or research purposes, please cite both the original algorithm paper and this repository.

```bibtex
@software{afzal_2026_ols_rbf_neural_network,
  author = {Afzal, Arshad},
  title = {OLS-RBF-Neural-Network: Orthogonal Least Squares Learning for Radial Basis Function Neural Networks},
  year = {2026},
  url = {https://github.com/arshadafzal/OLS-RBF-Neural-Network}
}
```

## Author

Arshad Afzal
