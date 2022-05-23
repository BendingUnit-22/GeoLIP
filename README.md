# GeoLIP
This is the anonymous submission to neurips.

## Dependencies
To run this project, you will need the [MOSEK](https://www.mosek.com/) solver. We provide two implementations: one is based on **MATLAB** and the [CVX](http://cvxr.com/cvx/) system, and the other is based on **CVXPY** package. It is highly recommended to use the **MATLAB** version when it is possible.

Other dependencies include: PyTorch, NumPy, SciPy, CVXPY. To use the MATLAB implementation, you will also need to install [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

## Instructions
### Train a neural network and store the trained network
`python3 mnist_eval.py --model toy --train`

There are 9 different choices for network structures, specified using `--model`.
### Load the trained network, and compute the FLC of the network
`python3 mnist_eval.py --model toy --method sdp_dual`

After training the network, one can load the model and choose different methods to estimate the FLC.