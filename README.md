# GeoLIP
This is the anonymous submission to neurips.

## Dependencies
To run this project, you will need the [MOSEK](https://www.mosek.com/) solver. We provide two implementations: one is based on **MATLAB** and the [CVX](http://cvxr.com/cvx/) system, and the other is based on **CVXPY** package. It is highly recommended to use the **MATLAB** version when it is possible.

Other dependencies include: PyTorch, NumPy, SciPy, CVXPY. To use the MATLAB implementation, you will also need to install [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

## Instructions
`python3 mnist_eval.py --model toy --method sdp_dual`

There are 9 different choices for network structures, specified using `--model`.

There are different methods to compute the FGL, specified using `--method`.

If the model has not been trained, the script will first train the model, and save the model and weights `.mat` file. Otherwise, the script will directly load the trained model, and compute the FGL of the neural network.

