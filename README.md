# GeoLIP
This is the anonymous submission to neurips.

## Dependencies
To run this project, you will need the [MOSEK](https://www.mosek.com/) solver. We provide two implementations: one is based on **MATLAB** and the [CVX](http://cvxr.com/cvx/) system, and the other is based on **CVXPY** package. It is highly recommended to use the **MATLAB** version when it is possible.

Other dependencies include: `PyTorch`, `NumPy`, `SciPy`, `CVXPY`. To use the MATLAB implementation, you will also need to install [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

## Instructions
`python3 mnist_eval.py --model toy --method sdp_dual`

There are 8 different choices for network structures, specified using `--model`. They correpond to different architectures we used in the paper submission:
1. `net2_8` is a two-layer network with 8 hidden units;
2. `net2_16` is a two-layer network with 16 hidden units;
3. `net2` is a two-layer network with 64 hidden units;
4. `net2_128` is a two-layer network with 128 hidden units;
5. `net2_256` is a two-layer network with 256 hidden units;
6. `net3` is a 3-layer network, and each hidden layer has 64 units;
7. `net7` is a 7-layer network, and each hidden layer has 64 units;
8. `net8` is a 8-layer network, and each hidden layer has 64 units;


There are 6 methods to compute the FGL, specified using `--method`:
1. `brute` is a brute-force search for the extreme activation pattern. Because it take exponential-time to run, we can only work with two-layer networks with 8 or 16 units in our experiments;
2. `product` is the matrix-norm-product method, which is a naive upper bound of the FGL;
3. `sdp` is the GeoLIP in the natural relaxation form. Notice that this only applies for two-layer networks.
4. `sdp_dual` is the GeoLIP in the dual form.
5. `sdp_py` is our CVXPY implementation of GeoLIP.
6. `sampling` is a random sampling in the input space and calculate its gradient norm of each point. This is a lower bound of **true** Lipschitz constant of the neural network, and thus a lower bound of FGL.

If the model has not been trained, the script will first train the model, and save the model and weights `.mat` file. Otherwise, the script will directly load the trained model, and compute the FGL of the neural network.

