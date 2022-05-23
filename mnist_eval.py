from time import time
import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from numpy import linalg as LA
import matlab.engine
from scipy.io import savemat

from utils.mnist import NeuralNet, NeuralNetToy, NeuralNet2, NeuralNet2_128, NeuralNet2_256, NeuralNet2_512, NeuralNet3, NeuralNet7, NeuralNet8, NeuralNetToy2
from utils.naiveNorms import NaiveNorms
from utils.solver import GL_Solver


#Model training globals
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28 * 28
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Import MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

def models_from_parser(args):
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('mats'):
        os.makedirs('mats')

    model_path = "mnist_model_toy"
    mat_path = "mnist_weight_toy.mat"
    model = NeuralNetToy(input_size, num_classes).to(device)

    if args.model == "net2_8":
        model_path = "mnist_model_toy2"
        mat_path = "mnist_weight_toy2.mat"
        model = NeuralNetToy2(input_size, num_classes).to(device)
    elif args.model == "net2":
        model_path = "mnist_model2"
        mat_path = "mnist_weight_model2.mat"
        model = NeuralNet2(input_size, num_classes).to(device)
    elif args.model == "net2_128":
        model_path = "mnist_model2_128"
        mat_path = "mnist_weight_model2_128.mat"
        model = NeuralNet2_128(input_size, num_classes).to(device)
    elif args.model == "net2_256":
        model_path = "mnist_model2_256"
        mat_path = "mnist_weight_model2_256.mat"
        model = NeuralNet2_256(input_size, num_classes).to(device)
    elif args.model == "net3":
        model_path = "mnist_model3"
        mat_path = "mnist_weight_model3.mat"
        model = NeuralNet3(input_size, num_classes).to(device)
    elif args.model == "net7":
        model_path = "mnist_model7"
        mat_path = "mnist_weight_model7.mat"
        model = NeuralNet7(input_size, num_classes).to(device)
    elif args.model == "net8":
        model_path = "mnist_model8"
        mat_path = "mnist_weight_model8.mat"
        model = NeuralNet8(input_size, num_classes).to(device)
    return "models/"+model_path, "mats/"+mat_path, model

def train_model(model_path, mat_path, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train(criterion, optimizer)

    torch.save(model.state_dict(),model_path)

    model.evaluate()
    weights = []
    for layer in model.modules():
        if type(layer) is nn.Linear:
            weight = layer.weight.cpu().detach().numpy()    
            weights.append(weight)

    data = {'weights':np.array(weights, dtype=np.object)}
    savemat(mat_path, data)

def main(args):
    model_path, mat_path, model = models_from_parser(args)

    if not os.path.exists(model_path):
        train_model(model_path, mat_path, model)
        
    model.load_state_dict(torch.load(model_path))
    weights = []
    for layer in model.modules():
        if type(layer) is nn.Linear:
            weight = layer.weight.cpu().detach().numpy()    
            weights.append(weight)
            #print(weight.shape)
    weights_num = len(weights)
    classes = weights[-1].shape[0]
    if len(weights) > 2 and args.method == "sdp":
        print("We will use the dual program to estimate the FGL")
        args.methods = "sdp_dual"
    start_time = time()

    if args.method == "product":
        jcb_norm = 1
        for i in range(weights_num-1):
            if args.l2:
                jcb_norm *= LA.norm(weights[i].transpose(), 2)
            else:
                jcb_norm *= LA.norm(weights[i].transpose(), 1)
        norms = []
        for i in range(classes):
            vec = weights[-1][i, :]
            if args.l2:
                norms.append(LA.norm(vec.transpose(), 2)*jcb_norm)
            else:
                norms.append(LA.norm(vec.transpose(), 1)*jcb_norm)
        print("Matrix Product Norms are: ", norms)

    if args.method == "brute" and (args.model == "net2_8" or args.model == "net2_16"):
        NN = NaiveNorms(weights[0], weights[1])
        if args.l2:
            vec = NN.BFNorms(2)
        else:
            vec = NN.BFNorms(1)
        print("Brute Force Norms are: ", vec)
        
    if args.method == "sampling":
        lbs = []
        for i in range(classes):
            noise = torch.rand(200000, input_size) - torch.ones(input_size)/2
            center = torch.zeros(input_size)
            x = (10 * noise + center).to(device)
            x.requires_grad = True
            x.retain_grad()
            model(x)[:,i].sum().backward()
            if args.l2:
                norms = torch.norm(x.grad, p=2, dim=1)
            else:
                norms = torch.norm(x.grad, p=1, dim=1)
            lbs.append(torch.max(norms).item())
        print("Sampling Lower Bounds are: ", lbs)

    
    if args.method == "sdp":
        eng = matlab.engine.start_matlab()
        eng.addpath(r'matlab_solver')
        if args.l2:
            lcs = eng.GeoLIP(mat_path, '2', False)
        else:
            lcs = eng.GeoLIP(mat_path, 'inf', False)
        print("SDP Norms are: ", lcs)

    if args.method == "sdp_dual":
        eng = matlab.engine.start_matlab()
        eng.addpath(r'matlab_solver')
        if args.l2:
            lcs = eng.GeoLIP(mat_path, '2', True)
        else:
            lcs = eng.GeoLIP(mat_path, 'inf', True)
        print("SDP Norms are: ", lcs)
        
    if args.method == "sdp_py":
        gs = GL_Solver(weights=weights, dual=True, approx_hidden = False, approx_input=False)
        print("CVXPY norms are:", gs.sdp_norm(parallel=False))   
    
    print(f'Total time: {float(time() - start_time):.5} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", 
        nargs='?', 
        const="net2_16", 
        default="net2_16", 
        choices=['net2_8', 'net2_16', 'net2', 'net2_128', 'net2_256', 'net3', 'net7', 'net8'], 
        help="which model to use")
    parser.add_argument("--method", 
        nargs='?', 
        const="product", 
        default="product", 
        choices=['brute', 'product', 'sdp', 'sdp_dual', 'sdp_py', 'sampling'], 
        help="which method to use")
    parser.add_argument('--l2', 
        action='store_true', 
        help="estimate l_2 FGL or l_inf")

    args = parser.parse_args()
    
    main(args)