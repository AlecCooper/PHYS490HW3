# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 3
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import json, argparse
import torch.optim as optim
import torch as torch
import numpy as np
from data_parse import Data
from nn_gen import RBM
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Generative model of 1d classical ising chain in PyTorc")
    parser.add_argument("data", metavar="data/in.txt", type=str)
    parser.add_argument("params",metavar="params/param_file_name.json",type=str)
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)

    # Create our dataset
    print("Importing data.....")
    data = Data(args.data,hyper["num test"])
    print("Done")

    # Create our Restricted Boltzman Machine
    model = RBM(data.data)

    # KL Divergence, our loss function
    loss_func = torch.nn.KLDivLoss()
    # Stochastic Gradient Descent, our optimizer
    optimizer = optim.SGD(model.parameters(),lr=hyper["learning rate"],momentum=0.9)

    # lists to track preformance of network
    obj_vals= []

    # Number of training epochs for out training loop
    num_epochs = hyper["epochs"]

    # Training loop
    for epoch in range(1, num_epochs + 1):

        # Clear our gradient buffer
        optimizer.zero_grad()
        # Clear gradients
        model.zero_grad()

        model.forward_hidden()

        # Graph our progress
        #obj_vals.append(loss)

        # High verbosity report in output stream
        if args.v>=2:
            if not ((epoch + 1) % hyper["display epochs"]):
                print('Epoch [{}/{}]'.format(epoch, num_epochs) +\
                    '\t ')

        


    # Ouput results into a dictonary of couplers
    output = np.empty(4)
    indices = range(len(output))
    out_dict = {}
    ctr = 0
    for spin in output:
        if (ctr+1 < len(indices)):
            out_dict["%s,%s"%(indices[ctr], indices[ctr+1])] = spin
        else:
            out_dict["%s,%s"%(indices[ctr], indices[0])] = spin
        ctr +=1


    # Save file
    with open("out.json", "w") as json_file:
        json.dump(out_dict,json_file)

    print(out_dict)

                