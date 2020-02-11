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
#from data_parse import Data
#from nn_gen import Net
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="Generative model of 1d classical ising chain in PyTorc")
    parser.add_argument("params",metavar="params/param_file_name.json",type=str)
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)

    num_epochs = hyper["epochs"]

    # Training loop
    for epoch in range(1, num_epochs + 1):

        # High verbosity report in output stream
        if args.v>=2:
            if not ((epoch + 1) % hyper["display epochs"]):
                print('Epoch [{}/{}]'.format(epoch, num_epochs) +\
                    '\t ')

                