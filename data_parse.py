# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 3
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import numpy as np
import random as rand
import torch as torch

class Data():

    ## parses the data to be fed into our neural network
    ## file_location is the location of the txt file to be read
    ## n_test is the number of samples to set aside for the testing set
    def __init__(self,file_location,n_test):

        data = np.loadtxt(file_location,dtype=str)
        
        # Array of boolean ising data
        ising = np.empty((len(data),len(data[0])),dtype = np.float64)
        
        # index in loop
        row = 0

        # loop through our data and convert string + and - into boolean data
        for line in data:
            
            # index in loop
            col = 0
            for cell in line:

                # Convert to boolean
                if cell == "+":
                    ising[row,col] = 1.0
                else:
                    ising[row,col] = 0.0

                # increment index
                col += 1

            # increment index
            row += 1

        print(ising)

        self.data = torch.from_numpy(ising)
        #self.data.requires_grad = True
        self.len = np.size(ising[0])