import torch
import torch.nn as nn

#instead of guessing/calculating the form of the wavefunction we will use a neural network as its representation

class WaveFunctionNN(nn.Module):

    def __init__(self, HiddenDim=32):
        super().__init__()

        self.net = nn.Sequential(   #Sequential passes the data through each following layer in order
            nn.Linear(1, HiddenDim),         #Input layer, a linear transformation, takes the single input and broadcasts to HiddenDim
            nn.Tanh(),                       #Activation function, 
            nn.Linear(HiddenDim, HiddenDim), #Hidden layer, takes the 32 signals and mixes them and outputs 32 new signals, builds the complexity
            nn.Tanh(),                       #Smooth, non-linear function to the outputs of hidden layer
            nn.Linear(HiddenDim, 1)          #Amplitude of the wavefunction at given position
        )

    def forward(self, x):
        psi = torch.exp(self.net(x) -0.5 * x**2) #Passing positions through the network
        return psi