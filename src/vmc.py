import torch
import torch.optim as optim
from src.hamiltonian import ComputeLocalEnergy

def TrainVMC(Ansatz, Sampler, NumEpochs=200, LearnRate=0.01, SampleSteps=10):
    #The Adam optimiser is changing the weights of the network  
    Optimiser = optim.Adam(Ansatz.parameters(), lr=LearnRate)
    EnergyHistory = []

    #Put the walkers near the origin, the sampler will naturally move them towards high prob. areas
    XWalkers = torch.randn(Sampler.WalkerNum, 1)

    for Epoch in range(NumEpochs):
        #Generate data, move the walkers around a few steps to find where the particle is most likely to be
        _, XWalkers, AcceptRate = Sampler.Sample(NumSteps=SampleSteps, InitX=XWalkers)
        
        #Take the final positions of the walkers, detach takes out the positon from whatever happened in the previous loop
        X = XWalkers.detach().requires_grad_(True)

        #Calculate energy 
        Psi = Ansatz(X)
        LocalEnergy = ComputeLocalEnergy(Ansatz, X)
        MeanEnergy = LocalEnergy.mean() #The expectation value of energy

        #Difference between local energy and average energy, this is the reward/penalty
        EnergyCentred = LocalEnergy.detach() - MeanEnergy.detach()
        #When the Loss function is differentiated it matches the analytical gradient for minimising an expectation value
        #Walkers that found lower than average energy pull the network towards them
        Loss = torch.mean(2 * EnergyCentred * torch.log(torch.abs(Psi)))


        Optimiser.zero_grad() #Removes previous loop data
        Loss.backward()       #PyTorch moves backwards from Loss to adjust the network
        Optimiser.step()      #Adam optimiser applies the adjustments 

        EnergyHistory.append(MeanEnergy.item())

        if Epoch % 20 == 0:
            print(f"Epoch {Epoch:03d} | Energy: {MeanEnergy.item():.5f} | Acceptance Rate: {AcceptRate:.2f}")

    return EnergyHistory