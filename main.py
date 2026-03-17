import torch
import matplotlib.pyplot as plt
import numpy as np

from src.ansatz import WaveFunctionNN
from src.sampler import Sampler
from src.vmc import TrainVMC

def main():
    HiddenDim = 32
    NumWalkers = 2000
    StepSize = 2
    NumEpochs = 400
    LearningRate = 0.005
    MCSteps = 20

    torch.manual_seed(67) #Haha

    ansatz = WaveFunctionNN(HiddenDim=HiddenDim)
    sampler = Sampler(ansatz, StepSize=StepSize, WalkerNum=NumWalkers)

    EnergyHistory = TrainVMC(
        Ansatz= ansatz,
        Sampler= sampler,
        NumEpochs= NumEpochs,
        LearnRate= LearningRate,
        SampleSteps= MCSteps
    )

    #Just a ton of plotting, stunning
    plt.figure(figsize=(10,6))
    plt.plot(EnergyHistory, label='Neural Network Energy', color='blue', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', label='True Ground State')
    plt.title("Variational Monte Carlo: 1D Harmonic Oscillator")
    plt.xlabel("Training Epochs")
    plt.ylabel("Expectation Value of Energy $\langle E \\rangle$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("HarmonicOscillatorEnergyConvergence.png")
    plt.show()

    PlotWavefunction(ansatz)

def PlotWavefunction(ansatz):
    #Create a grid of x values to plot over
    Xnp = np.linspace(-5, 5, 1000)

    #Convert to pytorch tensor
    X_Tensor = torch.tensor(Xnp, dtype=torch.float32).view(-1, 1)

    #Get neural net predicted amplitude
    with torch.no_grad():
        PsiNN = ansatz(X_Tensor).squeeze().numpy()

    #Some light numerical integration to normalise the wavefunction
    dx = Xnp[1] - Xnp[0]
    area = np.sum(PsiNN**2)*dx
    NormalisedPsiNN = PsiNN / np.sqrt(area)

    #Plot the analytic solution of the harmonic oscillator
    RealPsi = (1/np.pi**0.25) * np.exp(-0.5 * Xnp**2)
    plt.figure(figsize=(10, 6))
    plt.plot(Xnp, RealPsi, 'r--', linewidth=3, label="Analytical $\psi_0(x)$")
    
    #Plot the NN solution, gorgeous
    plt.plot(Xnp, NormalisedPsiNN, 'b-', linewidth=2, alpha=0.8, label="Neural Network")
    plt.title("Neural Network vs. Analytic Physics")
    plt.xlabel("Position ($x$)")
    plt.ylabel(r"Amplitude $\psi(x)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("HarmonicOscillatorWavefunction.png")
    plt.show()

main()
