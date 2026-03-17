# Neural Network Quantum States (NQS) with PyTorch
Implements a simple Variational Monte Carlo (VMC) method with PyTorch to solve for ground state energies and wavefunctions of various 1D Quantum Systems.

## The Method
### Ansatz (Neural Network)
Instead of solving boring DEs, the wavefunction is parameterised as a small Multilayer Perceptron with a twice-differentiable activation function (tanh). The network takes the position x and spits out the (unnormalised) probability amplitude. I also used a Gaussian envelope to enforce that as x tends to larger values the wavefunction tends to 0.

### Sampler (The Walking Dead)
A whole bunch of walkers (intentional zombie reference) explore the space, spending more time where probability density is greater, think of it like the zombies wandering to wherever Rick is most likely to be.

### Hamiltonian (I dont have anything fun to say)
Calculates the local energy at each point. Because derivatives are difficult when there's no function apparently, I used PyTorch's automatic differentiation engine (autograd) to compute kinetic energy. The formula at the bottom of this file can be altered to represent any arbitrary potential, it's set to the traditional harmonic oscillator.

### Variational Principle
The VP states that if you guess any wavefunction, the average energy of the guess will always be greater or equal to the true ground state. Convenient. Because of this the progam is just optimising itself to go tolower energy, each time the energy is calculated you tweak the weights to get slightly lower. It can't go any lower than the lowest, because quantum mechanics apparently.

A known loss function is used to find the gradients, then PyTorch's Adam optimiser changes the NN weights.

## Structure
ansatz.py - Defines parameterised wavefunction

sampler.py - The vectorised walking dead

hamiltonian.py - Finds local energy with auto differentiation

vmc.py - The training loop and the loss calculations

main.py - Just runs the thing and plots it innit





