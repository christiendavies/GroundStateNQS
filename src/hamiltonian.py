import torch

def ComputeLocalEnergy(Ansatz, x):
    #We need spatial derivatives, this will tell PyTorch to track over input positions
    x.requires_grad_(True)

    #Get wavefunction at x
    psi = Ansatz(x)

    #Calculate first derivative
    dpsi_dx = torch.autograd.grad(          #Computes sum of gradients
        psi, x,                             #derivative of psi wrt x
        grad_outputs=torch.ones_like(psi),  #Computes derivative of each walker separately
        create_graph=True                   #We need the second derivative so need the graph used to calculate the first derivative
    )[0]                                    #grad returns a tuple, but only need the actual gradient

    #Calculate second derivative
    d2psi_dx2 = torch.autograd.grad(
        dpsi_dx, x,
        grad_outputs=torch.ones_like(dpsi_dx),
        create_graph=True
    )[0]

    KineticE = -0.5 * (d2psi_dx2 / psi) #T = -1/2 d2/dx2        
    
    PotentialE = 0.5 * (x**2)     #V = 1/2 x2   This line can be changed to represent the various potentials.

    LocalE = KineticE + PotentialE

    return LocalE