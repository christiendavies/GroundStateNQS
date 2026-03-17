import torch

class Sampler:
    def __init__(self, Ansatz, StepSize=0.5, WalkerNum=1000):
        self.Ansatz = Ansatz
        self.StepSize = StepSize
        self.WalkerNum = WalkerNum

    def Sample(self, NumSteps, InitX=None):
        #Initialise walkers randomly
        if InitX is None:
            X = torch.randn(self.WalkerNum, 1)
        else:
            X = InitX.clone()

        Samples = []
        AcceptedMoves = 0
        
        with torch.no_grad(): #Gradient shenanigans are not needed (i dont understand it), avoids wasting memory on stuff i wont use
            #Get wavefunction amplitude
            PsiX = self.Ansatz(X) 

            for i in range(NumSteps):
                #Propose a new random step for each walker
                XNew = X + self.StepSize * torch.randn_like(X)
                PsiXNew = self.Ansatz(XNew)

                #Calculates the prob of moving to new x, if prob at new x is much greater than old x we move there
                ProbRatio = (PsiXNew**2) / (PsiX**2)

                #u is a uniform random number from 0 to 1, if PsiXNew is greater than PsiX, PsiXNew is accepted, but if lower then gives a chance of the walker exploring the lower prob. area
                u = torch.rand_like(ProbRatio)
                accept = u < ProbRatio

                #If accepted take XNew, if rejected keep X, vectorised if/else, updates all 1000
                X = torch.where(accept, XNew, X)
                PsiX = torch.where(accept, PsiXNew, PsiX)

                Samples.append(X.clone())
                AcceptedMoves += accept.sum().item()

        #How often moves were accepted, would expect 50%
        AcceptRate = AcceptedMoves / (NumSteps*self.WalkerNum)

        return torch.stack(Samples), X, AcceptRate



