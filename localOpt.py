import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver

class LocalOpt():
    def __init__(self, params, init_x, lr=0.01):
        self.solver = Solver(params)
        self.x = nn.Parameter(init_x)

        self.optimizer = torch.optim.Adam([self.x], lr=lr) #betas = (params.beta1, params.beta2), weight_decay = params.weight_decay)
        #self.optimizer = torch.optim.LBFGS([self.x], lr=lr)
        self.FoM_history = []
        self.output_dir = params.output_dir


    def optimize(self, it):

        for i in range(it):
            self.optimizer.zero_grad()

            FoM = self.solver.forward(self.x)

            loss = FoM.sum()
            loss.backward()
            self.optimizer.step()

            self.FoM_history.append(FoM.mean().detach())


        plt.figure(figsize = (6, 5))
        plt.plot(self.FoM_history)
        plt.ylabel('Average FoM', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.ylim([0, 1])
        plt.savefig(self.output_dir+'/FoM_local.png')
        plt.close()


    def optima(self):
        FoM = self.solver.forward(self.x, is_norm=False)
        _, indices = torch.sort(FoM)

        FoM_opt = FoM[indices[0]]
        x_opt = self.x[indices[0]]

        return x_opt, FoM_opt
