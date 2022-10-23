import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from solver import Solver
from net import PGGenerator
import numpy as np


class GLOnet():
    def __init__(self, params):

        # GPU 
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        # construct the generator
        self.generator = self._init_generator(params)
        if self.cuda: 
            self.generator.cuda()
        self.num_layers = params.num_layers
        
        # construct the optimizer
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=params.lr) #betas = (params.beta1, params.beta2), weight_decay = params.weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = params.step_size, gamma = params.gamma)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        # training parameters
        self.dim = params.dim
        self.num_iter = params.num_iter
        self.batch_size = params.batch_size
        self.sigma = params.sigma

        self.z = (torch.randn(params.batch_size, 64, requires_grad=True)).type(self.dtype)

        # FoM solver
        self.solver = Solver(params)
    
        # tranining history
        self.loss_history = []
        self.FoM_history = []
        self.x_history = []
        self.output_dir = params.output_dir
    
    def _init_generator(self, params):
        return PGGenerator(64, params.dim, params.num_layers)
 

        
    def train(self):
        self.generator.train()
            
        # training loop
        with tqdm(total=self.num_iter) as t:
            for it in range(self.num_iter):

                # normalized iteration number
                normIter = it / self.num_iter
                self.update_alpha(normIter)
                # sample z
                z = self.sample_z(self.batch_size)

                # Generate a batch of devices. 
                x = self.generator(z)

                #print(x)
                # calculate FoM and gradients 
                FoM, Grad = self.solver.run(x, grad=True) 
                #Grad = Grad * torch.std(x.detach(), dim=0, keepdim=True)
                #Grad = Grad / (torch.std(torch.abs(Grad), dim=0, keepdim=True) + 1e-2)
                #Grad = Grad/torch.norm(Grad, dim=1, keepdim=True)
                #self.sigma = FoM.mean()
                # free optimizer buffer 
                self.optimizer.zero_grad()

                # construct the loss 
                loss = self.loss_function(x, FoM, Grad)

                # record history
                self.loss_history.append(loss.detach())
                self.FoM_history.append(torch.log(FoM.mean().detach())/np.log(10))
                self.x_history.append(torch.log(torch.mean(torch.abs(x-self.solver.xopt/self.solver.ub), dim=0).mean().detach())/np.log(10))
            
                if it % 20 == 0:
                    xdistance = torch.mean(x-self.solver.xopt/self.solver.ub, dim=0).detach()
                    plt.figure(figsize = (6, 5))
                    plt.hist(xdistance.view(-1), bins = 20)
                    plt.xlabel('x - xopt', fontsize=18)
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.savefig(self.output_dir+'/diversity/iter{}.png'.format(it))
                    plt.close()

                # train the generator
                loss.backward()
                self.optimizer.step()

                if it % 20 == 0:
                    self.scheduler.step()
                self.viz_training()
                # update progress bar
                t.update()
    

    def evaluate(self, batch_size):
        z = self.sample_z(batch_size)
        # Generate a batch of devices. 
        x = self.generator(z)

        # calculate efficiencies and gradients 
        FoM = self.solver.run(x, grad=False, is_norm=False) 

        # plot histogram
        plt.figure(figsize = (6, 5))
        plt.hist(FoM.view(-1), bins = 10)
        plt.xlabel('FoM', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(self.output_dir+'/hist.png')
        plt.close()
        return x, FoM

        
    def sample_z(self, batch_size):
        return (torch.randn(batch_size, 64)).type(self.dtype)

    
    def loss_function(self, x, FoM, Grad):
        # x: (B, M)
        # FoM: (B, 1)
        # Grad: (B, M)
        if self.cuda:
            FoM = FoM.cuda()
            Grad = Grad.cuda()
        loss =  x * Grad * (1./self.sigma) * (torch.exp(-FoM/self.sigma))
        return torch.mean(loss.view(-1))


    def viz_training(self):
        plt.figure(figsize = (6, 5))
        plt.plot(self.FoM_history)
        plt.ylabel('Average FoM', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.ylim([0, 1])
        plt.savefig(self.output_dir+'/FoM.png')
        plt.close()

        plt.figure(figsize = (6, 5))
        plt.plot(self.x_history)
        plt.ylabel('Average |x - xopt|', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.ylim([0, 1])
        plt.savefig(self.output_dir+'/x.png')
        plt.close()

        plt.figure(figsize = (6, 5))
        plt.plot(self.loss_history)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #plt.ylim([0, 1])
        plt.savefig(self.output_dir+'/loss.png')
        plt.close()
        

    def update_alpha(self, normIter):
        di = 1./(self.num_layers+1)
        for i in range(self.num_layers):
            alpha = torch.sigmoid(torch.tensor(4.*(normIter/di - i + 0.5)))
            self.generator.update_alpha(i, alpha)







        