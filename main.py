from GLOnet import GLOnet
from solver import Solver
import torch
import utils
import os
import logging
import argparse
import scipy.io as io
from localOpt import LocalOpt
# parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='results', help="folder to save results")
parser.add_argument('--func', default='1', help="function number")

if __name__ == '__main__':
    args = parser.parse_args()
    # initialize params
    json_path = os.path.join(args.output_dir,'params','Params_f{}.json'.format(args.func))
    if os.path.isfile(json_path):
        params = utils.Params(json_path)
    else:
        params = utils.Params()

    params.func_num = int(args.func)

    output_dir = args.output_dir + '/f' + str(params.func_num)
    utils.set_logger( output_dir + '/train.log')
    

    # Add attributes to params

    params.net = 'PGNet'
    params.dim = 1000
    params.num_layers = 4 # number of resnet blocks
    params.lr = 0.002 # learning rate
    params.num_iter = 500
    params.batch_size = 200
    params.sigma = 0.5
    
    params.output_dir = output_dir 
    os.makedirs(output_dir + '/diversity', exist_ok = True)


    # solver parameters
    func_data = io.loadmat('data/f{0:02d}.mat'.format(params.func_num))
    params.xopt = torch.tensor(func_data['xopt'], dtype=torch.float).view(1, -1)
    params.ub = torch.tensor(func_data['ub']).view(1, -1)
    params.lb = torch.tensor(func_data['lb']).view(1, -1)


    if params.func_num > 2:
        params.s = [int(i) for i in func_data['s']]
        params.w = [float(i) for i in func_data['w']]
        params.R25 = torch.tensor(func_data['R25']).unsqueeze(0)
        params.R50 = torch.tensor(func_data['R50']).unsqueeze(0)
        params.R100 = torch.tensor(func_data['R100']).unsqueeze(0)
        params.P = torch.tensor(func_data['p']-1, dtype=torch.long).view(-1)
    else:
        params.s = None
        params.w = None
        params.R25 = None
        params.R50 = None
        params.R100 = None
        params.P = None

    params.m = 0
    


    # run glonet
    glonet = GLOnet(params)
    glonet.train()
    glonet.viz_training()

    x, FoM = glonet.evaluate(100)


    _, indices = torch.sort(FoM)
    logging.info("Best case: f = {}".format(FoM[indices[0]]))
    
    
    localopt = LocalOpt(params, x, 1e-3)
    localopt.optimize(1000)
    x, FoM = localopt.optima()

    logging.info("Local optimization: f = {}".format(FoM))

  


