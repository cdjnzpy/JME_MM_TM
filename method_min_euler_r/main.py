import argparse, os
#from scipy.fft import fftn
import torch, random
import numpy as np
from data import data
from model import model
from train import train
#from utils import utils
import matplotlib.pyplot as plt


path = os.path.dirname(__file__)

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    para_arg = add_argument_group('Parameters')  
    para_arg.add_argument('--beta', type=float, default=0.9)
    para_arg.add_argument('--rate', type=float, default=1.04)
    para_arg.add_argument('--gamma', type=float, default=2.0)
    para_arg.add_argument('--sigma', type=float, default=0.1)
    para_arg.add_argument('--T_period', type=int, default=2)
    para_arg.add_argument('--T_period_test', type=int, default=100)
    para_arg.add_argument('--W_range', type=list, default=[0.1,4])

    data_arg = add_argument_group('Data')  
    data_arg.add_argument('--draw_grid_each_epoch', type=int, default=1)
    data_arg.add_argument('--draw_sample_test', type=int, default=64)
    data_arg.add_argument('--draw_shocks_each_epoch', type=int, default=128)

    model_arg = add_argument_group('Model')
    model_arg.add_argument("--input_dim", type=int, default = 1+1)
    model_arg.add_argument("--output_dim", type=int, default = 1+1)
    model_arg.add_argument("--hidden_layer", type=list, default = [64,64,64])
    model_arg.add_argument("--save_model_path", type=str, default = path+"\\save\\model\\Phi_h_Model_64X64.pt")
    model_arg.add_argument("--save_data_path", type=str, default = path+"\\save\\data_64X64.npy")

    model_arg.add_argument("--save_test_data_path", type=str, default = path+"\\save\\test_data.npy")
    model_arg.add_argument("--save_learning_pho", type=str, default = path+"\\save\\learning_pho\\")

    train_arg = add_argument_group('Train')
    train_arg.add_argument("--train_epoch", type=int, default = 50000)
    train_arg.add_argument("--learning_rate", type=float, default = 0.001)
    

    learn_arg = add_argument_group('Learn')
    train_arg.add_argument('--random_seed', type=int, default=1)

    args, unparsed = get_args()
    set_seed(args.random_seed)

    Data_set = data.Data_set(args)
    Trainer = train.Trainer(args,Data_set)
    Trainer.train()

    #plot_different
    