import numpy as np
import matplotlib.pyplot as plt
import torch
from model.model import NN_net_phi_h_eval
import os

path = os.path.dirname(__file__)

def load_model(load_path, Model):
    Model.load_state_dict(torch.load(load_path))
    return Model

def plot_policy(Model):
    #输出policy
    ws = []
    cs = []
    for i in range(400):
        step = (4-0.1)/400
        w_grid = 0.1+ i*step

        y = 0
        
        input_data = torch.tensor([y,w_grid]).type(torch.FloatTensor)

        c_w_h = Model(input_data)
        c_w_ratio = 1/(1+torch.exp(-c_w_h[0]))
        #h = torch.exp(c_w_h[1])
        
        c_ = c_w_ratio * w_grid
            
        c = c_.detach().numpy().tolist()

        ws.append(w_grid)
        cs.append(c)

    return ws, cs
def plot_figure(args):
    #8
    data_8 = np.load(path+"\\save\\data_8X8.npy")
    FB_8 = data_8[0]
    utility_8 = data_8[1]

    #16
    data_16 = np.load(path+"\\save\\data_16X16.npy")
    FB_16 = data_16[0]
    utility_16 = data_16[1]

    #32
    data_32 = np.load(path+"\\save\\data_32X32.npy")
    FB_32 = data_32[0]
    utility_32 = data_32[1]

    #64
    data_64 = np.load(path+"\\save\\data_64X64.npy")
    FB_64 = data_64[0]
    utility_64 = data_64[1]

    len_x = list(range(len(utility_8)))

    #plot choice
    
    Phi_h_Model = NN_net_phi_h_eval(args, 2, [8,8,8], 3)

    load_path = path + "\\save\\model\\Phi_h_V_Model_8X8.pt"
    Model = load_model(load_path,Phi_h_Model)
    _,cs_1 = plot_policy(Model)

    Phi_h_Model = NN_net_phi_h_eval(args, 2, [16,16,16], 3)
    load_path = path + "\\save\\model\\Phi_h_V_Model_16X16.pt"
    Model = load_model(load_path,Phi_h_Model)
    _,cs_2 = plot_policy(Model)

    Phi_h_Model = NN_net_phi_h_eval(args, 2, [32,32,32], 3)
    load_path = path + "\\save\\model\\Phi_h_V_Model_32X32.pt"
    Model = load_model(load_path,Phi_h_Model)
    _,cs_3 = plot_policy(Model)

    Phi_h_Model = NN_net_phi_h_eval(args, 2, [64,64,64], 3)
    load_path = path + "\\save\\model\\Phi_h_V_Model_64X64.pt"
    Model = load_model(load_path,Phi_h_Model)
    ws,cs_4 = plot_policy(Model)




    plt.subplot(1,3,1)
    plt.plot(len_x, FB_8,label = "8X8",color = "red")
    plt.plot(len_x, FB_16,label = "16X16",color = "blue")
    plt.plot(len_x, FB_32,label = "32X32",color = "green")
    plt.plot(len_x, FB_64,label = "64X64",color = "yellow")
    plt.legend()
    plt.xlabel("train iter")
    plt.ylabel("FB loss")

    plt.subplot(1,3,2)
    plt.plot(len_x, utility_8,label = "8X8",color = "red")
    plt.plot(len_x, utility_16,label = "16X16",color = "blue")
    plt.plot(len_x, utility_32,label = "32X32",color = "green")
    plt.plot(len_x, utility_64,label = "64X64",color = "yellow")
    plt.legend()
    plt.xlabel("train iter")
    plt.ylabel("utility")

    plt.subplot(1,3,3)
    plt.plot(ws, cs_1,label = "8X8",color = "red")
    plt.plot(ws, cs_2,label = "16X16",color = "blue")
    plt.plot(ws, cs_3,label = "32X32",color = "green")
    plt.plot(ws, cs_4,label = "64X64",color = "yellow")
    plt.legend()
    plt.xlabel("w have")
    plt.ylabel("cosum choice")
    plt.show()