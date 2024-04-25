import numpy as np
import torch
import random


class Data_set():
    def __init__(self, args):
        self.args = args
        self.load_test_data()


    def draw_data(self,T_period):
        all_draw_data = []

        # draw w
        w_grids = self.draw_w_grid()
        self.w_grids = w_grids

        # draw shocks
        for _ in range(self.args.draw_shocks_each_epoch):
            draw_once_y,draw_once_shocks = self.draw_once_data(T_period)

            each_draw_data = [draw_once_y]+w_grids+draw_once_shocks
            all_draw_data.append(each_draw_data)

        #print(all_draw_data[0])
        data = torch.tensor(all_draw_data).type(torch.FloatTensor)

        return data # batch_each * (1 y + 64 grid + T shock)
    
    def draw_once_data(self, T_period):
        
        #draw y and sigma_squence
        random_norm = np.random.normal(0,1,(T_period+1,1))

        #fix y
        y_draw = random_norm[0]*self.args.sigma

        sigma_sq_draw = random_norm[1:]

        draw_data_y = y_draw[0]
        draw_data_shocks = sigma_sq_draw.T.tolist()[0]

        return draw_data_y, draw_data_shocks
    
    def draw_w_grid(self):
        #draw w[0.1,4]
        w_grids = []
        for _ in range(self.args.draw_grid_each_epoch):
            w_draw = random.random()*(self.args.W_range[1]-self.args.W_range[0])+self.args.W_range[0]
            w_grids.append(w_draw)

        return w_grids
    
    def load_test_data(self):
        test_sample = np.load(self.args.save_test_data_path)
        test_sample_all = []
        for i_test in range(test_sample.shape[0]):
            each_test = test_sample[i_test,:,:]

            each_test_tensor = torch.from_numpy(each_test).type(torch.FloatTensor)

            w_grids = each_test[0,1]
            w_grids = torch.tensor(w_grids).type(torch.FloatTensor)

            test_sample_all.append([each_test_tensor,w_grids])

        self.test_sample_all = test_sample_all
    
    @property
    def get_w_grid(self):
        out_w_grids = torch.tensor(self.w_grids).type(torch.FloatTensor)
        return out_w_grids
    
    @property
    def get_test_data(self):
        return self.test_sample_all
        