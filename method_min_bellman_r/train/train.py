import torch
import numpy as np
from model.model import NN_net_phi_h
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, args, Data_set):
        self.args = args
        self.Data_set = Data_set

    def train(self):
        Phi_h_Model = NN_net_phi_h(self.args)
        optimizer = torch.optim.Adam(Phi_h_Model.parameters(),lr=self.args.learning_rate)

        #保留模型训练数据
        FB_save_list = []
        utility_save_list = []

        #抽测试集数据，只抽一个给定的
        sample_test_combine = []
        for _ in range(self.args.draw_sample_test):
            data_test = self.Data_set.draw_data(self.args.T_period_test)
            w_grids_t = self.Data_set.get_w_grid
            sample_test_combine.append([data_test, w_grids_t])


        #开始进入循环
        for i_epoch in range(self.args.train_epoch):
            #开始随机抽取数据
            data_train = self.Data_set.draw_data(self.args.T_period) #batch * (1y + 1w + 1shock)
            
            #输出w维度和 初始收入冲击
            w_grids = self.Data_set.get_w_grid.unsqueeze(0) #1*1

            w_grids = w_grids.repeat(self.args.draw_shocks_each_epoch,1) #128*1

            y_shock = data_train[:,0].unsqueeze(1) #128*1

            #初始输入状态数据
            model_state = torch.cat([y_shock, w_grids], dim = 1) #128*2

            c_w_h_V = Phi_h_Model(model_state)

            c_w_ratio = 1/(1+torch.exp(-c_w_h_V[:,0].unsqueeze(1)))
            h = torch.exp(c_w_h_V[:,1].unsqueeze(1))
            V = c_w_h_V[:,2].unsqueeze(1)

            FB_now = self.FB_function(1 - c_w_ratio, 1- h)
            c_now = c_w_ratio * w_grids


            #shock_1
            shock_eps_1 = data_train[:,1+self.args.draw_grid_each_epoch].unsqueeze(1)
            V_part_next_1, Euler_part_next_1 = self.get_shock_change(Phi_h_Model, shock_eps_1, c_now, w_grids, h, V)

            #shock_2
            shock_eps_2 = data_train[:,1+self.args.draw_grid_each_epoch+1].unsqueeze(1)

            V_part_next_2, Euler_part_next_2 = self.get_shock_change(Phi_h_Model, shock_eps_2, c_now, w_grids, h, V)

            #计算欧拉剩余
            v_ = 0
            v_h = 0

            loss = torch.mean(V_part_next_1 * V_part_next_2 + v_ * torch.square(FB_now) + v_h * (Euler_part_next_1 * Euler_part_next_2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #验证模型
            if i_epoch%100 == 0:
                utility_test_mean, FB_mean = self.eval_model(Phi_h_Model, sample_test_combine)

                utility_test_mean = utility_test_mean
                FB_mean = FB_mean

                utility_save_list.append(utility_test_mean)
                FB_save_list.append(FB_mean)

                print(utility_test_mean, FB_mean)

        self.save_model(Phi_h_Model)
        self.plot_reward_loss(FB_save_list, utility_save_list)
        self.plot_policy(Phi_h_Model)

        data_save = [FB_save_list, utility_save_list]
        data_save = np.array(data_save)
        np.save(self.args.save_data_path, data_save)
        
    def save_model(self, Model):
        torch.save(Model.state_dict(), self.args.save_model_path)

    def load_model(self, Model):
        Model.load_state_dict(torch.load(self.args.save_model_path))
        return Model
    
    def plot_reward_loss(self, FB, uility):
        plt.figure()
        plt.plot(list(range(len(FB))),FB)
        plt.xlabel("Train iter")
        plt.ylabel("FB Loss")

        plt.figure()
        plt.plot(list(range(len(FB))),uility)
        plt.xlabel("Train iter")
        plt.ylabel("Utility")
        plt.show()


    def plot_policy(self, Model):
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
            h = torch.exp(c_w_h[1])
            
            c_ = c_w_ratio * w_grid

            if w_grid >2:
                print(c_w_ratio,h)
                
            c = c_.detach().numpy().tolist()

            ws.append(w_grid)
            cs.append(c)

        plt.figure()
        plt.plot(ws,cs)
        plt.xlabel("w grids")
        plt.ylabel("c grids")
        plt.show()

    def get_shock_change(self, Model, shock, c_now, w_last, h, V):
        #shock_next state
        y_shock_next = self.args.sigma * shock

        w_next = self.args.rate * (w_last - c_now) + torch.exp(y_shock_next)

        model_state_next = torch.cat([y_shock_next, w_next],dim = 1)

        c_w_h_V_next = Model(model_state_next)

        c_w_ratio_next = 1/(1 + torch.exp(-c_w_h_V_next[:,0].unsqueeze(1)))
        h_next = torch.exp(c_w_h_V_next[:,1].unsqueeze(1))
        V_next = c_w_h_V_next[:,2].unsqueeze(1)

        util_c = (c_now**(1-self.args.gamma)-1)/(1-self.args.gamma)

        c_next = c_w_ratio_next * w_next

        #V_part_one
        V_part_next = V - util_c - self.args.beta * V_next

        #Euler part        

        #partial V_w
        partial_V_w = (V_next - V)/(w_next - w_last)

        Euler_part_next = (self.args.beta * partial_V_w)/(c_now**(-self.args.gamma)) - h

        return V_part_next, Euler_part_next


    def eval_model(self, Model, sample_test_combine):
        #开始验证模型
        for i_test in range(self.args.draw_sample_test):
            #导入测试数据
            data_test = sample_test_combine[i_test][0]
            w_grids_t = sample_test_combine[i_test][1]

            #输出w维度和 初始收入冲击
            w_grids_test = w_grids_t.unsqueeze(0).repeat(self.args.draw_shocks_each_epoch,1) #128*1
            y_shock = data_test[:,0].unsqueeze(1) #128*1

            #更新期数同时要计算欧拉剩余
            for t_period in range(self.args.T_period):
                #叠加正则输入
                model_state = torch.cat([y_shock, w_grids_test], dim = 1) #128*2
                
                c_w_h_V= Model(model_state)  #128*2 to 128*1
                c_w_ratio = 1/(1+torch.exp(-c_w_h_V[:,0].unsqueeze(1)))
                h = torch.exp(c_w_h_V[:,1].unsqueeze(1))
                V = c_w_h_V[:,2].unsqueeze(1)

                c_now = c_w_ratio * w_grids_test #128*1


                #跨期
                if t_period != self.args.T_period:
                    #下期冲击
                    y_shock = self.args.sigma * data_test[:,1+self.args.draw_grid_each_epoch+t_period].unsqueeze(1) #128*1

                    #进行跨期计算
                    w_grids_test = self.args.rate*(w_grids_test - c_now) + torch.exp(y_shock) # 128*1

                #计算当期效用
                utility_now = (c_now**(1-self.args.gamma)-1)/(1-self.args.gamma)

                #效用跨期迭代
                if t_period == 0:
                    utility_value_test_iter = utility_now
                else:
                    utility_value_test_iter += self.args.beta**t_period * utility_now

                #计算欧拉剩余，即两个跨期之间进行
                if t_period != 0:
                    h_last = self.args.beta * self.args.rate*(c_now/last_c)**(-self.args.gamma)
                    if t_period == 1:
                        FB_test_iter = self.FB_function(1-last_c_w_ratio,1-h_last)/(self.args.T_period)#128*1
                    else:
                        FB_test_iter += self.FB_function(1-last_c_w_ratio,1-h_last)/(self.args.T_period)

                last_c_w_ratio = c_w_ratio
                last_c = c_now

            #计算均值
            utility_test_iter_mean =  torch.mean(utility_value_test_iter) #128*1 to 1*1
            FB_iter_mean = torch.mean(FB_test_iter)

            if i_test == 0:
                utility_test_mean = utility_test_iter_mean.detach().numpy().tolist()/self.args.draw_sample_test
                FB_mean = FB_iter_mean.detach().numpy().tolist()/self.args.draw_sample_test
            else:
                utility_test_mean += utility_test_iter_mean.detach().numpy().tolist()/self.args.draw_sample_test
                FB_mean += FB_iter_mean.detach().numpy().tolist()/self.args.draw_sample_test      

        return utility_test_mean, FB_mean

    def FB_function(self,a,h):
        return a+h - torch.sqrt(a**2+h**2)
                

            




                
        