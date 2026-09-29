import numpy as np

import torch
import torch.nn as nn

from .model import ModelNet


class Agent ():

    def __init__(self,pos_x, pos_y,pred_potential):

        self.type = "agent"

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.vel_x = 0.0
        self.vel_y = 0.0

        self.legal_move = True

        self.energy = 0
        self.reward_for_repr = 0

        self.body_size = 1 #+ int(self.energy/10)

        #self.pred_potential = rand_sign()*1
        self.pred_potential = pred_potential


        if self.pred_potential < 0:
            #self.body_color = [0,(-1)*self.pred_potential/2,0]
            self.body_color = [0,0,1]

        if self.pred_potential > 0:
            self.body_color = [0.5,0,0]
            #self.body_color = [self.pred_potential/10,0,1-self.pred_potential/10]


        self.sense_sq_area = 26

        self.input_net_size = self.sense_sq_area*self.sense_sq_area*3
        self.hidden_net_size_1 = 80
        self.output_net_size = 9

        self.epsilon_hidden_size = 10

        self.numb_of_hyperparams = 10

        self.model =  ModelNet(self.input_net_size,
                           self.hidden_net_size_1,
                           self.output_net_size ,
                           B_hyper_param= np.ones((self.numb_of_hyperparams)))


    def create_all_learnable_params(self,solution,list_elim_hidden,list_elim_hidden_out):

        sens_net_size =  self.input_net_size
        hidden_net_size_1= self.hidden_net_size_1
        output_net_size = self.output_net_size

        numb_of_hyperparams = self.numb_of_hyperparams

        #list_elim_hidden = [20,58,63,70]
        #list_elim_hidden = []


        idx1 = sens_net_size*hidden_net_size_1
        idx2 = idx1 + hidden_net_size_1*hidden_net_size_1
        idx3 = idx2 + hidden_net_size_1*output_net_size
        idx4 = idx3 + sens_net_size * 10
        idx5 = idx4 + 10  # sol_epsilon_hidden
        idx6 = idx5 + sens_net_size #sol_th_sens
        idx7 = idx6 + hidden_net_size_1 #sol_leak_hidden_layer
        idx8 = idx7 + output_net_size

        # Extract and reshape weight matrices
        W_sens_hidden = np.reshape(solution[:idx1], (sens_net_size, hidden_net_size_1) )
        W_hidden = np.reshape(solution[idx1:idx2],  (hidden_net_size_1, hidden_net_size_1) )
        W_hidden_out = np.reshape(solution[idx2:idx3], (hidden_net_size_1,output_net_size) )



        W_hidden[np.absolute(W_hidden) < 0.0] = 0

        W_hidden[list_elim_hidden,:] = 0
        W_hidden_out[list_elim_hidden_out,:] = 0


        W_epsilon_sens = np.reshape(solution[idx3:idx4], (sens_net_size, 10))
        W_epsilon_hidden = np.reshape(solution[idx4:idx5], (10, 1))

        leak_hidden_layer = np.absolute(solution[idx6:idx7])

        #sol_th_sens = solution[idx5:idx6]
        #sol_th_hidden = solution[idx6:idx7]
        #sol_th_out = solution[idx7:idx8]

        #sol_th_sens[sol_th_sens<0] = 0
        #sol_th_hidden[sol_th_hidden<0] = 0
        #sol_th_out[sol_th_out<0] = 0


        B_hyper_param = solution[-numb_of_hyperparams::]


        W_sens_hidden = nn.Parameter(torch.from_numpy(W_sens_hidden).float())
        W_hidden = nn.Parameter(torch.from_numpy(W_hidden).float())
        W_hidden_out = nn.Parameter(torch.from_numpy(W_hidden_out).float())
        W_epsilon_sens = nn.Parameter(torch.from_numpy(W_epsilon_sens).float())
        W_epsilon_hidden = nn.Parameter(torch.from_numpy(W_epsilon_hidden).float())

        leak_hidden_layer = nn.Parameter(torch.from_numpy(leak_hidden_layer).float())


        #sol_th_sens= nn.Parameter(torch.from_numpy(sol_th_sens).float())
        #sol_th_hidden = nn.Parameter(torch.from_numpy(sol_th_hidden).float())
        #sol_th_out = nn.Parameter(torch.from_numpy(sol_th_out).float())

        #return  W_sens_hidden,W_hidden, W_hidden_out, W_epsilon_sens, W_epsilon_hidden,  sol_th_sens, sol_th_hidden, sol_th_out, B_hyper_param
        return  W_sens_hidden,W_hidden, W_hidden_out, W_epsilon_sens, W_epsilon_hidden,  leak_hidden_layer, B_hyper_param



    def take_discrete_action(self,output):

        if np.max(output)>0:

            output_rs = np.reshape(output,(3,3))
            ind = np.where(output_rs==np.max(np.max(output_rs)))
            rand_ind = np.random.randint(0,np.shape(ind)[1])

            move_row=ind[0][rand_ind]-1
            move_col=ind[1][rand_ind]-1

        else:
            move_row=0
            move_col=0

        return move_row,move_col



    def move_discrete(self, move_row, move_col,uni):

        open_field = game_cont.open_field

        pos_new_x = self.pos_x + move_col
        pos_new_y = self.pos_y + move_row

        if open_field == True:

            pos_min = 7
            pos_max = uni.size_universe-7

            if pos_new_x > pos_max:
                pos_new_x = pos_min + 1
            if pos_new_y > pos_max:
                pos_new_y = pos_min + 1
            if pos_new_x < pos_min:
                pos_new_x = pos_max - 1
            if pos_new_y < pos_min:
                pos_new_y = pos_max - 1

        if open_field == False:

            edge_thick = 3

            pos_min = edge_thick
            pos_max = uni.size_universe-edge_thick

            if pos_new_x > pos_max:
                pos_new_x = pos_max
            if pos_new_y > pos_max:
                pos_new_y = pos_max
            if pos_new_x < pos_min:
                pos_new_x = pos_min
            if pos_new_y < pos_min:
                pos_new_y = pos_min

        self.pos_x = pos_new_x
        self.pos_y = pos_new_y


    def move_contin(self,move_row, move_col, uni, open_field=False):

        #open_field = game_cont.open_field
        size_universe =  uni.size_universe

        pers = 1
        drag = 0.1

        vel_max = 1
        vel_min = -1

        self.vel_x = self.vel_x + move_col/pers
        self.vel_y = self.vel_y + move_row/pers

        #self.vel_x = self.vel_x - (np.sign(self.vel_x)*drag)
        #self.vel_y = self.vel_y - (np.sign(self.vel_y)*drag)

        if self.vel_x > vel_max:
            self.vel_x = vel_max
        if self.vel_y > vel_max:
            self.vel_y = vel_max

        if self.vel_x < vel_min:
            self.vel_x = vel_min
        if self.vel_y < vel_min:
            self.vel_y = vel_min

        pos_new_x = self.pos_x + self.vel_x
        pos_new_y = self.pos_y + self.vel_y

        if open_field == True:

            pos_min = 7
            pos_max = size_universe-7

            if pos_new_x > pos_max:
                pos_new_x = pos_min + 1
            if pos_new_y > pos_max:
                pos_new_y = pos_min + 1
            if pos_new_x < pos_min:
                pos_new_x = pos_max - 1
            if pos_new_y < pos_min:
                pos_new_y = pos_max - 1

        if open_field == False:

            edge_thick = 3

            pos_min = edge_thick+self.body_size
            pos_max = size_universe-edge_thick-self.body_size

            if pos_new_x > pos_max:
                pos_new_x = pos_max
            if pos_new_y > pos_max:
                pos_new_y = pos_max
            if pos_new_x < pos_min:
                pos_new_x = pos_min
            if pos_new_y < pos_min:
                pos_new_y = pos_min

        self.pos_x = int(pos_new_x)
        self.pos_y = int(pos_new_y)


    def accelerate(self, acc_x, acc_y):
        self.vel_x = self.vel_x + acc_x
        self.vel_y = self.vel_y + acc_y

    def add_self_to_sense_image(self,raw_im):

        ob_size_x = self.body_size
        ob_size_y = self.body_size
        ob_color = self.body_color #add to blue channel

        raw_im[13-ob_size_y:13+ob_size_y,
             #int(self.pos_x)-ob_size_x:int(self.pos_x)+ob_size_x,2] = 1 #[.7,.45,.100]
             13-ob_size_x:13+ob_size_x,2] = ob_color[0] #[.7,.45,.100]

        return raw_im


    def sense(self,uni_mat):

        open_field = False
        #open_field = True

        #open_field = game_cont.open_field

        #sense_cube = np.zeros((self.sense_sq_area, self.sense_sq_area, 3))

        if open_field == True:

            uni_mat_tiled = np.tile(uni_mat,(3,3,1))

            pos_r = np.shape(uni_mat)[0] + int(self.pos_y)
            pos_c = np.shape(uni_mat)[1] + int(self.pos_x)

            sense_cube = uni_mat_tiled[pos_r-int(self.sense_sq_area/2):pos_r+int(self.sense_sq_area/2) ,
                         pos_c-int(self.sense_sq_area/2):pos_c+int(self.sense_sq_area/2) ,:]

        if open_field == False:

            uni_mat_tiled = np.tile(uni_mat,(3,3,1))
            uni_mat_tiled[0:np.shape(uni_mat)[0],:,:] = [0,0,0]
            uni_mat_tiled[np.shape(uni_mat)[0]*2:np.shape(uni_mat)[0]*3,:,:] = [0,0,0]

            uni_mat_tiled[:,0:np.shape(uni_mat)[0],:] = [0,0,0]
            uni_mat_tiled[:,np.shape(uni_mat)[0]*2:np.shape(uni_mat)[0]*3,:] = [0,0,0]

            pos_r = np.shape(uni_mat)[0] + int(self.pos_y)
            pos_c = np.shape(uni_mat)[1] + int(self.pos_x)

            sense_cube = uni_mat_tiled[pos_r-int(self.sense_sq_area/2):pos_r+int(self.sense_sq_area/2) ,
                         pos_c-int(self.sense_sq_area/2):pos_c+int(self.sense_sq_area/2) ,:]


        im = sense_cube
        im = np.transpose(im,(2,0,1))
        im = np.expand_dims(im,axis=0)
        im = torch.tensor(im,dtype=torch.float32)

        raw_im=self.add_self_to_sense_image(sense_cube)

        return im, raw_im


    def update_energy (self,count):

        max_energy = 100

        if count <= 1:
            self.energy = self.energy + 15

        if count > 1:
            self.energy = self.energy - (count -1)*5

        if self.energy < 0:
            self.energy = 0

        if self.energy > max_energy:
            self.energy = max_energy