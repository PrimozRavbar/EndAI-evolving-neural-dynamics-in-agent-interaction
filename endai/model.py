import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fun


def rand_sign():
    return 1 if np.random.rand() < 0.5 else -1


class ModelNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, B_hyper_param):
        super(ModelNet, self).__init__()

        # Initialize hyperparameters
        self.B_hyper_param = B_hyper_param

        self.hidden_max = B_hyper_param[0] * 10000
        self.add_hidden = B_hyper_param[1] * 0.5
        self.add_sens = B_hyper_param[2] * 0.01
        self.leak_sens = B_hyper_param[3] * 0.1
        self.leak_hidden = B_hyper_param[4] * 0.1
        self.max_epsilon = B_hyper_param[5] * 1


        self.W_epsilon_sens = nn.Parameter(torch.randn(input_size, hidden_size))  # (input_size, hidden_size)
        self.W_epsilon_hidden = nn.Parameter(torch.randn(hidden_size, output_size))  # (hidden_size, output_size)

        self.W_sens_hidden = nn.Parameter(torch.randn(input_size, hidden_size))  # (input_size, hidden_size)
        self.W_hidden = nn.Parameter(torch.randn(hidden_size, hidden_size))  # (hidden_size, hidden_size)
        self.W_hidden_out = nn.Parameter(torch.randn(hidden_size, output_size))  # (hidden_size, output_size)

        self.hidden_layer = torch.zeros(hidden_size)
        self.out_layer = torch.zeros(output_size)

        self.leak_hidden_layer = torch.zeros(hidden_size)

        #self.th_sens = np.random.uniform(0,1, input_size)
        #self.th_hidden = np.random.uniform(0,1, hidden_size)
        #self.th_out = np.random.uniform(0,1, output_size)

        self.hidden_net_size_1 = hidden_size
        self.numb_outputs = output_size



    def forward(self, input_state):

        self.hidden_max = self.B_hyper_param[0] * 10000
        self.add_hidden = self.B_hyper_param[1] * 0.5
        self.add_sens = self.B_hyper_param[2] * 0.01
        self.leak_sens = self.B_hyper_param[3] * 0.1
        self.leak_hidden = self.B_hyper_param[4] * 0.5
        self.max_epsilon = self.B_hyper_param[5] * 1

        # Flatten the input state
        trans_state = input_state.view(-1)  # Shape: (input_size,)
        trans_state = trans_state*1
        ######################################################
        # Compute epsilon values

        #epsilon_hidden_layer = torch.matmul(trans_state, self.W_epsilon_sens)  # Shape: (hidden_size,)
        #epsilon = torch.matmul(epsilon_hidden_layer, self.W_epsilon_hidden)  # Shape: (output_size,)
        #epsilon = torch.clamp(epsilon, -self.max_epsilon, self.max_epsilon)

        #epsilon = epsilon*0

        epsilon = self.hidden_layer[79]*100
        epsilon = torch.clamp(epsilon, 0, self.max_epsilon)*1
       # print(epsilon)


        ##################################################################



        trans_state = torch.matmul(self.W_sens_hidden.T,trans_state.view(-1, 1))
        trans_state = trans_state.squeeze(1)
        #print( trans_state.shape )

        trans_state = torch.clamp(trans_state, 0, self.hidden_max)

        # Update hidden layer with recurrent effects
        recurrent_effect = torch.matmul(self.W_hidden,self.hidden_layer)

        #print(np.sum(self.hidden_layer))
        #print(self.leak_hidden, type(self.leak_hidden))
        #print("hidden_layer size:", self.hidden_layer.shape)
        #print("leak_hidden:", self.leak_hidden)
        #print("trans_state size:", trans_state.shape)
        #print("recurrent_effect size:", recurrent_effect.shape)
        #print("th_hidden size:", self.th_hidden.shape)

        #print(torch.min(trans_state))

        self.hidden_layer = (
            self.hidden_layer - self.leak_hidden_layer*self.hidden_layer
        )

        self.hidden_layer = torch.clamp(self.hidden_layer, 0, self.hidden_max)

        self.hidden_layer = (
            self.hidden_layer - self.leak_hidden*self.hidden_layer
            + trans_state * self.add_sens
            + recurrent_effect * self.add_hidden
            #- self.th_hidden
        )

       ## print(self.leak_hidden)
        self.hidden_layer = torch.clamp(self.hidden_layer, 0, self.hidden_max)  # Shape: (hidden_size,)

        # Compute output layer
        #self.out_layer = torch.matmul(self.hidden_layer, self.W_hidden_out)  # Shape: (output_size,)
        self.out_layer = torch.matmul(self.W_hidden_out.T,self.hidden_layer)
        #self.out_layer = torch.clamp(self.out_layer, 0, self.hidden_max)
        # Add noise
        N_out = torch.randn(self.numb_outputs) * self.out_layer * epsilon
        #print(N_out)
        #print(self.out_layer)
        #print(epsilon)
        final_state = self.out_layer + N_out  # Shape: (output_size,)
        final_state = torch.clamp(final_state, min=0)  # Shape: (output_size,)

        return final_state