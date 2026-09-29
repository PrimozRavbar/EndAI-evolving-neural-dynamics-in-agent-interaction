import numpy as np
import imageio.v3 as iio

import torch

from .universe import Universe
from .agent import Agent
from .game_control import GameControl


#construct solution initial

def construct_init_solution(init_type="zeros"):

    sens_net_size = 2028
    hidden_net_size_1=80
    output_net_size=9

    #if zero_init==True:
    if init_type =="zeros":

        W_sens_hidden=np.zeros((sens_net_size, hidden_net_size_1))
        W_hidden = np.zeros((hidden_net_size_1, hidden_net_size_1))
        W_hidden_out = np.zeros((hidden_net_size_1, output_net_size))

        #leak_hidden_layer = np.zeros((sens_net_size, 10))

        W_epsilon_sens = np.zeros((sens_net_size, 10))
        W_epsilon_hidden = np.zeros((10, 1))

    if init_type =="random":

        W_sens_hidden= np.random.uniform(-1,1,(sens_net_size, hidden_net_size_1))
        W_hidden = np.random.uniform(-1,1,(hidden_net_size_1, hidden_net_size_1))
        W_hidden_out = np.random.uniform(-1,1,(hidden_net_size_1, output_net_size))

        W_epsilon_sens = np.random.uniform(-1,1,(sens_net_size, 10))
        W_epsilon_hidden = np.random.uniform(-1,1,(10, 1))

    if init_type =="mixed":

        W_sens_hidden= np.random.uniform(-1,1,(sens_net_size, hidden_net_size_1))
        W_hidden = np.zeros((hidden_net_size_1, hidden_net_size_1))
        W_hidden_out = np.zeros((hidden_net_size_1, output_net_size))

        W_epsilon_sens = np.random.uniform(-1,1,(sens_net_size, 10))
        W_epsilon_hidden = np.random.uniform(-1,1,(10, 1))

    solution_sens = W_sens_hidden.flatten()
    solution_hidden = W_hidden.flatten()
    solution_out = W_hidden_out.flatten()

    solution_epsilon_sens = W_epsilon_sens.flatten()
    solution_epsilon_hidden = W_epsilon_hidden.flatten()
    #solution_epsilon = W_epsilon.flatten()
    #solution_epsilon = 0
    #solution_agent_pred_pot = 1

    general_th = 10

    sol_th_sens = np.random.uniform(0,1,sens_net_size)* general_th
    #sol_th_hidden = np.random.uniform(0,1,hidden_net_size_1) * general_th

    sol_leak_hidden_layer = np.random.uniform(0,1,hidden_net_size_1)

    sol_th_out = np.random.uniform(0,1,output_net_size) * general_th

    solution_hyperparams = np.ones((10))

    solution = np.hstack((solution_sens,solution_hidden,solution_out,
                          solution_epsilon_sens,solution_epsilon_hidden,
                          sol_th_sens,
                          sol_leak_hidden_layer,
                          sol_th_out,
                          solution_hyperparams))

    print(np.shape(solution_sens),np.shape(solution_hidden),np.shape(solution_out),
          np.shape(solution_epsilon_sens),np.shape(solution_epsilon_hidden))
    print(np.shape(solution))

    return solution



def create_new_agent(
    solution,
    size_universe,
    list_elim_hidden,
    list_elim_hidden_out
):

    pos_x = int(np.random.rand()*size_universe)
    pos_y = int(np.random.rand()*size_universe)

    pred_pot = 1

    new_agent = Agent(pos_x, pos_y,pred_pot)

    input_net_size = new_agent.input_net_size
    hidden_net_size_1 = new_agent.hidden_net_size_1
    numb_outputs = new_agent.output_net_size
    epsilon_hidden_size = new_agent.epsilon_hidden_size

   # W_sens_hidden,W_hidden, W_hidden_out, W_epsilon_sens, W_epsilon_hidden,sol_th_sens, sol_th_hidden, sol_th_out, B_hyper_param  = new_agent.create_all_learnable_params(solution,
   #  list_elim_hidden,list_elim_hidden_out)

    W_sens_hidden,W_hidden, W_hidden_out, W_epsilon_sens, W_epsilon_hidden, leak_hidden_layer, B_hyper_param  = new_agent.create_all_learnable_params(solution,
     list_elim_hidden,list_elim_hidden_out)


    new_agent.model.W_sens_hidden = W_sens_hidden
    new_agent.model.W_hidden = W_hidden
    new_agent.model.W_hidden_out = W_hidden_out

    new_agent.model.W_epsilon_sens = W_epsilon_sens
    new_agent.model.W_epsilon_hidden = W_epsilon_hidden

    new_agent.model.leak_hidden_layer = leak_hidden_layer

    #new_agent.model.th_sens = sol_th_sens
    #new_agent.model.th_hidden = sol_th_hidden
    #new_agent.model.th_out = sol_th_out

    new_agent.model.B_hyper_param = B_hyper_param

    return new_agent


def run_episode(solution,popul,TOTAL_TIME, list_elim_hidden = [], list_elim_hidden_out = [],display_movie=False,record_data=False):

    start_new_universe = True

    frames = []

    if start_new_universe == True:

        open_field = False
        leave_trace = False

        size_universe = 100
        energy_in_universe=2000

        uni = Universe(size_universe)

        energy_field = uni.create_energy_field(size_universe,energy_in_universe)
        uni.uni_mat[:,:,1] = energy_field



    game_cont = GameControl(open_field,leave_trace)

    game_cont.leave_trace = False

    total_energy = 0

    st_rec_cube = np.zeros((100,100,5))

    numb_of_agents = np.shape(popul)[0]

    list_of_agents = []

    for a in range(0,numb_of_agents):
        solution_current = popul[a,:]
        new_agent = create_new_agent(
            solution_current,
            size_universe,
            list_elim_hidden,
            list_elim_hidden_out
        )
        new_agent.pred_potential = 1
        new_agent.body_color = [0.5,0,0]
        list_of_agents.append(new_agent)


    ########################################

    fitness_vect = np.zeros((1,numb_of_agents))

    #######################################


    ###########################################

    ########################################################


    if record_data == False:
        output_rec = []
        hidden_rec = []
        data_rec_dic = { "output_rec" : output_rec,
                         "hidden_rec" : hidden_rec}

    if record_data == True:

        ag = list_of_agents[0]

        output_rec = np.zeros((np.shape(ag.model.out_layer.detach().cpu().numpy())[0] , TOTAL_TIME ))
        hidden_rec = np.zeros((np.shape(ag.model.hidden_layer.detach().cpu().numpy())[0] , TOTAL_TIME ))

        pos_rec = np.zeros((2, TOTAL_TIME))

        data_rec_dic = { "output_rec" : output_rec,
                         "hidden_rec" : hidden_rec,
                         "pos_rec" :  pos_rec}


    ########################################################

    for t in range(0,TOTAL_TIME):
        for a in range(0,np.shape(list_of_agents)[0]):

            agent=list_of_agents[a]

            if game_cont.open_field == False:
                uni.draw_edges()

            #if game_cont.leave_trace == False:
            #    uni.clear_previous_object (agent)
            if t==0:
                uni.add_object_to_uni(agent)

            state, raw_im = agent.sense(uni.uni_mat)

            if game_cont.leave_trace == False:
                uni.clear_previous_object (agent)

            #ts=time.time()

            input_state = torch.from_numpy(raw_im).float()

            output = agent.model.forward(input_state)
            output = output.detach().cpu().numpy()
            #te=time.time()
            #if a==0:
               # print(ts-te)
            move_row, move_col = agent.take_discrete_action(output)

            #if a==5: print(move_row, move_col)

            #agent.move_discrete(move_row , move_col)
            agent.move_contin(move_row , move_col,uni, open_field)

            #if record_data==True:

                #st_rec_cube[agent.pos_y,agent.pos_x,0] = agent.model.epsilon
                #st_rec_cube[agent.pos_y,agent.pos_x,1] = agent.model.hidden_layer[19]
                #st_rec_cube[agent.pos_y,agent.pos_x,2] = agent.model.hidden_layer[0]
                #st_rec_cube[agent.pos_y,agent.pos_x,3] = agent.model.hidden_layer[8]

               # activity_rec_mat[:,t] = agent.model.hidden_layer


            if agent.pred_potential != 0:
                energy_added = energy_field[agent.pos_y,agent.pos_x]
                energy_field[agent.pos_y,agent.pos_x] = 0
                agent.energy = agent.energy + energy_added
                agent.body_size =  1 + int(np.sqrt(agent.energy)/5)

            if agent.energy<0:
                agent.energy=0

            uni.uni_mat[:,:,1] = energy_field

            game_cont.encounter(agent,list_of_agents)


            fitness_vect[0,a] = agent.energy


            uni.add_object_to_uni(agent)
            #ts = time.time()


        if record_data == True:
            output_rec[:,t] = output
            hidden_rec[:,t] = agent.model.hidden_layer.detach().cpu().numpy()

            pos_rec[0,t] = agent.pos_x
            pos_rec[1,t] = agent.pos_y


        if display_movie == True:

            frame = game_cont.render_video(uni, list_of_agents)
            frames.append(frame)

            """
            """

    param_cost = False
    if param_cost == True:
        param_cost_vect = np.expand_dims(np.count_nonzero(popul[:,:],axis=1),axis=0)
        fitness_vect = fitness_vect-param_cost_vect/300






    if record_data == False:
        data_rec_dic = None

    if record_data == True:
        data_rec_dic = { "output_rec" : output_rec,
                        "hidden_rec" : hidden_rec,
                        "pos_rec" :  pos_rec}

    if display_movie == True:

        iio.imwrite("simulation.mp4", frames, fps=20)

    return fitness_vect, raw_im, list_of_agents, data_rec_dic, frames