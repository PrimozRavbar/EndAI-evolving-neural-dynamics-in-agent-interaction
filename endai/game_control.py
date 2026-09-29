import numpy as np

import imageio.v3 as iio
from PIL import Image


class GameControl ():

    def __init__(self,open_field,leave_trace):

        self.rule = 'pred'
        self.max_pop = 5
        self.energy_change_predation = 5
        self.energy_change_movement = 1

        self.open_field = open_field
        #self.uni_mat = uni.uni_mat

        self.leave_trace = leave_trace


    def get_eu_dist(self,agent1, agent2):

        eu_dist = np.sqrt((agent1.pos_x - agent2.pos_x)**2 + (agent1.pos_y - agent2.pos_y)**2)
        return eu_dist

    def movement_consequences(self,agent,action):

        if action[0] == 0 and action[1] == 0:
            standing = True

        else: standing = False


        if agent.pred_potential > 0 and standing == False:
            agent.energy = agent.energy - 0.1
        if agent.pred_potential < 0 and standing == False:
            agent.energy = agent.energy + 0.1*(agent.pred_potential)



    def encounter(self,agent,list_of_agents):

        agent1 = agent

        total_encountered = 0


        for ind2,agent2 in enumerate(list_of_agents):

            if agent1 != agent2:

                eu_dist = self.get_eu_dist(agent1, agent2)
                #print(ind1,ind2,eu_dist)

                #eu_dist_th = (agent1.body_size + agent2.body_size)+2
                #eu_dist_th = 20

                eu_dist_th = np.sqrt((agent1.body_size*1)**2+(agent1.body_size*1)**2) + \
                np.sqrt((agent2.body_size*1)**2+(agent2.body_size*1)**2)

                eu_dist_th_legal = (agent1.body_size + agent2.body_size)

                if eu_dist < eu_dist_th_legal:
                    agent1.legal_move = False
                    #print("illegal")
                    #print(eu_dist)
                    break
                else: agent1.legal_move = True

                #energy_curr = agent1.energy

                #if eu_dist < eu_dist_th and agent1.pred_potential > 0 and agent2.pred_potential < 0:
                #if eu_dist < eu_dist_th and agent1.body_size > agent2.body_size:

                if eu_dist < eu_dist_th and agent1.energy > agent2.energy:
                #if eu_dist < eu_dist_th and agent1.energy < agent2.energy:

                    #agent1.energy = agent1.energy-np.absolute(agent2.energy/100)
                    #agent2.energy = agent2.energy-np.absolute(agent2.energy/100)

                    agent1.energy = agent1.energy+np.absolute(agent2.energy/10)
                    agent2.energy = agent2.energy-np.absolute(agent2.energy/10)

                    if agent1.energy<0: agent1.energy=0
                    if agent2.energy<0: agent2.energy=0



    def render_video(self, uni, list_of_ag):

        uni_mat = uni.uni_mat

        rgbArray = np.zeros(
            (uni_mat.shape[0], uni_mat.shape[1], 3),
            'uint8'
        )

        rgbArray[..., 0] = uni_mat[:, :, 2] * 255
        rgbArray[..., 1] = uni_mat[:, :, 1] * 255
        rgbArray[..., 2] = uni_mat[:, :, 0] * 255

        for ind_a, ag in enumerate(list_of_ag):

            nn_plot = ag.model.hidden_layer.detach().cpu().numpy()
            nn_plot = np.reshape(nn_plot, (8, 10))
            nn_plot = (nn_plot / np.max(np.max(nn_plot))) * 250

            if ag.pos_x < 90 and ag.pos_y < 90:

                rgbArray[
                    ag.pos_y:ag.pos_y+8,
                    ag.pos_x:ag.pos_x+10,
                    0
                ] = nn_plot

        # This is the frame
        #im_to_show = rgbArray

        im_to_show = np.array(Image.fromarray(rgbArray).resize((600, 600), Image.Resampling.NEAREST))

        return im_to_show

    """
    def render_video(self,uni,list_of_ag):

        uni_mat = uni.uni_mat

        rgbArray = np.zeros((uni_mat.shape[0],uni_mat.shape[1],3), 'uint8')
        rgbArray[..., 0] = uni_mat[:,:,2]*255
        rgbArray[..., 1] = uni_mat[:,:,1]*255
        rgbArray[..., 2] = uni_mat[:,:,0]*255

        for ind_a,ag in enumerate(list_of_ag):
            nn_plot = ag.model.hidden_layer.detach().cpu().numpy()
            nn_plot = np.reshape(nn_plot,(8,10))
            nn_plot = (nn_plot/np.max(np.max(nn_plot)))*250
            #nn_plot[nn_plot>250]=250

            if ag.pos_x<90 and ag.pos_y<90:

                rgbArray[ag.pos_y:ag.pos_y+8,
                         ag.pos_x:ag.pos_x+10, 0] = nn_plot



        im_to_show = rgbArray

        image_size_x = 600
        image_size_y = 600

        scaing_size = image_size_x/np.shape(uni_mat)[0]
        im_to_show = cv2.resize(im_to_show, (600, 600))

        #energy_text =  str(int(agent.energy))
        font = cv2.FONT_HERSHEY_SIMPLEX
        #org = (50, 20)
        #fontScale = 1
        #color = (255, 0, 0)
        #thickness = 2
        for ind_a,ag in enumerate(list_of_ag):
            #ag = list_of_ag[a]

            bc = ag.body_color

            ind_energy_text = str(int(ag.energy))
            org = (int(ag.pos_x*scaing_size), int(ag.pos_y*scaing_size))
            fontScale = 0.5
            color = (255-bc[0]*255, 255-bc[1]*255, 0-bc[2]*255)
            thickness = 2

            im_to_show = cv2.putText(im_to_show, ind_energy_text, org, font, fontScale,
                                     color, thickness, cv2.LINE_AA)

            ind_text = str(int(ind_a))
            org = (int(ag.pos_x*scaing_size), int(ag.pos_y*scaing_size)+12)
            fontScale = 0.5
            color = (255-bc[0]*255, 255-bc[1]*255, 0-bc[2]*255)
            thickness = 2

            im_to_show = cv2.putText(im_to_show, ind_text, org, font, fontScale,
                                     color, thickness, cv2.LINE_AA)

        #cv2.imshow('test_mat',im_to_show)
        clear_output(wait=True)
        cv2_imshow(im_to_show)

        plt.clf()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
            #break
        return im_to_show

   """