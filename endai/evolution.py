import numpy as np
import scipy

import pickle
import os
import re
import natsort

from .simulation import run_episode, construct_init_solution


class Evolution():

    def __init__(self,):

        self.evo_type = 0

    def fitness_func_GA(ga_instance, solution, solution_idx):

        pop_curr = ga_instance.population
        print(solution_idx)

        fitness = 0
        TIME_TOT = 1000

        for r in range(0,3):

            fitness1 = run_episode(solution,pop_curr,popul_prey,solution_idx,TIME_TOT,display_movie=False,record_data=False)
            fitness = fitness + fitness1

        #if fitness > 0:
        print(fitness)

        return fitness


    def fitness_func(self,solution, pop_curr):

        TIME_TOT = 1000
        fitness_vect = np.zeros((1,np.shape(pop_curr)[0]))

        for r in range(0,3):

            #ts=time.time()
            fitness_vect1, raw_im,list_of_agents, data_rec_dic = run_episode(solution,popul,TIME_TOT,display_movie=False,record_data=False)
            fitness_vect = fitness_vect + fitness_vect1
            #te=time.time()
            #print(ts-te)

        return fitness_vect


    def mutate(self,sol,mut_rate,mut_amp,mut_type="add_rand_val"):

        sol_next = np.copy(sol)
        #mask_mut = np.ones_like(sol)
        m_ind_vect = np.random.randint(len(sol), size=int(len(sol)*mut_rate))

        numb_of_hyperparams = 10

        if mut_type == "add_rand_val":

            for m in range(0,len(m_ind_vect)):
                if sol[m_ind_vect[m]] != 0:
                    sol_next[m_ind_vect[m]] = sol[m_ind_vect[m]]+np.random.rand()*rand_sign()*mut_amp

        if mut_type == "mask":

            for m in range(0,len(m_ind_vect)-numb_of_hyperparams):
                #mut_triplet = [0,np.random.rand()*rand_sign(),1]

                if sol[m_ind_vect[m]] == 0:
                    sol_next[m_ind_vect[m]]=np.random.rand()*rand_sign()*mut_amp
                if sol[m_ind_vect[m]] != 0:
                    sol_next[m_ind_vect[m]]=0


                #mask_mut[m_ind_vect[m]] = mut_triplet[np.random.randint(0,3)]

        return sol_next

    def fitness_func_dust(self,solution,pop_curr):

        TIME_TOT = 1000
        fitness_vect = np.zeros((1,np.shape(pop_curr)[0]))
        #fitness_vect1 = np.zeros((1,np.shape(pop_curr)[0]))

        for s in range(np.shape(pop_curr)[0]):

            solution = pop_curr[s,:]
            fitness_sol = 0

            for r in range(0,3):

                dust_removed_tot, rgbArray, ag, output, energy_field, data_rec_dic  = run_episode (solution,TIME_TOT,display_movie=False,record_data=False)
                fitness_sol =  fitness_sol + dust_removed_tot

                Wh = ag.model.W_hidden.detach().cpu().numpy()
                Wo = ag.model.W_hidden_out.detach().cpu().numpy()
                #param_cost = (np.count_nonzero(Wh) + np.count_nonzero(Wo))/(np.size(Wh) + np.size(Wo))
                param_cost =  (np.count_nonzero(Wh) + np.count_nonzero(Wo))/100

            #fitness_vect[0,s] = fitness_sol/(param_cost*3)
            fitness_vect[0,s] = fitness_sol #- param_cost #- energy_loss

        return fitness_vect

    def mutate(self,sol,mut_rate,mut_amp,mut_type="add_rand_val"):

        numb_of_hyperparams = 10

        sol_next = np.copy(sol)
        #mask_mut = np.ones_like(sol)
       # m_ind_vect = np.random.randint(len(sol)-numb_of_hyperparams, size=int(len(sol)*mut_rate))



        if mut_type == "add_rand_val":

            m_ind_vect = np.random.randint(len(sol), size=int(len(sol)*mut_rate))

            for m in range(0,len(m_ind_vect)):
                if sol[m_ind_vect[m]] != 0:
                    sol_next[m_ind_vect[m]] = sol[m_ind_vect[m]]+np.random.rand()*rand_sign()*mut_amp

        if mut_type == "mask":

            m_ind_vect = np.random.randint(len(sol)-numb_of_hyperparams, size=int(len(sol)*mut_rate))

            #mask_lim1 = 245760 #size of sens_hidden W
            #mask_lim2 = 245760 + 6400 + 1200 #size of sens_hidden W

            for m in range(0,len(m_ind_vect)-numb_of_hyperparams):
            #for m in range(mask_lim1,mask_lim2):
                #mut_triplet = [0,np.random.rand()*rand_sign(),1]

                if sol[m_ind_vect[m]] == 0:
                    sol_next[m_ind_vect[m]]=np.random.rand()*rand_sign()*mut_amp
                if sol[m_ind_vect[m]] != 0:
                    sol_next[m_ind_vect[m]]=0

        if mut_type == "mixed":

            m_ind_vect = np.random.randint(len(sol), size=int(len(sol)*mut_rate))

            for m in range(0,len(m_ind_vect)):
                if sol[m_ind_vect[m]] != 99999999:
                    sol_next[m_ind_vect[m]] = sol[m_ind_vect[m]]+np.random.rand()*rand_sign()*mut_amp

                #mask_mut[m_ind_vect[m]] = mut_triplet[np.random.randint(0,3)]

        return sol_next


    def find_oldest_file(self,file_list):
        ages_vec = np.zeros((len(file_list),1))

        for f in range(len(file_list)):
            fl_str=file_list[f]
            ages_vec[f]=int(re.findall(r'\d+', fl_str)[1])

        oldest_file =  np.argmin(ages_vec)
        return oldest_file

    def remove_oldest_indiv(self,pool_folder_name, min_pool_size):

        file_list = os.listdir(pool_folder_name)
        if len(file_list) > min_pool_size:
            ind_oldest = self.find_oldest_file(file_list)
            name_of_oldest = pool_folder_name+file_list[ind_oldest]
            os.remove(name_of_oldest)

    def add_to_pool_1 (self,popul,fitness_vect,pool_folder_name, max_pool_size,gen_ind=0):

        energy_in_universe = 1000

        pool_saving_size = 10
        #max_pool_size = 2000

        fit_vect_rel = (fitness_vect/(energy_in_universe*1))*pool_saving_size
        to_next_gen_in_pool = np.floor(fit_vect_rel)

        #print(fit_vect_rel)

        if np.max(to_next_gen_in_pool) < 1:
            #print(to_next_gen_in_pool)
            to_next_gen_in_pool[0,np.argmax(fitness_vect)]=1
            #print(to_next_gen_in_pool)

        n=0

        for indiv in range(0,np.shape(to_next_gen_in_pool)[1]):

            sol = np.expand_dims(popul[indiv,:],axis=0)
            n = to_next_gen_in_pool[0,indiv]

            for ind_save in range(0,int(n)):
                sol_sparse = scipy.sparse.csr_matrix(sol)
                sol_name = f"{pool_folder_name}_{int(fitness_vect[0,indiv])}_solution_{gen_ind}_{int(ind_save)}"
                with open(sol_name, "wb") as f:
                    pickle.dump(sol_sparse, f)

            file_list = os.listdir(pool_folder_name)
            if len(file_list) > max_pool_size:
                numb_delete_files = int(len(file_list)-max_pool_size)
                sl=natsort.natsorted(file_list)
                for d in range(0, numb_delete_files):
                    name_del_file = pool_folder_name+sl[d]
                    #print(name_del_file)
                    os.remove(name_del_file)


    def select_popul_from_pool_1(self,numb_samples,pool_folder_name):

        file_list = os.listdir(pool_folder_name)
        if len(file_list)==0:
            #new_sol=construct_init_solution(zero_init=True)
            #new_sol=construct_init_solution(init_type="mixed")
            new_sol=construct_init_solution(init_type="zeros")

            select_popul = np.zeros((numb_samples,np.shape(new_sol)[0]))
            #select_popul = np.random.rand(numb_samples,np.shape(new_sol)[0])
            for r in range(0,numb_samples+0):
                #select_popul[r,:] = construct_init_solution(init_type="mixed")
                select_popul[r,:] = construct_init_solution(init_type="zeros")

        if len(file_list)>0:

            file_ind_vect = np.random.randint(len(file_list), size=int(numb_samples))

            selected_file_name = file_list[file_ind_vect[0]]
            with open(pool_folder_name + selected_file_name, "rb") as f:
                sol_sparse = pickle.load(f)

            if scipy.sparse.issparse(sol_sparse) == True:

                select_popul = sol_sparse.toarray()

            else:

                 select_popul = np.expand_dims(sol_sparse,axis=0)

            for s in range(1,numb_samples):

                selected_file_name = file_list[file_ind_vect[s]]

                with open(pool_folder_name + selected_file_name, "rb") as f:
                    sol_sparse = pickle.load(f)

                if scipy.sparse.issparse(sol_sparse) == True:
                    sol = sol_sparse.toarray()
                else:
                    sol = np.expand_dims(sol_sparse,axis=0)

                select_popul = np.vstack((select_popul,sol))

        return select_popul
