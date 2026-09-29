from .evolution import Evolution
from .simulation import construct_init_solution

## main training loop

generations=100000000
start_gen = 8511

version_name = "16p9_att_leakNew_fullRNN"

#pool_folder_name = "C:\\Users\\ravbar\\Desktop\\ALP_python\\pool_folder_12\\"
#pool_folder_name = "C:\\Users\\primoz\\Desktop\\USB\\python_code\\ALP\\pool_folder_12\\"
pool_folder_name = "C:\\Users\\primo\\Desktop\\USB\\ALP_python\\pool_att_16p9_leakNew_fullRNN\\"
#pool_folder_name = "C:\\Users\\ravbar\\Desktop\\ALP_python\\pool_att_16p8_leakNew_fullRNN\\"

save_records = False

eliticism = 99 # pool operations
mut_rate = 0.1
max_mut_rate = 0.1
mut_rate_of_rate = 50
mut_amp = 0.1

mut_rate_mask = 0.001
max_mut_rate_mask = 0.001
mut_rate_of_rate_mask = 1
mut_amp_mask = 1

population_episode_size = 3
min_pool_size = 250
max_pool_size = 500

max_fitness_prev = 0
sum_fitness_prev = 0

evol =  Evolution()

#solution=construct_init_solution(zero_init=False)
#solution=construct_init_solution(init_type="mixed")
solution=construct_init_solution(init_type="zeros")

fitness_vect=np.zeros((1,population_episode_size))+0.01
fitness_vect[fitness_vect<0.01]=0.01

list_elim_hidden = []
list_elim_hidden_out = []

mut_rate_vec = np.zeros((1,population_episode_size))+0.3

prev_max_fit =0

for g in range(start_gen,generations):

    popul= evol.select_popul_from_pool_1(population_episode_size,pool_folder_name)
    sol = popul[0,:]
    fitness_vect = evol.fitness_func(sol, popul)
    fitness_vect[fitness_vect<0.01]=0.01

    if np.min(fitness_vect) != 0:
        mut_rate_vec = mut_rate_of_rate/fitness_vect
        mut_rate_vec[mut_rate_vec>max_mut_rate]=max_mut_rate
    else:
        mut_rate_vec = np.zeros((1,population_episode_size))+max_mut_rate

    if np.min(fitness_vect) != 0:
        mut_rate_vec_mask = mut_rate_of_rate_mask/fitness_vect
        mut_rate_vec_mask[mut_rate_vec_mask>max_mut_rate_mask]=max_mut_rate_mask
    else:
        mut_rate_vec_mask = np.zeros((1,population_episode_size))+max_mut_rate_mask

    #popul=select_popul_from_pool_1(population_episode_size,pool_folder_name)

    for s in range(0,np.shape(popul)[0]):
        sol = popul[s,:]
        mut_rate = mut_rate_vec[0,s]
        mut_rate_mask = mut_rate_vec_mask[0,s]

        sol_next = evol.mutate(sol,mut_rate,mut_amp,mut_type="add_rand_val")
        sol_next = evol.mutate(sol_next,mut_rate_mask,mut_amp_mask,mut_type="mask")

        #####sol_next[-10:] =  np.ones((10))

        popul[s,:]=sol_next

    evol.remove_oldest_indiv(pool_folder_name, min_pool_size)
    evol.add_to_pool_1 (popul,fitness_vect,pool_folder_name, max_pool_size,gen_ind=g)

    max_fitness = np.max(fitness_vect)
    sum_fitness = np.sum(fitness_vect)

    #if g % 10 == 0:
     #   gen_rec[int(g/10),:] = popul[np.argmax(fitness_vect),:]
     #   fitness_rec[int(g/10),0] = fitness_vect[0,np.argmax(fitness_vect)]



    if save_records == True:

        if max_fitness>max_fitness_prev:

            best_solution=sol_next
            best_solution_spr = scipy.sparse.csr_matrix(best_solution)
            sol_name = 'solution_' + str(int(max_fitness)) + version_name
            with open(sol_name, "wb") as f:
                pickle.dump(best_solution_spr, f)
            popul_spr = scipy.sparse.csr_matrix(popul)
            pop_name = 'popul_' + str(int(max_fitness)) + version_name
            with open(pop_name, "wb") as f:
                pickle.dump(popul_spr, f)

            max_fitness_prev = max_fitness

        if sum_fitness>sum_fitness_prev:

            best_solution=sol_next
            best_solution_spr = scipy.sparse.csr_matrix(best_solution)
            sol_name = 'solution_' + str(int(max_fitness)) + '__sum_' +str(int(sum_fitness)) + version_name
            with open(sol_name, "wb") as f:
                pickle.dump(best_solution_spr, f)
            popul_spr = scipy.sparse.csr_matrix(popul)
            pop_name = 'popul_' + str(int(max_fitness)) + '__sum_' + str(int(sum_fitness)) + version_name
            with open(pop_name, "wb") as f:
                pickle.dump(popul_spr, f)

            sum_fitness_prev = sum_fitness

    solution = sol_next

    if  prev_max_fit <  max_fitness:

        print(f'{g}__fitness__{np.round(fitness_vect)} __ {np.round(np.sum(fitness_vect))} _mr {mut_rate} _mrm= {mut_rate_mask}')
        print("popul_neur__",np.count_nonzero(popul))

        prev_max_fit =  max_fitness