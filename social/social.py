import numpy as np
from copy import copy

def simulate_episode(init_prob_matrix, n_steps_max, initial_product, lambd):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]
    initial_active_nodes = np.zeros(5, dtype=int)
    initial_active_nodes[initial_product] = 1
    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes
    t=0
    while(t < n_steps_max and np.sum(newly_active_nodes) > 0):
        #print("giro ", t)
        #print(prob_matrix)
        p = (prob_matrix.T * active_nodes).T
        #print("p matrix : \n", p)
        lambda_vector = np.zeros(n_nodes)
        lambda_matrix = np.zeros(shape = (n_nodes, n_nodes))
        for i in range(0, n_nodes):
            idx1 = -1
            idx2 = -1
            lambda_vector = np.zeros(5)
            while(idx1 == idx2 or idx1 == i or idx2 == i):
                [idx1, idx2] = np.random.choice(5, 2)
            lambda_vector[idx1] = 1
            lambda_vector[idx2] = lambd
            lambda_matrix[i] = lambda_vector
        #print("lambda matrix : \n", lambda_matrix)
        p = (p * lambda_matrix)
        #print("p after lambda vec : \n", p)
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        #print("activated edges : \n", activated_edges)
        prob_matrix = prob_matrix * ((p!=0) == activated_edges)
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
        active_nodes = np.array(active_nodes + newly_active_nodes)
        #print("active nodes at t+1: \n", active_nodes)
        history = np.concatenate((history, [newly_active_nodes]), axis = 0)
        #print("history : \n", history)
        t += 1
    return history
    
def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_prob = np.ones(n_nodes) *1.0/(n_nodes - 1)
    credits = np.zeros(n_nodes)
    occurr_v_active = np.zeros(n_nodes)
    for episode in dataset:
        idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
        if len(idx_w_active)>0 and idx_w_active>0:
            active_nodes_in_prev_step = episode[idx_w_active - 1,:].reshape(-1)
            credits += active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step)
        for v in range(0, n_nodes):
            if(v!=node_index):
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len( idx_v_active) >0 and(len(idx_w_active)==0 or idx_v_active <  idx_w_active):
                    occurr_v_active[v]+=1
    estimated_prob = credits/occurr_v_active
    estimated_prob = np.nan_to_num(estimated_prob)
    return estimated_prob

n_nodes = 5
n_episodes = 50000
prob_matrix = np.random.uniform(0.0, 1.0, (n_nodes, n_nodes))
for i in range(0, n_nodes):
    prob_matrix[i][i] = 0
alphas = np.random.dirichlet([1, 1, 1, 1, 1])
node_index = 1
lambd = 0.5
dataset = []

for e in range(0, n_episodes):
    initial_product = np.random.choice(5, 1, [a for a in alphas])[0]
    dataset.append(simulate_episode(init_prob_matrix = prob_matrix, n_steps_max=3, initial_product = initial_product, lambd=lambd))

estimated_matrix = np.empty(shape=(n_nodes, n_nodes))
for n in range(0, 5):
    estimated_prob = estimate_probabilities(dataset=dataset, node_index=n, n_nodes=n_nodes)
    estimated_matrix[:, n] = estimated_prob

print("True P Matrix: \n", prob_matrix)
print("Estimated P Matrix: \n", estimated_matrix)