import numpy as np
from copy import copy

def simulate_episode(init_prob_matrix, n_steps_max):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]
    # active nodes at time 0
    # n is the number of trials
    # p is the probability of success
    initial_active_nodes = np.random.binomial(n=1, p=0.1, size=(n_nodes))
    # dataset that we want to exploit to  estimate probabilities
    history = np.array([initial_active_nodes])
    # all activated nodes in our episode
    active_nodes = initial_active_nodes
    # nodes that has been activated in this timestep
    newly_active_nodes = active_nodes
    t=0
    while(t < n_steps_max and np.sum(newly_active_nodes) > 0):
        # select from probability matrix only the rows related to active nodes
        p = (prob_matrix.T * active_nodes).T
        # compute value of activated edges by sampling a value from a random distribution from 0 to 1
        # and comparing this value with the probability of each edge
        # if the value of the probability is bigger than the one drawn from the distribution
        # then the corresponding edges will be activated
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        # remove from prob_matrix all the values related to previously activated nodes
        prob_matrix = prob_matrix * ((p!=0) == activated_edges) 
        # compute the value of newly activated nodes: sum the values of activated edges matrix in 0 axis
        # since we are saying that a node is activated if at least one edge was activated in the prev step
        # then we multiply it by (1 - active_nodes) in this way we are saying that newly active nodes can be 
        # previously activated nodes
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
        # we update the active nodes by adding newly_active_nodes
        active_nodes = np.array(active_nodes + newly_active_nodes)
        # append to history dataset the newly active nodes
        history = np.concatenate((history, [newly_active_nodes]), axis = 0)
        t += 1
    return history
    
def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_prob = np.ones(n_nodes) *1.0/(n_nodes - 1)
    credits = np.zeros(n_nodes)
    occurr_v_active = np.zeros(n_nodes)
    n_episodes = len(dataset)
    for episode in dataset:
        idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
        if len(idx_w_active)>0 and idx_w_active>0:
            active_nodes_in_prev_step = episode[idx_w_active - 1,:].reshape(-1)
            credits += active_nodes_in_prev_step/np.sum(active_nodes_in_prev_step)
        for v in range(0, n_nodes):
            if(v!=node_index):
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len( idx_v_active) >0 and( idx_v_active <  idx_w_active or len(idx_w_active)==0):
                    occurr_v_active[v]+=1
    estimated_prob = credits/occurr_v_active
    estimated_prob = np.nan_to_num(estimated_prob)
    return estimated_prob

n_nodes = 5
n_episodes = 1000
prob_matrix = np.random.uniform(0.0,0.1,(n_nodes, n_nodes))
node_index = 4
dataset = []

for e in range(0, n_episodes):
    dataset.append(simulate_episode(init_prob_matrix=prob_matrix, n_steps_max=10))

estimated_prob = estimate_probabilities(dataset=dataset, node_index=node_index, n_nodes=n_nodes)

print("True P Matrix: ", prob_matrix[:, 4])
print("Estimated P Matrix: ", estimated_prob)