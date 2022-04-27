from scipy.optimize import linear_sum_assignment
from Non_Stationary_Environment import Non_Stationary_Environment
from CUSUM_UCB_Matching import CUSUM_UCB_Matching
from UCB_Matching import UCB_Matching
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p0= np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
    p1 = np.array([[1, 1/4, 1/4], [1/2, 1/4, 1/4], [1/4, 1/4, 1]])
    p2 = np.array([[1, 1/4, 1/4], [1/2, 1, 1/4], [1/4, 1/4, 1]])
    P = [p0, p1, p2] # buyers are matched with sellers 
    # we need to find a match that maximizes the probability of a buy-sell
    T = 15
    n_exp = 5 #(100)
    regret_cusum = np.zeros((n_exp, T))
    regret_ucb = np.zeros((n_exp, T))
    detections = [[] for _ in range(n_exp)]
    M = 100
    eps = 0.1
    h = np.log(T)*2
    for j in range(n_exp):
        e_UCB = Non_Stationary_Environment(n_arms=p0.size, probabilities=P, horizon=T)
        e_CD = Non_Stationary_Environment(n_arms=p0.size, probabilities=P, horizon=T)
        learner_CD = CUSUM_UCB_Matching(p0.size, *p0.shape, M, eps, h) #n_arms, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01
        learner_UCB = UCB_Matching(p0.size, *p0.shape)
        opt_rew = []
        rew_CD = []
        rew_UCB = []
        for t in range(T):
            p = P[int(t / e_UCB.phase_size)] #choose the probability based on the curr phase
            opt = linear_sum_assignment(-p)
            opt_rew.append(p[opt].sum())

            pulled_arm = learner_CD.pull_arm()
            reward = e_CD.round(pulled_arm)
            learner_CD.update(pulled_arm, reward)
            rew_CD.append(reward.sum())

            pulled_arm = learner_UCB.pull_arm()
            reward = e_UCB.round(pulled_arm)
            learner_UCB.update(pulled_arm, reward)
            rew_UCB.append(reward.sum())
        regret_cusum[j, :] = np.cumsum(opt_rew)-np.cumsum(rew_CD)
        regret_ucb[j,:] = np.cumsum(opt_rew)-np.cumsum(rew_UCB)
        print(f'Experiment #{j} has finished')


plt.figure(0)
plt.ylabel('Regret')
plt.xlabel("t")
plt.plot(np.mean(regret_cusum, axis=0))
plt.plot(np.mean(regret_ucb, axis=0))
plt.show()