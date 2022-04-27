from Learner import *

class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms) #calling the construction of super class passing the n_arms parameter
        self.expected_rewards = np.zeros((n_arms))
        
    def pull_arm(self):
        # With greedy learner we need to ensure that each arm is pulled at least once
        # this code initialize to pull arm 0 at time 0 and so on
        if(self.t < self.n_arms):
            return self.t
        # choose arm with the highest reward
        idx = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        # since there can be many arms with highest value, we need to pick one of those randomly
        pulled_arm = np.random.choice(idx)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm]*(self.t - 1) + reward) / self.t
        