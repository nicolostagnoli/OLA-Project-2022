import numpy as np
class Environment():
    # define the class
    def __init__ (self, n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities
        '''
        model interaction with the learner 
        this function takes an input of super arms
        called pulled_arms and returns a stochastic reward
        '''
    def round(self, pulled_arms):
        reward = np.random.binomial(1, self.probabilities[pulled_arms]) 
        return reward
        '''
        1 - Bernoulli distribution, 
        success probability related to super arm, that we 
        identified in the construction
        ''' 
# ----- the end of implementation of environment class

'''
a learner object is defined by 
    the number of arms that he can pull
    the current round
    the list of the collected rewards

the learner interacts with the environment by selecting the arm to pull
at each rounf and observing the reward given by the environment
'''