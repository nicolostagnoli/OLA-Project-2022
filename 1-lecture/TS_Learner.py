from Learner import *

class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms) #calling the construction of super class passing the n_arms parameter
        self.beta_parameters = np.ones((n_arms,2)) # parameters of Beta distribution
        
    def pull_arms(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:,0],self.beta_parameters[:,1]))
        # Beta distributions are defined by 2 parameters: alpha and beta
        # Beta distribution draws the probability of each pulled arm
        return idx
        # 1 - 06 - slide 36 and ongoing
    
    def update(self, pulled_arms, reward):
        self.t += 1
        self.update_observations(pulled_arms,reward)
        self.beta_parameters[pulled_arms,0] = self.beta_parameters[pulled_arms,0] + reward
        self.beta_parameters[pulled_arms,1] = self.beta_parameters[pulled_arms,1] + 1.0 - reward