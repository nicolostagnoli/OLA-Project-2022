from Environment import *
import numpy as np

class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0
        n_phases = len(self.probabilities)
        self.phase_size = horizon/n_phases

    def round(self, pulled_arms):
        current_phase = int(self.t / self.phase_size)
        p = self.probabilities[current_phase][pulled_arms]
        reward = np.random.binomial(1, p)
        self.t += 1
        return reward

    