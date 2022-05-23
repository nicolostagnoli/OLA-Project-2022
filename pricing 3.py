import numpy as np

class Learner:
    def __init__(self, n_arms):
        self.t = 0
        self.n_arms = n_arms
        self.rewards = []
        self.reward_per_arm = [[] for _ in range(n_arms)]
        self.pulled = []  
    def reset(self):
        self.__init__(self.n_arms, self.prices, self.T)   
    def pull_arm(self):
        pass  
    def update(self, arm_pulled, reward):
        self.t += 1
        self.rewards.append(reward)
        self.reward_per_arm[arm_pulled].append(reward)
        self.pulled.append(arm_pulled)

class UCB(Learner):
    def __init__(self, n_arms, prices, T):
        super().__init__(n_arms)
        self.T = T
        self.means = np.zeros(n_arms)
        self.widths = np.array([np.inf for _ in range(n_arms)])
        self.prices = prices

    def pull_arm(self):
        idx = np.argmax((self.means+self.widths)*self.prices)
        return idx

    def update(self, arm_pulled, reward):
        reward = reward>0
        super().update(arm_pulled, reward)
        self.means[arm_pulled] = np.mean(self.reward_per_arm[arm_pulled])
        for idx in range(self.n_arms):
            n = len(self.reward_per_arm[idx])
            if n>0:
                self.widths[idx] = np.sqrt(2*np.log(self.T)/n)
            else:
                self.widths[idx] = np.inf

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

class UserClass:
  def __init__(self,number_of_user, alfas, num_products_bought, p_matrix ) :
    self.conversion_rate_matrix = np.random.uniform(0.0,1.0,(5,4))      #randomly generates the conversion rates for every price
    self.number_of_user = number_of_user
    self.alfas = alfas
    self.num_products_bought = num_products_bought
    self.p_matrix = p_matrix
  
class Product:
    def __init__(self,id, price_vector) :
      self.id = id
      self.price_vector = price_vector

def buy_or_not(prob):
    reward = np.random.binomial(1, prob) 
    return reward

def simulate_episode(init_prob_matrix, n_steps_max, initial_product, lambd, userClass, pulled_arms, prices, mabs):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]
    initial_active_nodes = np.zeros(5, dtype=int)
    initial_active_nodes[initial_product] = 1
    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes
    t=0
    buy_probability_matrix = userClass.conversion_rate_matrix
    rewards_per_product = np.zeros(5)
    while(t < n_steps_max and np.sum(newly_active_nodes) > 0):

        buy_or_not_nodes = np.zeros(5)
        for i, node in enumerate(active_nodes):
            if node == 1:
                buy = buy_or_not(buy_probability_matrix[i, pulled_arms[i]])
                buy_or_not_nodes[i] = buy
                if buy:
                    rewards_per_product[i] += uc.num_products_bought[i] * prices[i]
                mabs[i].update(pulled_arms[i], buy)
        p = (prob_matrix.T * buy_or_not_nodes).T

        #p is used to select from the prob matrix only the rows with active nodes  (.T compute the transpose)... returns the set of probabilities corresponding to the edges leaving from an active_node
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
        #lambda matrix has primary products on the rows, secondary on the columns, 1 if the secondary is in the first slot, lambda otherwise 
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

        #all edges leaving from the node that didn't buy go to 0 
    return history, rewards_per_product



p_matrix = np.array([[0,0.5,0,0.5,0],
                     [0.5,0,0.5,0,0],
                     [0,0.5,0,0,0.5],
                     [0.5,0,0,0,0.5],
                     [0,0.5,0,0.5,0]], dtype = float)
num_products_bought = [1, 1, 1, 1, 1]
userClass1 = UserClass(100, np.random.dirichlet([1, 1, 1, 1, 1]), num_products_bought, p_matrix )
userClass2 = UserClass(200, np.random.dirichlet([1, 1, 1, 1, 1]), num_products_bought, p_matrix )
userClass3 = UserClass(500, np.random.dirichlet([1, 1, 1, 1, 1]), num_products_bought, p_matrix )
userClasses = [userClass1, userClass2, userClass3]

price_vector = [10, 20, 30, 40]
p1 = Product("p1", price_vector )
p2 = Product("p2", price_vector )
p3 = Product("p3", price_vector )
p4 = Product("p4", price_vector )
p5 = Product("p5", price_vector )

n_episodes = 10
ucb1 = UCB(4, price_vector, n_episodes)
ucb2 = UCB(4, price_vector, n_episodes)
ucb3 = UCB(4, price_vector, n_episodes)
ucb4 = UCB(4, price_vector, n_episodes)
ucb5 = UCB(4, price_vector, n_episodes)
ts1 = TS_Learner(4)
ts2 = TS_Learner(4)
ts3 = TS_Learner(4)
ts4 = TS_Learner(4)
ts5 = TS_Learner(4)
ucbs = [ucb1, ucb2, ucb3, ucb4, ucb5]
tss = [ts1, ts2, ts3, ts4, ts5]

total_daily_users = userClass1.number_of_user + userClass2.number_of_user + userClass3.number_of_user

print("UCB: ")
for ep in range(n_episodes):

    pulled_prices = []
    for ucb in ucbs:
        pulled_arm = ucb.pull_arm()
        pulled_prices.append(pulled_arm)
    prices = [price_vector[p] for p in pulled_prices]
    
    total_rewards = np.zeros(5)
    for uc in userClasses:
      for j in range(0, uc.number_of_user):

        initial_product = np.random.choice(5, 1, [a for a in uc.alfas])[0]
        history, exp_rewards = simulate_episode(uc.p_matrix, 10, initial_product, 0.5, uc, pulled_prices, prices, ucbs)
        total_rewards += exp_rewards

    print("Daily rewards: ", total_rewards)

for i, ucb in enumerate(ucbs):
    print("Product ", i)
    print("Empirical Means: ", ucb.means)
    print("Confidence: ", ucb.widths)

