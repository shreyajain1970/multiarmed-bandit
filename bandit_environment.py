import numpy as np

class Bandit:
    def __init__(self,k):
        self.k=k
        self.true_values=np.random.normal(0,1,k)
    def pull(self,action):
        reward=np.random.normal(self.true_values[action],1)
        return reward