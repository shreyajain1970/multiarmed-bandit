import numpy as np

class Random_Agent:
    def __init__(self,k):
        self.k=k
    
    def select_action(self):
        action= np.random.randint(self.k)
        return action
    
    def update(self,action,reward):
        pass


class Epsilon_Greedy_Agent:
    def __init__(self,k,e):
        self.k=k
        self.epsilon=e
        self.N_values=np.zeros(k)        #records the number of times a particular action is chosen
        self.Q_values=np.zeros(k)        #records the predicted value of the mean on the basis of previous results

    def select_action(self):
        if np.random.rand() < self.epsilon:
            action= np.random.randint(self.k)
            return action
        else:
            action=np.argmax(self.Q_values)
        return action

    def update(self,action,reward):
        n=self.N_values[action]
        q=self.Q_values[action]
        self.N_values[action]+=1
        q=((q*n)+reward)/(n+1)
        self.Q_values[action]=q


class Softmax_Agent:
    def __init__(self,k,tau):
        self.k=k
        self.temp=tau
        self.N_values=np.zeros(k)        #records the number of times a particular action is chosen
        self.Q_values=np.zeros(k)        #records the predicted value of the mean on the basis of previous results

    def select_action(self):
        probs=np.exp(self.Q_values/self.temp)/np.sum(np.exp(self.Q_values/self.temp))
        action=np.random.choice(len(self.Q_values),p=probs)
        return action

    def update(self,action,reward):
        n=self.N_values[action]
        q=self.Q_values[action]
        self.N_values[action]+=1
        q=((q*n)+reward)/(n+1)
        self.Q_values[action]=q


class UCB_Agent:
    def __init__(self,k,c):
        self.k=k
        self.c=c
        self.N_values=np.zeros(k)        #records the number of times a particular action is chosen
        self.Q_values=np.zeros(k)        #records the predicted value of the mean on the basis of previous results

    def select_action(self,t):
        ucb=np.zeros(self.k)
        for i in range(self.k):
            ucb[i]=self.Q_values[i] + self.c * np.sqrt(np.log(t+1)/(self.N_values[i] + 1e-5))
        action=np.argmax(ucb)
        return action

    def update(self,action,reward):
        n=self.N_values[action]
        q=self.Q_values[action]
        self.N_values[action]+=1
        q=((q*n)+reward)/(n+1)
        self.Q_values[action]=q








