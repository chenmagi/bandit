# an agent proceeds the action-value method

import numpy as np


class ActionValue_Agent:
    arms = 10
    hist_rewards = None
    last_action = 0
    def __init__(self,kvalue):
        self.arms = kvalue
        self.hist_rewards = np.array([[0,0.0] for i in range(kvalue)]) # [0] for historical use count, [1] for historical accumulated rewards 
        return
    
    def __exploration(self, est_values)->int:
        discard = np.argmax(est_values)
        
        others = list(range(0,discard))+list(range(discard+1,self.arms))
        rand = np.random.randint(0,self.arms-1)
                
        return others[rand]
    
    def __exploitation(self, est_values)->int:    
        return np.argmax(est_values)
    

    def __update_hist(self,id,reward):
        self.hist_rewards[id][0]+=1
        self.hist_rewards[id][1]+=reward
        return
    
    def __estimated_values(self) -> np.ndarray:
        ret = np.array([0 for i in range(self.arms)])
        for k in range(self.arms):
            if self.hist_rewards[k][0] > 0:
                ret[k]=self.hist_rewards[k][1]/self.hist_rewards[k][0]
        return ret
            
        
    
    def forward(self,reward, epsilon, time)->int:
        if time == 0:
            np.ndarray.fill(self.hist_rewards,0)
            init_A = np.random.randint(0,self.arms)
            self.last_action = init_A
            return init_A
        self.__update_hist(self.last_action,reward)
        
        tmp_values = self.__estimated_values()
        prob = np.random.rand()
        
        if prob > epsilon:
            self.last_action = self.__exploitation(tmp_values)
        else:
            self.last_action = self.__exploration(tmp_values)
        
        
        return self.last_action
    
    def output_avgreward(self):
        acc=np.sum(self.hist_rewards,axis=0)
        #print('acc={}  {}'.format(acc[0],acc[1]))
        #return the average rewards
        if acc[0] == 0: return 0
        return acc[1]/acc[0]
    
    def terminate(self):
        #print(self.hist_rewards)
        #print("sum of rewards={:.4f}".format(np.sum(self.hist_rewards,axis=0)[1]))
        #acc=np.sum(self.hist_rewards,axis=0)
        #print('acc={}  {}'.format(acc[0],acc[1]))
        return
    
   
    
        
    
    


