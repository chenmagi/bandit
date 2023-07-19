import numpy as np
import math
import agent


class BanditEnv:
    arms = 10
    variance=1
    rewards=[]
    reward_size=100
    av_agent = None
    
    def __init__(self, kvalue=10, variance=1):
        self.arms = kvalue
        self.variance=1
        self.regenerate_rewards()
        self.av_agent = agent.ActionValue_Agent(kvalue)
        
    def __interact(self,action_id):
        return np.random.choice(self.rewards[action_id])
    
    def regenerate_rewards(self):
        for k in range(self.arms):
            
            random_mean = np.random.randint(-15,20)/10.0
            prob_dist = np.random.normal(random_mean,  math.sqrt(self.variance),self.reward_size)
            self.rewards.append(prob_dist)
        
    
    def run(self,epsilon, steps):
        avgrewards=[]
        reward=-100.000001  # an extra small value for initial reward
        for t in range(steps):
            next_action=self.av_agent.forward(reward,epsilon,t)
            reward=self.__interact(next_action)
            avgrewards.append(self.av_agent.output_avgreward())
            
        self.av_agent.terminate()
        return avgrewards
           
        
        
            