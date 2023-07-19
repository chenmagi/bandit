#!/usr/bin/env python3

#implement action-value method for k-armed bandit problem

import numpy as np
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from environment import BanditEnv

matplotlib.use('TkAgg')

def main():
    steps=2000
    k_bandit_env = BanditEnv(10,1.0)
    x_range=list(range(0,steps))
    epsilons=[0.0, 0.01, 0.1]
    colors=['blue','orange','green']
    
    y_collect=[]
    for epsilon in epsilons:
        records=k_bandit_env.run(epsilon,steps)
        y_collect.append(records)
    for i,y in enumerate(y_collect):
        label='e={}'.format(epsilons[i])
        plt.plot(x_range, y, label=label,color=colors[i])
    
    plt.legend()
    plt.title("10-armed bandit")
    plt.show()        
    
    
    

if __name__ == "__main__":
    main()