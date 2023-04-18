import numpy as np
import pandas as pd
from scipy.stats import norm

def hybrid_pred(df, parameters):

    # counter of the number of action classified correctly (accuracy)
    accuracy = 0 
    num_of_trials = len(df)
    choices_probs_0 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_list = df['action'].astype(int)
    reward_list = df['reward'].astype(np.float32)
 
    # set up paramters
    beta = parameters[0]
    gamma = parameters[1]
    
    for t in range(num_of_trials):
        
        if t%10 == 0:
            Q = np.zeros(2)
            sigma = np.zeros(2)
            sigma[0] = 100
            sigma[1] = 100

            tau = np.zeros(2)
            tau[0] = 10
            tau[1] = 10

        V = Q[0] - Q[1]
        RU = np.sqrt(sigma[0]) - np.sqrt(sigma[1])
        TU = np.sqrt((sigma[0]) + (sigma[1]))

        p = norm.cdf(beta*(V/TU) + gamma*RU )
        action_predict = np.argmax([p, 1-p])
        choices_probs_0[t] = p
        
        action = action_list[t]
        reward = reward_list[t]

        kalman_gain = sigma[action] / (sigma[action]+tau[action])
        Q[action] = Q[action] + kalman_gain * (reward-Q[action])
        sigma[action] = sigma[action] - kalman_gain*sigma[action]
        
        # cheek if prediction match the true action
        if action_predict == action_list[t]:
            accuracy+=1
            
    return (accuracy/num_of_trials), choices_probs_0 
    


    
