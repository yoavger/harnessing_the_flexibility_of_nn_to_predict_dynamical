import numpy as np
import pandas as pd

def qp_pred(df,parameters):

    # counter of the number of action classified correctly (accuracy)
    accuracy = 0 
    num_of_trials = len(df)
    choices_probs_0 = np.zeros(num_of_trials)

    # upload data of the subject/agent
    action_list = df['action'].astype(int)
    reward_list = df['reward'].astype(np.float32)
    block_list = df['block'].astype(int)
 
    # set up paramters of the agent     
    alpha = parameters[0] 
    beta = parameters[1]
    kappa = parameters[2] -.5
    
    # initialize q-values and preservation
    q = np.zeros(2)
    pres_array = np.zeros(2)
    
    block = 0

    for t in range(num_of_trials):
            
        if block_list[t] == block:
            q = np.zeros(2)
            pres_array = np.zeros(2)
            block += 1
            if block == 12:
                block = 0
            
        if t > 0 and block_list[t] == block_list[t-1]:
            pres_array[action] = kappa
            pres_array[1-action] = 0

        p = np.exp( beta* (q+pres_array)  ) / np.sum( np.exp( beta * (q+pres_array) ) ) 
        
        # predict action according max probs 
        action_predict = np.argmax(p)
        choices_probs_0[t] = p[0]

        # get true action and reward 
        action = action_list[t]
        reward = reward_list[t]

        # prediction error
        prediction_error = reward - q[action] 

        # update q_learning formula
        q[action] = q[action] + alpha*prediction_error 

        # cheek if prediction match the true action
        if action_predict == action_list[t]:
            accuracy+=1
            
    return (accuracy/num_of_trials), choices_probs_0 
