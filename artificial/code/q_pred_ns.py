import numpy as np
import pandas as pd

def q_pred_ns(df, parameters):

    # counter of the number of action classified correctly (accuracy)
    accuracy = 0 
    num_of_trials = len(df)
    choices_probs_0 = np.zeros(num_of_trials)
    
    # upload data of the subject/agent
    action_list = df['action'].astype(int)
    reward_list = df['reward'].astype(np.float32)

    # initialize q-values and preservation
    q = np.zeros(2)

    for t in range(num_of_trials):
        
        alpha = parameters[t,0]
        beta = parameters[t,1]
        
        if t%100 == 0:
            q = np.zeros(2)

        p = np.exp( beta*q ) / np.sum( np.exp( beta * q ) ) 
        
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

