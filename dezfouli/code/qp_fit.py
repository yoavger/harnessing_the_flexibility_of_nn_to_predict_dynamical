import numpy as np
import pandas as pd
from scipy.optimize import minimize

def qp_fit(df,num_of_parameters_to_recover=3):

    # sample initial guess of the parameters to recover
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[0] = np.random.uniform(0,0.2)
    initial_guess[1] = np.random.uniform(0,10)
    
    # set bounds to the recover parameters 
    bounds = [(0,0.2), (0,10), (0,1)] 
    res = minimize(
                    fun=parameters_recovary,
                    x0=initial_guess,
                    args=df,
                    bounds=bounds,
                    method='L-BFGS-B'
    )
    return res

def parameters_recovary(parameters, df):

    # objective to minimize
    log_loss = 0 
    
    num_of_trials = len(df)
    choices_probs = np.zeros(num_of_trials)
    
    # upload data of the subject/agent
    action_list = df['action'].astype(int)
    reward_list = df['reward'].astype(np.float32)
    block_list = df['block'].astype(int)

    # set up paramters for recovary    
    alpha = parameters[0] 
    beta = parameters[1]
    kappa = parameters[2] - .5

    # initialize q-values
    q = np.zeros(2)
    pers_array = np.zeros(2)
    
    block = 0

    for t in range(num_of_trials):
        
        if block_list[t] == block:
            q = np.zeros(2)
            pres_array = np.zeros(2)
            block += 1
            if block == 12:
                block = 0
            
        if t > 0 and block_list[t] == block_list[t-1]:
            pers_array[action] = kappa
            pers_array[1-action] = 0

        # get true first action
        action = action_list[t]
        choices_probs[t] = np.exp( beta * ( q[action] + pers_array[action] ) ) / np.sum( np.exp( beta* (q+pers_array) ) ) 

        # get true reward
        reward = reward_list[t]

        # prediction error
        prediction_error = reward - q[action] 

        # update q_learning formula
        q[action] = q[action] + alpha*prediction_error 
        
    eps = 1e-10
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss



