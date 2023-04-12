import numpy as np
import pandas as pd
from scipy.optimize import minimize

def q_fit(df,num_of_parameters_to_recover=2):

    # sample initial guess of the parameters to recover
    initial_guess = [np.random.uniform(0,1) for _ in range(num_of_parameters_to_recover)]
    initial_guess[1] = np.random.uniform(0,10)
    
    # set bounds to the recover parameters 
    bounds = [(0,1), (0,10)] 
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

    # set up paramters for recovary    
    alpha = parameters[0] 
    beta = parameters[1]

    # initialize q-values
    q = np.zeros(2)

    for t in range(num_of_trials):
        
        if t%100 == 0:
            q = np.zeros(2)

        # get true first action
        action = action_list[t]
        choices_probs[t] = np.exp( beta * q[action] ) / np.sum( np.exp( beta*q ) ) 

        # get true reward
        reward = reward_list[t]

        # prediction error
        prediction_error = reward - q[action] 

        # update q_learning formula
        q[action] = q[action] + alpha*prediction_error 
        
    eps = 1e-10
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss



