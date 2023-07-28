import numpy as np
import pandas as pd 
from scipy.stats import norm
from scipy.optimize import minimize

def gershman_fit(df,num_of_parameters_to_recover=2):

    # sample initial guess of the parameters to recover
    initial_guess = [np.random.uniform(0,4) for _ in range(num_of_parameters_to_recover)]
    initial_guess[1] = np.random.uniform(0,1)
    # set bounds to the recover parameters 
    bounds = [(0,4), (0,1)]
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

        p = norm.cdf( beta*(V/TU) + gamma*RU )
        action = action_list[t]
        choices_probs[t] = p if action==0 else 1-p
        
        reward = reward_list[t]

        kalman_gain = sigma[action] / (sigma[action]+tau[action])
        Q[action] = Q[action] + kalman_gain * (reward-Q[action])
        sigma[action] = sigma[action] - kalman_gain*sigma[action]

    eps = 1e-10
    log_loss = -(np.sum(np.log(choices_probs + eps)))
    return log_loss

