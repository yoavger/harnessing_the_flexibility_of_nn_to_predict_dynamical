import numpy as np
import pandas as pd

def q_sim_rw(index_agent ,parameters, num_of_trials, expected_reward, param_drift_rate):

    data = { 
    'agent':[],
    'block':[],
    'trial':[],
    'action':[],
    'reward':[],
    'p_0':[],
    'drift_0':[],
    'drift_1':[],
    'Q_0':[],
    'Q_1':[],
    'alpha':[],
    'beta':[]
    }

    # set up parameters
    alpha = np.random.normal(0,1) # parameters['alpha']
    beta = np.random.normal(1,1)  # parameters['beta']
    
    # q_values 
    q = np.zeros(2) 
    block = -1

    for t in range(num_of_trials):

        if t%100 == 0:
            q = np.zeros(2)
            block+=1
        
        beta_tran = np.exp(beta).clip(0,10)
           
        # calc prob with softmax 
        p = np.exp( beta_tran*q ) / np.sum( np.exp( beta_tran*q ) ) 
        
        # choose action according to prob 
        action = np.random.choice([0,1] , p=p)

        probability_reward = [ (1-expected_reward[action, t]), expected_reward[action, t]]
       
        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=probability_reward)     
        
        # prediction error
        prediction_error = reward - q[action] 
        
        alpha_tran = 1 / (1 + np.exp(-alpha))
        
        # update q values 
        q[action] = q[action] + alpha_tran*prediction_error 

        # stroe data of the trial
        data['agent'].append(index_agent)
        data['block'].append(block-1)
        data['trial'].append(t)
        data['action'].append(action)
        data['reward'].append(reward)
        data['p_0'].append(p[0])
        data['drift_0'].append(expected_reward[0,t])
        data['drift_1'].append(expected_reward[1,t])
        data['Q_0'].append(q[0])
        data['Q_1'].append(q[1])
        data['alpha'].append(alpha_tran)
        data['beta'].append(beta_tran)
        
        alpha += np.random.normal(0,param_drift_rate[0])
        beta += np.random.normal(0,param_drift_rate[1])

    df = pd.DataFrame(data)
    
    return df