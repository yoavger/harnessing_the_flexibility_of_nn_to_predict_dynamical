import numpy as np
import pandas as pd

def q_sim(index_agent ,parameters, num_of_trials, expected_reward,
          probability_to_switch_parameters, max_change):
    
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
    alpha = parameters['alpha']
    beta = parameters['beta']
    
    c_a , c_b = 0 , 0 
    cc_a , cc_b = 0 , 0 

    # q_values 
    q = np.zeros(2) 
    
    block = -1
    
    for t in range(num_of_trials):
      
        if t%100 == 0:
            q = np.zeros(2)
            block+=1
    
        cc_a +=1
        cc_b +=1
        
        # switch alpha 
        if cc_a > 100 and c_a < max_change and np.random.random() < probability_to_switch_parameters:
            alpha = np.random.uniform(0, 1)
            c_a += 1
            cc_a = 0

        # switch beta 
        if cc_b > 100 and c_b < max_change and np.random.random() < probability_to_switch_parameters:
            beta = np.random.uniform(0, 10)
            c_b += 1
            cc_b = 0
            
        # calc prob with softmax 
        p = np.exp( beta*q ) / np.sum( np.exp( beta*q ) ) 
        
        # choose action according to prob 
        action = np.random.choice([0,1] , p=p)

        probability_reward = [ (1-expected_reward[action, t]), expected_reward[action, t]]
       
        # check if the trial is rewarded
        reward = np.random.choice([0,1] , p=probability_reward)     
        
        # prediction error
        prediction_error = reward - q[action] 
        
        # update q values 
        q[action] = q[action] + alpha*prediction_error 
        
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
        data['alpha'].append(alpha)
        data['beta'].append(beta)

    df = pd.DataFrame(data)
    return df
