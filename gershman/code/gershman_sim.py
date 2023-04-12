import numpy as np
import pandas as pd 
from scipy.stats import norm

def gershman_sim(index_agent, parameter, n_blocks, n_trials,
                 probability_to_switch_parameters, max_change):


    data = { 
        'agent':[],
        'block':[],
        'trial':[],
        'mu_0':[],
        'mu_1':[],
        'action':[],
        'reward':[],
        'p_0':[],
        'Q_0':[],
        'Q_1':[],
        'sigma_0':[],
        'sigma_1':[],
        'kalma_gain':[],
        'V':[],
        'RU':[],
        'TU':[],
        'beta':[],
        'gamma':[]
    }
    
    beta = parameter[0]
    gamma = parameter[1]
    
    c_b, c_g = 0, 0
    cc_b, cc_g = 0, 0
    
    for block in range(n_blocks):
        
        Q = np.zeros(2)
        sigma = np.zeros(2)
        sigma[0] = 100
        sigma[1] = 100

        tau = np.zeros(2)
        tau[0] = 10
        tau[1] = 10

        mu = np.round(np.random.normal(0,10,2))
        var = np.zeros(2) + 10
        
        for trial in range(n_trials):
            
 
            cc_b+=1
            cc_g+=1
            
            # switch beta 
            if cc_b > 10 and c_b < max_change and np.random.random() < probability_to_switch_parameters:
                beta = np.random.uniform(0,4)
                c_b += 1
                cc_b = 0

            # switch gamma 
            if cc_g > 10 and c_g < max_change and np.random.random() < probability_to_switch_parameters:
                gamma = np.random.uniform(0,1) 
                c_g += 1
                cc_g = 0
                       
            V = Q[0] - Q[1]
            RU = np.sqrt(sigma[0]) - np.sqrt(sigma[1])
            TU = np.sqrt((sigma[0]) + (sigma[1]))

            p = norm.cdf(beta*(V/TU) + gamma*RU )

            if np.random.random() < p:
                action = 0
            else:
                action = 1

            reward = np.round(np.random.normal(mu[action],var[action]))

            kalman_gain = sigma[action] / (sigma[action]+tau[action])
            Q[action] = Q[action] + kalman_gain * (reward-Q[action])
            sigma[action] = sigma[action] - kalman_gain*sigma[action]

            data['agent'].append(index_agent)
            data['block'].append(block)
            data['trial'].append(trial)
            data['mu_0'].append(mu[0])
            data['mu_1'].append(mu[1])
            data['action'].append(action)
            data['reward'].append(reward)
            data['p_0'].append(p)
            data['Q_0'].append(Q[0])
            data['Q_1'].append(Q[1])
            data['sigma_0'].append(sigma[0])
            data['sigma_1'].append(sigma[1])
            data['kalma_gain'].append(kalman_gain)
            data['V'].append(V)
            data['RU'].append(RU)
            data['TU'].append(TU)
            data['beta'].append(beta)
            data['gamma'].append(gamma)
            

    df = pd.DataFrame(data)

    return df
    
