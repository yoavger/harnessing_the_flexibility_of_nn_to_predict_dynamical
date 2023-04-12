import numpy as np
import pandas as pd

# utility funcation for configuration, simulation, storing 

def create_reward_probs(n_action,trials_block,total_trails,p_option):
    reward_probs = np.zeros(shape=(n_action,total_trails))
    
    for b in range(int(total_trails/trials_block)):
        
        start = b*trials_block
        end = (b+1)*trials_block
        p_0 = np.random.choice(p_option)
        
        reward_probs[0,start:end] = np.repeat(p_0,trials_block)
        reward_probs[1,start:end] = np.repeat(1-p_0,trials_block)
        
    return reward_probs


def configuration_parameters():
    # configuration 2 free parameters (α, β)     
    parameters = {
                'alpha': np.random.uniform(0, 1), # alpha 
                'beta' : np.random.uniform(0, 10) # inverse temperature beta 
    }
    return parameters

