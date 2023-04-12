import numpy as np
import pandas as pd

# utility funcation for configuration, simulation, storing the data  

def create_reward_probs(n_action, num_of_trials):

    reward = np.zeros(shape=(n_action,num_of_trials))

    six_option = np.array(
        [[.25,.05],
         [.125,.05],
         [.08,.05],
         [.05,.25],
         [.05,.125],
         [.05,.08],
         [.25,.05],
         [.125,.05],
         [.08,.05],
         [.05,.25],
         [.05,.125],
         [.05,.08]]
    )

    permut = np.random.permutation(six_option)

    for b in range(12):
        cur = permut[b]
        reward[0][b*100:(b+1)*100] = np.repeat(cur[0],100)
        reward[1][b*100:(b+1)*100] = np.repeat(cur[1],100)

    return reward


def configuration_parameters():
    # 3 free parameters (α, β, κ)     
    parameters = {
                'alpha' : np.random.uniform(0,0.2), # 0 <= alpha <= .2 learning rate
                'beta' : np.random.uniform(0,10), # 0 <= beta <= 10 inverse temperature  
                'kappa' : np.random.uniform() - .5 # -.5 <= kappa <= .5 preservation
    }
    return parameters

