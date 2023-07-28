import numpy as np
from scipy.stats import norm

def transition_function(particles, action, reward, N): 
    
    kalman_gain = particles[action+2] / (particles[action+2]+particles[action+4])
    particles[action] = particles[action] + kalman_gain * (reward-particles[action])
    particles[action+2] = particles[action+2] - kalman_gain*particles[action+2]

    # beta gamma dynamics 
    particles[6] += np.random.normal(0, 0.1, N)
    particles[7] += np.random.normal(0, 0.1, N)

    return particles

def observation_likelihood(particles, action, N):

    V = particles[0] - particles[1]
    RU = np.sqrt(particles[2]) - np.sqrt(particles[3])
    TU = np.sqrt((particles[2]) + (particles[3]))

    beta = (np.exp(particles[6,:])).clip(0,4)
    gamma = (np.exp(particles[7,:]))

    p = norm.cdf( beta*(V/TU) + gamma*RU )

    likelihood = p if action==0 else 1-p

    return likelihood, p

def resampling_systematic(w, N):

    num_particles = len(w)
    u = np.random.random()/N
    edges = np.concatenate((0, np.cumsum(w)), axis=None)
    samples = np.arange(u, 1,step=(1/N))
    idx = np.digitize(samples,edges)-1
    return idx

def predict(particles, weights, observations, N):
    particles = transition_function(particles, observations[0], observations[1], N)
    state = particles@weights
    return state, particles

def correct(particles, observation, weights, N):
 
    likelihood, p_0 = observation_likelihood(particles, observation[0], N)
    weights = weights
    weights = weights*likelihood
    weights = weights/sum(weights)

    N_eff = 1/np.sum(weights**2)
    resample_percent = 0.50
    Nt = resample_percent*N
    idx = np.arange(N, dtype=int)
    if N_eff < Nt:
        idx = resampling_systematic(weights,N)
        weights = np.ones(N)/N
        particles = particles[:, idx]

    return idx, particles, weights, likelihood, p_0
    
def bayesian_fit(obs):  
    
    N = 1_000

    # create 
    particles = np.zeros(shape=(8,N))

    particles[0] = np.zeros(N) # q_0 
    particles[1] = np.zeros(N) # q_1 

    particles[2] = np.zeros(N) # sigma_0 
    particles[3] = np.zeros(N) # sigma_1 

    particles[4] = np.zeros(N) # tau_0 
    particles[5] = np.zeros(N) # tau_1 

    particles[6] = np.random.normal(0,1,N) # beta
    particles[7] = np.random.normal(-2,1,N) # gamma

    num_observations = len(obs)
    observations = obs

    state_arr = np.zeros((num_observations, 8))
    weights = np.ones(N)/N
    weights_arr = np.zeros((num_observations, N))
    likelihood_arr = np.zeros((num_observations, N))
    p_0_arr = np.zeros((num_observations, N))

    
    for t in range(num_observations):

        if t%10 == 0:
            particles[0] = np.random.normal(0,0,N)
            particles[1] = np.random.normal(0,0,N)

            particles[2] = np.repeat(100,N)
            particles[3] = np.repeat(100,N)

            particles[4] = np.repeat(10,N)
            particles[5] = np.repeat(10,N)
            
        idx, particles, weights, likelihood, p_0 = correct(particles, observations[t,:2], weights, N)
        state, particles = predict(particles, weights, observations[t,:2], N)
        
        state_arr[t,:] = state
        weights_arr[t,:] = weights
        likelihood_arr[t,:] = likelihood
        p_0_arr[t,:] = p_0
        aLast = observations[t][0]

    return state_arr, likelihood_arr, p_0_arr
