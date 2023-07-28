import numpy as np

def transition_function(particles, action, reward, N): 
    
    alpha = 1 / (1 + np.exp(-particles[2,:]))
    
    # q dynamics 
    particles[action,:] = particles[action,:] + alpha*(reward - particles[action,:])
    
    # alpha beta kappa dynamics 
    particles[2] += np.random.normal(0, 0.05, N)
    particles[3] += np.random.normal(0, 0.005, N)
    particles[4] += np.random.normal(0, 0.05, N)

    return particles

def observation_likelihood(particles, action, N, aLast):
    
    q_0 = particles[0,:]
    q_1 = particles[1,:]    
        
    beta = (np.exp(particles[3,:])).clip(0,10)
    kappa = ( 1 / ( 1 + np.exp( - (particles[4,:])) ) ) - .5

    pres_array = np.zeros(shape=(2,N))
    
    if aLast > -1:
        pres_array[aLast] = kappa
        
    p_0 = (np.exp( beta*( q_0 + pres_array[0]))) /\
                                    ( np.exp( beta*( q_0 + pres_array[0]) ) + np.exp( beta*(q_1 + pres_array[1]) ) )
    
    likelihood = p_0 if action==0 else 1-p_0
    
    return likelihood, p_0

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

def correct(particles, observation, weights, N, aLast):
 
    likelihood, p_0 = observation_likelihood(particles, observation[0], N, aLast)
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
    particles = np.zeros(shape=(5,N))
    
    particles[0] = np.random.normal(0,0,N) # q_0 
    particles[1] = np.random.normal(0,0,N) # q_1 
    
    particles[2] = np.random.normal(-2,1.5,N) # alpha 
    particles[3] = np.random.normal(1,1,N) # beta  
    particles[4] = np.random.normal(0,1.5,N) # kappa

    num_observations = len(obs)
    observations = obs

    state_arr = np.zeros((num_observations, ß5))
    weights = np.ones(N)/N
    weights_arr = np.zeros((num_observations, N))
    likelihood_arr = np.zeros((num_observations, N))
    p_0_arr = np.zeros((num_observations, N))

    aLast = -1
    block = 0
    
    for t in range(num_observations):

        if observations[t,2] == block:
            particles[0] = np.random.normal(0,0,N)
            particles[1] = np.random.normal(0,0,N)
            block += 1
            aLast = -1
        
        idx, particles, weights, likelihood, p_0 = correct(particles, observations[t,:2], weights, N, aLast)
        state, particles = predict(particles, weights, observations[t,:2], N)
        
        state_arr[t,:] = state
        weights_arr[t,:] = weights
        likelihood_arr[t,:] = likelihood
        p_0_arr[t,:] = p_0
        aLast = observations[t][0]

    return state_arr, likelihood_arr, p_0_arr
