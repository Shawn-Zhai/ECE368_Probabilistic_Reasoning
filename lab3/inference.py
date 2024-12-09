import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # TODO: Compute the forward messages
    
    # Initialization
    #forward_messages[0] = rover.Distribution({})
    x0y0 = observations[0]
    for z0 in all_possible_hidden_states:
        
        if x0y0 != None:
            InitProbPositionGivenState = observation_model(z0)[x0y0]
        
        # missing, uninformative, 1
        else:
            InitProbPositionGivenState = 1
            
        Pz0 = prior_distribution[z0]
        forward_messages[0][z0] = InitProbPositionGivenState * Pz0
            
    forward_messages[0].renormalize()
    
    # i >= 1
    for i in range(1, num_time_steps):
        
        forward_messages[i] = rover.Distribution({})
        observation = observations[i]
        
        for zi in all_possible_hidden_states:
            
            if observation != None:
                probPositionGivenState = observation_model(zi)[observation]
                
            # missing, uninformative, 1
            else:               
                probPositionGivenState = 1
            
            summation = 0
            for zi_prev in forward_messages[i - 1]:
                summation += forward_messages[i - 1][zi_prev] * transition_model(zi_prev)[zi]
            
            # ignore 0
            if probPositionGivenState * summation != 0:
                forward_messages[i][zi] = probPositionGivenState * summation

        # normalize forward messages
        forward_messages[i].renormalize() 
                   
    # TODO: Compute the backward messages
    # Initialization
    backward_messages[num_time_steps - 1] = rover.Distribution({})
    for zn_prev in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][zn_prev] = 1
        
    # i != N-1
    for i in range(1, num_time_steps):
        backward_messages[num_time_steps - i - 1] = rover.Distribution({})
        
        for zi in all_possible_hidden_states:
            summation = 0
            
            for zi_next in backward_messages[num_time_steps-i]:
                observation = observations[num_time_steps-i]
                
                if observation != None:
                    probPositionGivenNextState = observation_model(zi_next)[observation]
                    
                # missing, uninformative, 1
                else:
                    probPositionGivenNextState = 1
                    
                summation +=  backward_messages[num_time_steps - i][zi_next] * probPositionGivenNextState * transition_model(zi)[zi_next]
            
            # ignore 0
            if summation != 0:
                backward_messages[num_time_steps - i - 1][zi] = summation
        
        # normalize backward messages
        backward_messages[num_time_steps - 1 - i].renormalize() 
    
    # TODO: Compute the marginals 
    for i in range (0, num_time_steps): 
        
        marginals[i] = rover.Distribution({})    
        summation = 0
        
        for zi in all_possible_hidden_states:
            
            # ignore 0
            if forward_messages[i][zi] * backward_messages[i][zi] != 0:
                marginals[i][zi] = forward_messages[i][zi] * backward_messages[i][zi]
                summation += forward_messages[i][zi] * backward_messages[i][zi]
            
        # normalize
        for zi in marginals[i].keys():
            marginals[i][zi] /=  summation
            
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    z_prev = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps

    # initialization
    w[0] = rover.Distribution({})
    x0y0 = observations[0]
    
    for z0 in all_possible_hidden_states:
        if x0y0 != None:
            InitProbPositionGivenState = observation_model(z0)[x0y0]
            
        # missing, uninformative, 1
        else:
            InitProbPositionGivenState = 1
            
        Pz0 = prior_distribution[z0]
        
        if InitProbPositionGivenState != 0 and Pz0 != 0:
            w[0][z0] = np.log(InitProbPositionGivenState) + np.log(Pz0)
    
    # i >= 1
    for i in range(1, num_time_steps):
        w[i] = rover.Distribution({})
        z_prev[i] = dict()
        observation = observations[i]
        
        for zi in all_possible_hidden_states:
            if observation != None:
                probPositionGivenState = observation_model(zi)[observation]
            
            # missing, uninformative, 1
            else:
                probPositionGivenState = 1
                
            maxZPrev = -1000
            for zi_1 in w[i - 1]:
                
                # ignore 0
                if transition_model(zi_1)[zi] != 0:
                    
                    if np.log(transition_model(zi_1)[zi]) + w[i - 1][zi_1] > maxZPrev:
                        maxZPrev = np.log(transition_model(zi_1)[zi]) + w[i - 1][zi_1]
                        # argmax zi-1
                        z_prev[i][zi] = zi_1 

            # ignore 0
            if probPositionGivenState != 0:
                w[i][zi] = np.log(probPositionGivenState) + maxZPrev
            
    # back track to find z*'s
    maxW = -1000
    for zi in w[num_time_steps - 1]:
        
        if w[num_time_steps - 1][zi] > maxW:
            maxW = w[num_time_steps - 1][zi]
            # the last state
            estimated_hidden_states[num_time_steps - 1] = zi
    
    for i in range(1, num_time_steps):
        estimated_hidden_states[num_time_steps - i - 1] = z_prev[num_time_steps - i][estimated_hidden_states[num_time_steps - i]]
    
    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    print('\n')
    # q4 error
    # viterbi
    numCorrect = 0
    for i in range(0, num_time_steps):
        
        if hidden_states[i] == estimated_states[i]:
            numCorrect += 1
            
    print("viterbi error:", 1 - numCorrect/100)
    
    # forward backward
    numCorrect = 0
    for i in range(0, num_time_steps):
        maxProb = -1
        
        for zi in marginals[i]:
            if marginals[i][zi] > maxProb:
                zhat = zi
                maxProb = marginals[i][zi]
        
        # q5
        print(i, ":", zhat)
        
        if hidden_states[i] == zhat:
            numCorrect += 1
        else:
            print("violate", i, ":", zhat)
            
    print("forward backward error:", 1 - numCorrect/100)
    
  
    
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
