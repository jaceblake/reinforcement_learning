
# coding: utf-8

# In[ ]:


import numpy as np
import numpy.testing as npt
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats
import logging
import time


# In[ ]:


import gym
env = gym.make('FrozenLake-v0' )


# 
# 
#     S: Start
#     F: Frozen (safe)
#     H: Hole
#     G: Goal
# 
# Environment:
# 
#     SFFF
#     FHFH
#     FFFH
#     HFFG
# 
# State indices:
# 
#     0123
#     4567
#     89..
#     ....
# 
# Action indices:
# 
#     0: Left
#     1: Down
#     2: Right
#     3: Up
# 
# 

# In[ ]:


state  = env.reset()
state


# In[ ]:


def act(action):
    state, reward, done, info = env.step(action)
    print (state,reward)
    env.render() 


# In[ ]:


act(0)


# 
# 
# ### Given is the following (non optimal) policy π
# 

# In[ ]:


pi = {0:1, 1:2, 2:1, 3:0, 4:1, 6:1, 8:2, 9:0, 10:1, 13:2, 14:2}


# In[ ]:




state  = env.reset()
print (state)
done = False
while not done:
    env.render()
    action = pi[state]
    print ("action:", action)
    state, reward, done, info = env.step(action)
    print (state, reward, done, info)
    if done:
        print ("return:", reward) # return for all visited states is here last reward

env.render(close=True)



# In[ ]:


def run_episode(policy,env=env):
    steps = []
    state  = env.reset()
    #print (state)
    done = False
    while not done:
        #env.render()
        action = policy[state]
        old_state_action = [state,action]
        #print ("random action:", action)
        state, reward, done, info = env.step(action)
        steps.append(old_state_action +[state, int(reward)])
    #env.render() 
    return steps


# In[ ]:


run_episode(pi)


# 
# 
# ### What is the average performance of the policy, i.e. 
# ### the percentage that the agent reach the goal state starting from the beginning.
# 

# In[ ]:


# number of states
state_space_size = env.observation_space.n

gamma = 0.99
epsilon=0.1


# In[ ]:


# Average percentage of policy
def compute_average_percentage(policy,nb_of_episodes):
    N = np.zeros(state_space_size, dtype=int)
    S = np.zeros(state_space_size)
    Gs = np.zeros(state_space_size)
    
    for e in range(nb_of_episodes):
        observations_and_reward_list = run_episode(policy)
        G = 0.
        for old_action, action, new_state, reward in reversed(observations_and_reward_list): 
            G = reward + gamma * G
            N[new_state] += 1
            S[new_state] += G 
            
    Gs[N!=0] = S[N!=0]/N[N!=0]
    return Gs

compute_average_percentage(pi,100000)


# In[ ]:


# deprecated function
# return v_s for every visit monte carlo 
'''
def every_visit_monte_carlo(target_state=0, nb_episodes=1000):
    
    N = np.zeros(state_space_size, dtype=int)
    S = np.zeros(state_space_size)
    V = np.zeros(state_space_size)
                 
    for e in range(nb_episodes):
        observations_and_reward_list = run_episode(pi)
        G = 0.
        for old_action, action, new_state, reward in reversed(observations_and_reward_list): 
            G = reward + gamma * G
            N[new_state] += 1
            S[new_state] += G 
            V[new_state] = S[new_state]/N[new_state]
                 
    return V
    
'''


# In[ ]:


'''

# deprecated function call
v_s = every_visit_monte_carlo()
print(v_s)

'''


# In[ ]:


Q = np.zeros([state_space_size,4]) # because we have 4 Actions
Q.shape


# In[ ]:


from collections import defaultdict
returns = defaultdict(list)


# In[ ]:


returns


# In[ ]:


pi


# In[ ]:




def compute_q_values_every_visit(nb_of_episodes, policy , run_episode):
    
    for i in range(nb_of_episodes):
        observations_and_reward_list = run_episode(policy)
    
        G = 0. 
        action_return = dict() 
        for old_action, action, new_state, reward in reversed(observations_and_reward_list): 
            G = reward + gamma * G
            action_return[old_action] = action, G 
     
        for old_state, (action, G) in action_return.items():
            if old_state is not None:
                returns[(old_state, action)].append(G) 
                re_mean = np.array(returns[(old_state, action)]).mean() 
                Q[(old_state,) + (action,)] = re_mean
            
                policy[old_state] = np.argmax(Q[(old_state)])
                
        if(i%50000 == 0.0):
            print("ok, im doing my work",i)
        
    return policy




# In[ ]:


#nb_of_episodes = 50000
#policy = compute_q_values_every_visit(nb_of_episodes, pi , run_episode)


# In[ ]:


#policy


# In[ ]:


#Q


# In[ ]:


def run_episode_exploring_start(policy, env=env):
    steps = []
    state  = env.reset()
    #print (state)
    done = False
    start = True
    while not done:
        if start:
            action = np.random.binomial(3, p=0.25)
            start = False
        else:
            action = pi[state]
        old_state_action = [state,action]
        #print ("random action:", action)
        state, reward, done, info = env.step(action)
        steps.append(old_state_action +[state, int(reward)])
    #env.render() 
    return steps


# In[ ]:


def run_episode_epsilon_greedy(policy, epsilon=0.1, env=env):
    steps = []
    state  = env.reset()
    #print (state)
    done = False
    while not done:
        if np.random.rand() > epsilon:
            action = policy[state]
        else:
            action = np.random.randint(0,4)
        old_state_action = [state,action]
        #print ("random action:", action)
        state, reward, done, info = env.step(action)
        steps.append(old_state_action +[state, int(reward)])
    #env.render() 
    return steps


# In[ ]:


nb_of_episodes = 500000
t0 = time.time()
policy = compute_q_values_every_visit(nb_of_episodes, pi , run_episode_epsilon_greedy)
print("\nCompute policy took time",time.time() - t0)


# In[ ]:


print("\nOptimal ppolicy ----- ",policy)


# In[ ]:


print("\nQ values ----\n\n",Q)


# In[ ]:


#policy = lambda a: pi[a]


# In[ ]:


#e.g.:
#v_s = first_visit_monte_carlo_prediction(target_state=0, nb_episodes=10000)
#print (v_s)


# 
# Exercise: Monte Carlo Prediction
# 
#     Computes the q-values of the vistited state-action pairs. qπ(s,a)
# 
# by Monte Carlo Evaluation (first visit or every visit) with γ=0.99
# 
# . Implement an appropriate python function.
# 
# Compute from qπ(s,a)
# the values of vπ(s)
# 
#     for all states that have been visited.
# 
# Exercise: Monte Carlo Control
# 
#     Modify the policy π
# 
# to a ϵ-greedy policy and use policy improvement to find an optimal ϵ-greedy policy (in combination with the Monte-Carlo policy evaluation).

# 
# Exercise
# 
# Compute the state-values vπ(s)
# of your optimal ϵ-greedy policy π from qπ(s,a)
# 
# :
# 
#     Give an mathematical expression for the computation.
#         Compute the numerical values for vπ(s)
# 
# from q(s,a) (with numpy).
