import random
import numpy as np
import matplotlib.pyplot as plt

# number of bandit arms
k = 10
runs = 10
time_steps = 1000

total_ave_reward = np.zeros(time_steps+1)
# initialize q-table with Q_1 = 0 for all actions
Q = np.zeros(k)
# initialize an array to store how many times each action has been called:
N = np.zeros(k)

# initialize rewards for each action - the mean reward for each action is drawn
# from gaussian centered at 0 with variance of 1
reward_means = np.zeros(k)
for i in range(0,len(reward_means)):
    reward_means[i] = random.gauss(0, 1)

print(reward_means)

def policy(epsilon):
    # we define an epsilon-greedy policy
    if random.uniform(0,1) < epsilon:
        # select a random action
        return random.randint(0, k-1)
    else:
        # select greedy action
        return np.argmax(Q)

def get_reward(action):
    # reward is drawn from gaussian centered at reward_means[action] w/ variance of 1
    return random.gauss(reward_means[action], 1)

for run in range(0,runs):
    run_ave_reward = np.zeros(time_steps+1)
    N = np.zeros(k)
    for time_step in range(1, time_steps+1):
        # select an action, a, according to our policy
        action = policy(0.01)
        # iterate number of times action is taken
        N[action] = N[action] + 1
        reward_obtained = get_reward(action)
        # increment total reward for this run
        run_ave_reward[time_step] = (reward_obtained + ((run_ave_reward[time_step - 1])*(time_step - 1)) )/time_step
        # updated Q(a) using resultant reward via our action value estimation method (using sample average here)
        Q[action] = Q[action] + (1/N[action])*(reward_obtained - Q[action])
    #print(N)
    #print(run_ave_reward[0:5])
    total_ave_reward = total_ave_reward + run_ave_reward

# ensemble average reward over total number of runs
print(Q)
total_ave_reward = total_ave_reward/runs
time = range(0,time_steps)
plt.plot(time, total_ave_reward[0:1000])
plt.xlabel('time step')
plt.ylabel('Average reward obtained')
plt.title('Average reward per timestep averaged over 10 bandits')
plt.show()