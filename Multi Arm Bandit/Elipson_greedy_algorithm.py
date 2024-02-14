import numpy as np
from scipy.stats import norm, beta, bernoulli
import matplotlib.pyplot as plt
import seaborn as sns

def EpsilonGreedySimulation(means, T, epsilon=.95):
    num_arms = len(means)
    total_reward = 0 # Our total reward at each time
    total_reward_history = [] # Array that maintains a history of rewards at each time
    wins = np.zeros(num_arms) #Array to main the number of wins of each arm
    losses = np.zeros(num_arms) # Array to maintain the number of losses of each arm
    pulls = np.zeros(num_arms) # Array to maintain the number of pulls of each arm


    emp_means = [] # Array to main the empirical means (ie the running average) of the proportion of wins of each arm

    # Pull each arm once
    for i in range(num_arms):
        It = i
        # Get a reward from the i-th arm
        r = bernoulli(means[It]).rvs()
        if r==1:
            # If the reward is 1, update the number of wins
            wins[i] += 1
        else:
            # If the reward is 0, update the number of losses
            losses[i] += 1
        # Update the number of pulls
        pulls[i] += 1
        emp_means.append(r/pulls[i])
        total_reward += r
        total_reward_history.append(total_reward)


    for t in range(T-num_arms):
        if np.random.rand() < epsilon:
            #95% of the time we play the highest empirical mean
            It = np.argmax(emp_means)
        else:
            #5% of the time we play a random arm
            It = np.random.choice(num_arms)
        rt = bernoulli(means[It]).rvs() # Get a reward from this arm when we play it.
        total_reward += rt # update the total amount of reward we have at each time.
        total_reward_history.append(total_reward)
        if rt == 1:
            wins[It] += 1
        else:
            losses[It] += 1
        pulls[It] += 1 # Update the number of pulls of the arm we pulled
        emp_means[It] = wins[It]/pulls[It]
    return total_reward_history, pulls
# Test your function
means = [.3, .25, .1, .05, .01]
total_reward_history, pulls = EpsilonGreedySimulation(means, 1000)

# Plot the total_reward_history and num pulls. To plot these next to each other,
# I am using plt.subplots.
fig, ax = plt.subplots(1, 2, figsize = (10,5))
ax[0].plot(total_reward_history)
ax[0].set_title('Reward over time of Epsilon Greedy')
ax[0].set_xlabel('Round t')
ax[1].bar(range(len(means)), pulls)
ax[1].set_title('Number of Pulls of Each arm')
ax[1].set_xlabel('Arm number')
plt.show()