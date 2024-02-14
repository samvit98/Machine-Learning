def UniformSimulation(means, T):
    num_arms = len(means)
    total_reward = 0
    total_reward_history = []
    wins = np.zeros(num_arms)
    losses = np.zeros(num_arms)
    pulls = np.zeros(num_arms)

    for i in range(num_arms): # Pull each arm once to start the game
        It = i
        r = bernoulli(means[It]).rvs()
        if r == 1:
            wins[It] += 1
        else:
            losses[It] += 1
        total_reward += r
        total_reward_history.append(total_reward)
        pulls[i] += 1

    # Now we enter the main part of the game
    for t in range(T-num_arms):
        It = np.random.choice(num_arms) # TODO: Pick an arm at random
        rt = bernoulli(means[It]).rvs() # TODO: Get a reward from that arm
        if rt==1:
            wins[It] += 1 #TODO: Update the number of wins
        else:
            losses[It] += 1 #TODO: Update the number of losses
        total_reward += rt # Update the total_reward
        total_reward_history.append(total_reward)
        pulls[It] += 1 #Update the number of pulls on arm It
    return total_reward_history, pulls

# Test your function
means = [.3, .25, .1, .05, .01]
total_reward_history, pulls = UniformSimulation(means, 100)
fig, ax = plt.subplots(1, 2, figsize = (10,5))
ax[0].plot(total_reward_history)
ax[0].set_title('Reward over time of Uniform')
ax[0].set_xlabel('Round t')
ax[1].bar(range(len(means)), pulls)
ax[1].set_title('Number of Pulls of Each arm')
ax[1].set_xlabel('Arm number')
plt.show()