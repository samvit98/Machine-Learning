def ThompsonSamplingSimulation(means, T):
    num_arms = len(means)
    total_reward = 0
    total_reward_history = []
    wins = np.zeros(num_arms)
    losses = np.zeros(num_arms)
    pulls = np.zeros(num_arms)

    posteriors = [] # Array to main the posterior beta distribution on each arm
    for i in range(num_arms): # Pull each arm once to start the game
        It = i
        r = bernoulli(means[It]).rvs()
        if r == 1:
            wins[It] += 1
        else:
            losses[It] += 1
        # add the beta posterior to our list of posteriors, we add 1 to make the beta distribution defined
        posteriors.append(beta(wins[It]+1, losses[It]+1))
        total_reward += r
        total_reward_history.append(total_reward)
        pulls[i] += 1

    for t in range(T - num_arms):
        draws = [posterior.rvs() for posterior in posteriors]  # Draw a q from each posterior
        It = np.argmax(draws)  # Find the arm with the highest q
        rt = bernoulli(means[It]).rvs()  # Draw a random reward from arm It
        if rt == 1:
            wins[It] += 1  # Update the number of wins on arm It
        else:
            losses[It] += 1  # Update the number of losses on arm It

        posteriors[It] = beta(wins[It] + 1, losses[It] + 1)  # Update the posterior on arm It
        total_reward += rt  # Update the total reward
        total_reward_history.append(total_reward)  # Update the total reward history
        pulls[It] += 1  # Update the number of pulls on arm It
    return total_reward_history, pulls

# Test your function
means = [.3, .25, .1, .05, .01]
total_reward_history, pulls = ThompsonSamplingSimulation(means, 100)
fig, ax = plt.subplots(1, 2, figsize = (10,5))
ax[0].plot(total_reward_history)
ax[0].set_title('Reward over time of Thompson Sampling')
ax[0].set_xlabel('Round t')
ax[1].bar(range(len(means)), pulls)
ax[1].set_title('Number of Pulls of Each arm')
ax[1].set_xlabel('Arm number')
plt.show()