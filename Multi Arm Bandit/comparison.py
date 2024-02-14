import numpy as np
from scipy.stats import norm, beta, bernoulli
import matplotlib.pyplot as plt
import seaborn as sns

def compare(T):
  means = [.3, .25, .1]
  T = T # if your code breaks, or you want to debug, turn this value down to 100 and it will run faster
  history_TS, pulls_TS = ThompsonSamplingSimulation(means, T)
  history_UN, pulls_UN = UniformSimulation(means, T)
  history_EG, pulls_EG = EpsilonGreedySimulation(means, T)

  fig, ax = plt.subplots(1, 2, figsize = (10,5))
  ax[0].plot(history_TS, label='Thompson Sampling')
  ax[0].plot(history_UN, label='Uniform Sampling')
  ax[0].plot(history_EG, label='Epsilon Greedy')
  ax[0].set_title('Total Reward over time')
  ax[0].set_xlabel('Round t')
  ax[0].legend()
  x = np.arange(1,len(means)+1)
  ax[1].bar(x, pulls_TS, width=0.2, label='Thompson Sampling')
  ax[1].bar(x+.2, pulls_UN, width=0.2, label='Uniform')
  ax[1].bar(x+.4,pulls_EG, width=0.2, label='Epsilon Greedy')
  ax[1].set_title('Number of Pulls of Each arm at the end')
  ax[1].set_xlabel('Arm number')
  ax[1].set_xticks(x+.2)
  ax[1].set_xticklabels( ['Arm {}'.format(i) for i in x] )
  ax[1].legend()
  plt.show()

T_vals = [10,100,400,1000,1500]
for val in T_vals:
  print("Value of T: ",val)
  compare(val)
