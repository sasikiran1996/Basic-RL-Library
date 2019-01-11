import gridWorld as gw
import mountainCar as mc
import actor_critic
import sarsa
import q_learning
import reinforce
import matplotlib.pyplot as plt
import numpy as np

# lambd - [0, 0.7] beyond 0.7 large variance

gw_sarsa_returns = np.array([])
for ii in range(100):
	print("Step " + str(ii) + " in progress...")
	d = gw.gridWorld(randomSeed = ii)
	q = sarsa.sarsa(0.2, 0.75, 0.0)
	a = q.run_tabular(d, initScale=4)
	gw_sarsa_returns = np.append(gw_sarsa_returns, a)


dataArr = gw_sarsa_returns
plotValues = np.zeros((100,100))
for ii in range(100):
	plotValues[ii] = dataArr[100*ii:100*(ii+1)]

meanValues1 = np.mean(plotValues, axis=0)
stdValues1 = np.std(plotValues, axis=0)
print("The mean is " + str(np.mean(meanValues1[50:])))
plt.errorbar(np.arange(1,101), meanValues1, yerr=stdValues1, color='red', ecolor='green', alpha=0.5)
# plt.plot(np.arange(1,101), 3.5*np.ones((100, )), color='blue', alpha=1.0)
plt.xlabel("Number of episodes")
plt.ylabel("Expected Returns")
plt.show()

#plt.plot(np.arange(100), a)
#plt.show()

# lambd - [0, 0.6] beyond 0.6 large variance
gw_q_returns = np.array([])
for ii in range(100):
	print("Step " + str(ii) + " in progress...")
	d = gw.gridWorld(randomSeed = ii)
	q = q_learning.q_learning(0.1, 0.5, 0.3)
	a = q.run_tabular(d, initScale=4)
	gw_q_returns = np.append(gw_q_returns, a)


dataArr = gw_q_returns
plotValues = np.zeros((100,100))
for ii in range(100):
	plotValues[ii] = dataArr[100*ii:100*(ii+1)]

meanValues1 = np.mean(plotValues, axis=0)
stdValues1 = np.std(plotValues, axis=0)
print("The mean is " + str(np.mean(meanValues1[50:])))
plt.errorbar(np.arange(1,101), meanValues1, yerr=stdValues1, color='red', ecolor='green', alpha=0.5)
# plt.plot(np.arange(1,101), 3.5*np.ones((100, )), color='blue', alpha=1.0)
plt.xlabel("Number of episodes")
plt.ylabel("Expected Returns")
plt.show()

# plt.plot(np.arange(100), a)
# plt.show()


# lr - 0.0075, eps - 0.02, noUpdCount=10, beyondDecay=1, 200ep - around -150 (gamma=0) -> check initScale
# lambd=0.2 & shown param, max=155, mean_50=~165
mc_sarsa_returns = np.array([])
for ii in range(100):
	print("Step " + str(ii) + " in progress...")
	d = mc.mountainCar(order = 5, randomSeed = ii)
	q = sarsa.sarsa(0.02, 0.0075, 0.0)
	a = q.run_fn_approx(d, episodeCount=100, noUpdCount=10, beyondDecay=1)
	mc_sarsa_returns = np.append(mc_sarsa_returns, a)

dataArr = mc_sarsa_returns
plotValues = np.zeros((100,100))
for ii in range(100):
	plotValues[ii] = dataArr[100*ii:100*(ii+1)]

meanValues1 = np.mean(plotValues, axis=0)
stdValues1 = np.std(plotValues, axis=0)
print("The mean is " + str(np.mean(meanValues1[50:])))
print("The max is " + str(np.max(meanValues1)))
plt.errorbar(np.arange(1,101), meanValues1, yerr=stdValues1, color='red', ecolor='green', alpha=0.5)
plt.xlabel("Number of episodes")
plt.ylabel("Expected Return")
plt.legend(["Sarsa on mountainCar"])
plt.axis([0, 100, -1000, 0])
# plt.plot(np.arange(1,101), -150*np.ones((100, )), color='red')
plt.show()

# print("The max attained is " + str(np.max(a)))
# plt.plot(np.arange(100), a)
# plt.legend(["Sarsa-lambda on mountainCar"])
# plt.show()


# mean=-165, max=-159
mc_q_returns = np.array([])
for ii in range(100):
	print("Step " + str(ii) + " in progress...")
	d = mc.mountainCar(order = 5, randomSeed = ii)
	q = q_learning.q_learning(0.01, 0.005, 0.0)
	a = q.run_fn_approx(d, episodeCount=100, noUpdCount=10, beyondDecay=1)
	mc_q_returns = np.append(mc_q_returns, a)

dataArr = mc_q_returns
plotValues = np.zeros((100,100))
for ii in range(100):
	plotValues[ii] = dataArr[100*ii:100*(ii+1)]

meanValues1 = np.mean(plotValues, axis=0)
stdValues1 = np.std(plotValues, axis=0)
print("The mean is " + str(np.mean(meanValues1[50:])))
print("The max is " + str(np.max(meanValues1)))
plt.errorbar(np.arange(1,101), meanValues1, yerr=stdValues1, color='red', ecolor='green', alpha=0.5)
plt.xlabel("Number of episodes")
plt.ylabel("Expected Return")
plt.legend(["Q-learning on mountainCar"])
plt.axis([0, 100, -1000, 0])
# plt.plot(np.arange(1,101), -150*np.ones((100, )), color='red')
plt.show()


# optimal for actor-critic gridworld
# mean=3.43, max=3.67
grid_ac_returns = np.array([])
for ii in range(100):
	print('Step ' + str(ii) + ' in progress...')
	d = gw.gridWorld(randomSeed = ii)
	q = actor_critic.actorCritic(alpha=0.8, lambd=0.35)
	a = q.run_tabular(d, episodeCount=100, smallUpdCount=100, smallDecay=1, initScale_w=4.0)
	grid_ac_returns = np.append(grid_ac_returns, a)

dataArr = grid_ac_returns
plotValues = np.zeros((100,100))
for ii in range(100):
	plotValues[ii] = dataArr[100*ii:100*(ii+1)]

meanValues1 = np.mean(plotValues, axis=0)
stdValues1 = np.std(plotValues, axis=0)
print("The mean is " + str(np.mean(meanValues1[50:])))
print("The max is " + str(np.max(meanValues1)))
plt.errorbar(np.arange(1,101), meanValues1, yerr=stdValues1, color='red', ecolor='green', alpha=0.5)
plt.xlabel("Number of episodes")
plt.ylabel("Expected Return")
# plt.plot(np.arange(1,101), 3.5*np.ones((100, )), color='blue', alpha=1.0)
plt.show()


# Note the last 50 mean = 151, max = 146 
mc_ac_returns = np.array([])
for ii in range(100):
	print('Step ' + str(ii) + ' in progress...')
	d = mc.mountainCar(order = 5, randomSeed = ii)
	q = actor_critic.actorCritic(alpha=0.002, lambd=0.6)
	a = q.run_fn_approx(d, episodeCount=100, initScale_w=1.0, noUpdCount=50, beyondDecay=10)
	mc_ac_returns = np.append(mc_ac_returns, a)

dataArr = mc_ac_returns
plotValues = np.zeros((100,100))
for ii in range(100):
	plotValues[ii] = dataArr[100*ii:100*(ii+1)]

meanValues1 = np.mean(plotValues, axis=0)
stdValues1 = np.std(plotValues, axis=0)
print("The mean is " + str(np.mean(meanValues1[50:])))
print("The max is " + str(np.max(meanValues1)))
plt.errorbar(np.arange(1,101), meanValues1, yerr=stdValues1, color='red', ecolor='green', alpha=0.5)
# plt.plot(np.arange(1,101), -150*np.ones((100, )), color='blue', alpha=1.0)
plt.xlabel("Number of episodes")
plt.ylabel("Expected Return")
plt.show()


gw_reinforce_returns = np.array([])
for ii in range(100):
	print('Step ' + str(ii) + ' in progress...')
	d = gw.gridWorld(randomSeed=ii)
	q = reinforce.reinforce(alpha=0.045)
	a = q.run_tabular(d, episodeCount=500, noUpdCount=250, beyondDecay=1)
	gw_reinforce_returns = np.append(gw_reinforce_returns, a)

dataArr = gw_reinforce_returns
plotValues = np.zeros((100,500))
for ii in range(100):
	plotValues[ii] = dataArr[500*ii:500*(ii+1)]

meanValues1 = np.mean(plotValues, axis=0)
stdValues1 = np.std(plotValues, axis=0)
print("The mean is " + str(np.mean(meanValues1[50:])))
print("The max is " + str(np.max(meanValues1)))
plt.errorbar(np.arange(1,501), meanValues1, yerr=stdValues1, color='red', ecolor='green', alpha=0.5)
plt.xlabel("Number of episodes")
plt.ylabel("Expected Return")
# plt.plot(np.arange(1,101), -150*np.ones((500, )), color='blue', alpha=1.0)
plt.show()


# d = mc.mountainCar(order = 5, randomSeed = 10)
# q = reinforce.reinforce(alpha=0.00001)
# a = q.run_fn_approx(d, episodeCount=100, noUpdCount=100, beyondDecay=1)
# plt.plot(np.arange(100), a)
# plt.show()

# plt.plot(np.arange(1,101), gridSarsaMean, alpha=0.5)
# plt.plot(np.arange(1,101), gridQMean, alpha=0.5)
# plt.plot(np.arange(1,101), gridACMean, alpha=0.5)
# plt.xlabel("Number of episodes")
# plt.ylabel("Expected Returns")
# plt.legend(["Sarsa-lambda on gridWorld", "Q-lambda on gridWorld", "Actor-Critic on gridWorld"])