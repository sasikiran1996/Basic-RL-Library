import numpy as np

class reinforce:

	def __init__(self, alpha):

		self.stepSize = alpha

	def calculate_G_t(self, rewardArr, gamma):
		'''
		Calculates G_t which is (unbiased) estimate of q_pi(a,a)

		Arguments:
			rewardArr: array of rewards from that time step
			gamma: the value of gamma for domain

		Returns:
			G_t: the computed G_t
		'''
		total_reward = 0
		for ii in range(rewardArr.shape[0]):
			total_reward += pow(gamma, ii)*rewardArr[ii]
		return total_reward


	def run_tabular(self, domain, episodeCount=100, initScale_theta=0, noUpdCount=50, beyondDecay=1):

		# Initialize theta 
		domain.policy_param = initScale_theta*np.ones((domain.stateNum, domain.actionNum))

		# keep track of total rewards
		totalRewardArr = np.array([])

		for episodeNum in range(episodeCount):

			# Keep track of history
			stateArr = np.array([])
			actionArr = np.array([])
			rewardArr = np.array([])

			# Generate the history for the episode
			time_step = 0
			# Sample initial state
			state = domain.sample_initial_state()
			stateArr = np.append(stateArr, state)
			# Iterate till termination
			while(1):
				# Sample A - no decay softmax
				action = domain.sample_action_softmax(1.0, state, criterion='policy_param')
				actionArr = np.append(actionArr, action)
				# Sample S'
				new_state = domain.sample_next_state(state, action)
				stateArr = np.append(stateArr, new_state)
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				rewardArr = np.append(rewardArr, reward)
				# End of step
				state = new_state
				time_step = time_step + 1
				# check termination of episode
				if domain.check_termination(new_state):
					break
				# termination based on time step for gridWorld
				if (time_step >= 200):
					break
			# End of episode
			# Add total reward
			totalRewardArr = np.append(totalRewardArr, self.calculate_G_t(rewardArr, domain.gamma))
			# compute gradient for the episode
			grad_J_theta = np.zeros(domain.policy_param.shape)
			for ii in range(actionArr.shape[0]):
				grad_J_theta += self.calculate_G_t(rewardArr[ii:], domain.gamma)*domain.grad_ln_pi(int(stateArr[ii]), int(actionArr[ii]))
			# find current step size -> NOTE: change step size routine
			if (episodeNum < noUpdCount):
				update_step = 0
			else:
				update_step = update_step + beyondDecay
			currentStepSize = self.stepSize/np.sqrt(update_step+1)
			# update policy param
			domain.policy_param += currentStepSize*grad_J_theta
		# return the total rewards
		return totalRewardArr

	def run_fn_approx(self, domain, episodeCount=100, initScale_theta=0, noUpdCount=50, beyondDecay=1):

		# Initialize theta 
		domain.init_policy_wts(initScale=initScale_theta)

		# keep track of total rewards
		totalRewardArr = np.array([])

		for episodeNum in range(episodeCount):

			# Keep track of history
			stateArr = np.array([])
			actionArr = np.array([])
			rewardArr = np.array([])

			# Generate the history for the episode
			time_step = 0
			# Sample initial state
			state = domain.sample_initial_state()
			stateArr = np.append(stateArr, state)
			# Iterate till termination
			while(1):
				# Sample A - no decay softmax
				action = domain.sample_action_softmax(1.0, state, criterion='policy_param')
				actionArr = np.append(actionArr, action)
				# Sample S'
				new_state = domain.sample_next_state(state, action)
				stateArr = np.append(stateArr, new_state)
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				rewardArr = np.append(rewardArr, reward)
				# End of step
				state = new_state
				time_step = time_step + 1
				# check termination of episode
				if domain.check_termination(new_state, time_step):
					break
				# termination based on time step for gridWorld
				if (time_step >= 2000):
					break
			# End of episode
			# print(time_step)
			stateArr = stateArr.reshape((-1,2))
			# Add total reward
			totalRewardArr = np.append(totalRewardArr, self.calculate_G_t(rewardArr, domain.gamma))
			# compute gradient for the episode
			grad_J_theta = np.zeros(domain.policy_wts.shape)
			for ii in range(actionArr.shape[0]):
				# print stateArr[ii]
				grad_J_theta += self.calculate_G_t(rewardArr[ii:], domain.gamma)*domain.grad_ln_pi(stateArr[ii], int(actionArr[ii]))
			# print(np.unique(grad_J_theta))
			# find current step size -> NOTE: change step size routine
			if (episodeNum < noUpdCount):
				update_step = 0
			else:
				update_step = update_step + beyondDecay
			currentStepSize = self.stepSize/np.sqrt(update_step+1)
			# update policy param
			domain.policy_wts += currentStepSize*grad_J_theta
		# return the total rewards
		return totalRewardArr

