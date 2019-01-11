import numpy as np

class q_learning:
	'''
	Class which defines Q-learning algorithm. Follows assumptions for convergence
	i.e. stepSize <- stepSize/sqrt(time_step)

	Arguments:
		epsilon: The value of epsilon in e-greedy
		stepSize: The value of stepSize in SARSA algorithm

	'''
	def __init__(self, epsilon, stepSize, lambd):

		# the epsilon in e-greedy
		self.epsilon = epsilon

		# the step size for updates
		self.stepSize = stepSize

		self.lambd = lambd

	def run_tabular(self, domain, actionSel='e_greedy', episodeCount=100, initScale=0, smallUpdCount=20, smallDecay=1):
		'''
		Runs the Q-learning algorithm on given domain in tabular setting
		Updates the domain's action_value i.e. q_pi(s,a)

		Arguments:
			domain: the input domain to run SARSA on.
		
		Returns:
			rewardArr: array of rewards for each epsiode during learning (used to plot) 
		'''
		# Initialization - optimistic
		actionValueArr = initScale*np.ones((domain.stateNum, domain.actionNum))
		domain.set_action_value_fn(actionValueArr)

		# Iterate over each episode
		update_step = 0
		rewardArr = np.array([])
		for episodeNum in range(episodeCount):
			time_step = 0
			total_reward = 0
			state = domain.sample_initial_state()
			# reset the e-traces
			e_trace = np.zeros((domain.stateNum,domain.actionNum))
			while(1):
				# Sample A
				if(actionSel == 'e_greedy'):
					action = domain.sample_action_e_greedy(self.epsilon, state)
				else:
					action = domain.sample_action_softmax(self.epsilon, state) 
				# Sample S'
				new_state = domain.sample_next_state(state, action)
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				total_reward = total_reward + pow(domain.gamma, time_step)*reward
				# Find (max A') q(S',A')
				maxQVal = -np.inf
				for kk in range(domain.actionNum):
					currQVal = domain.get_action_value_fn(new_state, kk)
					if (currQVal > maxQVal):
						maxQVal = currQVal
				# find the current step size
				currStepSize = self.stepSize/np.sqrt(update_step + 1)
				# The td-error
				q_delta = reward + domain.gamma*maxQVal - domain.get_action_value_fn(state, action)
				# update e-traces
				e_trace = self.lambd*domain.gamma*e_trace
				e_trace[domain.get_state_idx(state), action] = e_trace[domain.get_state_idx(state), action] + 1
				# Q-learning update
				domain.action_value = domain.action_value + currStepSize*q_delta*e_trace 
				# end
				state = new_state
				time_step = time_step + 1
				if (episodeNum > smallUpdCount):
					update_step = update_step + 1
				else:
					update_step = min(update_step + 1, smallDecay*(episodeNum + 1))
				# check termination of episode
				if domain.check_termination(new_state):
					break
			# End of episode 
			rewardArr = np.append(rewardArr, total_reward)
		return rewardArr


	def run_fn_approx(self, domain, actionSel='e_greedy', episodeCount=100, initScale=0, noUpdCount=50, beyondDecay=10):	

		# Initialization
		domain.init_q_weights(initScale)
		
		# Iterate over each episode
		update_step = 0
		stepSize_decay_count = 0
		rewardArr = np.array([])
		for episodeNum in range(episodeCount):
			time_step = 0
			total_reward = 0
			state = domain.sample_initial_state()
			# Reset the eligibility trace
			e_trace = np.zeros(domain.q_wts.shape)
			while(1):
				# Sample A
				if(actionSel == 'e_greedy'):
					action = domain.sample_action_e_greedy(self.epsilon, state)
				else:
					action = domain.sample_action_softmax(self.epsilon, state)
				# Sample S'
				new_state = domain.sample_next_state(state, action)
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				total_reward = total_reward + pow(domain.gamma, time_step)*reward
				# Find (max A') q(S',A')
				maxQVal = -np.inf
				for kk in range(domain.actionNum):
					currQVal = domain.get_action_value_fn(new_state, domain.action_array[kk])
					if (currQVal > maxQVal):
						maxQVal = currQVal
				# find the current step size
				currStepSize = self.stepSize/np.sqrt(stepSize_decay_count + 1)
				
				# The td-error term
				q_delta = reward + domain.gamma*maxQVal - domain.get_action_value_fn(state, action)
				# update the e-trace
				e_trace = domain.gamma*self.lambd*e_trace + domain.basis_expansion_q(state, action)
				# Q-learning update
				domain.set_q_weights(domain.q_wts + currStepSize*q_delta*e_trace)
				
				# end
				state = new_state
				time_step = time_step + 1
				update_step = update_step + 1
				# if(domain.basis == 'fourier'):
				# 	if (episodeNum > 50):
				# 		stepSize_decay_count = min(stepSize_decay_count + 1, 10*(episodeNum - 50))
				# 	else:
				# 		stepSize_decay_count = 0
				# else:
				# 	if(domain.basis == 'polynomial'):
				# 		if (episodeNum > 50):
				# 			stepSize_decay_count = min(stepSize_decay_count + 1, 1*(episodeNum - 50))
				# 		else:
				# 			stepSize_decay_count = 0
				if (episodeNum > noUpdCount):
					stepSize_decay_count = min(stepSize_decay_count + 1, beyondDecay*(episodeNum - noUpdCount))
				else:
					stepSize_decay_count = 0	
				# check termination of episode
				if domain.check_termination(new_state, time_step):
					# print currStepSize
					# print q_delta
					break
			# End of episode 
			rewardArr = np.append(rewardArr, total_reward)
		return rewardArr

