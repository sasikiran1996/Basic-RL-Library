import numpy as np

class sarsa:
	'''
	Class which defines SARSA algorithm. Follows GLIE assumptions for convergence
	i.e. epsilon <- epsilon/time_step
  		stepSize <- stepSize/sqrt(time_step)

	Arguments:
		epsilon: The value of epsilon in e-greedy or sigma in softmax
		stepSize: The value of stepSize in SARSA algorithm
		lambd: lambda version of SARSA
	'''
	def __init__(self, epsilon, stepSize, lambd):

		# the epsilon in e-greedy or sigma in softmax
		self.epsilon = float(epsilon)

		# the step size for updates
		self.stepSize = float(stepSize)

		# value of lambda 
		self.lambd = lambd

	def run_tabular(self, domain, actionSel='e_greedy', episodeCount=100, initScale=0, smallUpdCount=20, smallDecay=1):
		'''
		Runs the SARSA algorithm on given domain in tabular setting
		Updates the domain's action_value i.e. q_pi(s,a)

		Arguments:
			domain: the input domain to run SARSA on.
			actionSel: {e_greedy or softmax}
		
		Returns:
			rewardArr: array of rewards for each epsiode during learning (used to plot) 
		'''
		# Initialization - optimistic
		actionValueArr = initScale*np.ones((domain.stateNum, domain.actionNum))
		domain.set_action_value_fn(actionValueArr)

		# Iterate over each episode
		rewardArr = np.array([])
		update_step = 0
		currentEpsilon = self.epsilon/(update_step + 1)
		for episodeNum in range(episodeCount):
			time_step = 0
			total_reward = 0
			state = domain.sample_initial_state()
			if(actionSel == 'e_greedy'):
				action = domain.sample_action_e_greedy(currentEpsilon, state)
			else:
				action = domain.sample_action_softmax(currentEpsilon, state)
			# reset the e-traces
			e_trace = np.zeros((domain.stateNum,domain.actionNum)) 
			
			while(1):	
				# sample S'
				new_state = domain.sample_next_state(state, action)
				
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				total_reward = total_reward + pow(domain.gamma, time_step)*reward
				
				# sample A'
				if(actionSel == 'e_greedy'):
					new_action = domain.sample_action_e_greedy(currentEpsilon, new_state)
				else:
					new_action = domain.sample_action_softmax(currentEpsilon, new_state) 
				
				# find the current step size
				currStepSize = self.stepSize/np.sqrt(update_step + 1)
				
				# calculate TD error
				td_error = reward + domain.gamma*domain.get_action_value_fn(new_state,new_action) - domain.get_action_value_fn(state,action) 
				
				# update e-traces
				e_trace = self.lambd*domain.gamma*e_trace
				e_trace[domain.get_state_idx(state), action] = e_trace[domain.get_state_idx(state), action] + 1
				
				# SARSA update  
				domain.action_value = domain.action_value + currStepSize*td_error*e_trace   
				
				# end
				state = new_state
				action = new_action
				time_step = time_step + 1
				currentEpsilon = self.epsilon/(update_step + 1)
				# if (episodeNum > 20):
				# 	update_step = update_step + 1
				# else:
				# 	update_step = min(update_step + 1, 1*(episodeNum + 1))
				if (episodeNum > smallUpdCount):
					update_step = update_step + 1
				else:
					update_step = min(update_step + 1, smallDecay*(episodeNum + 1))
				# check termination of episode
				if domain.check_termination(new_state):
					# print update_step
					# print currStepSize
					break
			# End of episode 
			rewardArr = np.append(rewardArr, total_reward)
		return rewardArr


	def run_fn_approx(self, domain, actionSel='e_greedy', episodeCount=100, initScale=0, noUpdCount=50, beyondDecay=20):	

		# Initialization
		domain.init_q_weights(initScale)
		
		# Iterate over each episode
		update_step = 0
		stepSize_decay_count = 0
		rewardArr = np.array([])
		currentEpsilon = self.epsilon/(update_step + 1)
		for episodeNum in range(episodeCount):
			time_step = 0
			total_reward = 0
			state = domain.sample_initial_state()
			if(actionSel == 'e_greedy'):
				action = domain.sample_action_e_greedy(currentEpsilon, state)
			else:
				action = domain.sample_action_softmax(currentEpsilon, state)
			# Reset the eligibility trace
			e_trace = np.zeros(domain.q_wts.shape)
			while(1):
				# sample S'
				new_state = domain.sample_next_state(state, action)
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				total_reward = total_reward + pow(domain.gamma, time_step)*reward
				# sample A'
				if(actionSel == 'e_greedy'):
					new_action = domain.sample_action_e_greedy(currentEpsilon, new_state)
				else:
					new_action = domain.sample_action_softmax(currentEpsilon, new_state)
				# find the current step size
				currStepSize = self.stepSize/np.sqrt(stepSize_decay_count + 1)

				# The td error
				td_error = reward + domain.gamma*domain.get_action_value_fn(new_state,new_action) - domain.get_action_value_fn(state,action) 
				# update the e-traces
				e_trace = domain.gamma*self.lambd*e_trace + domain.basis_expansion_q(state, action)
				# SARSA update
				domain.set_q_weights(domain.q_wts + currStepSize*td_error*e_trace)  
				
				# end
				state = new_state
				action = new_action
				# update_step = min(update_step + 1, 20*(episodeNum+1))
				update_step = update_step + 1
				# TODO: decide on this
				# if(domain.basis == 'fourier'):
				# 	if (episodeNum > 50):
				# 		stepSize_decay_count = min(stepSize_decay_count + 1, 20*(episodeNum - 50))
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

				time_step = time_step + 1
				if(domain.basis == 'fourier' and actionSel == 'e_greedy'):
					currentEpsilon = self.epsilon/(update_step + 1)
				else:
					currentEpsilon = self.epsilon/np.sqrt(stepSize_decay_count + 1)
				# check termination of episode
				if domain.check_termination(new_state, time_step):
					# print currStepSize
					# print td_error
					break
			# End of episode 
			rewardArr = np.append(rewardArr, total_reward)
		return rewardArr
		