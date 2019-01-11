import numpy as np

class actorCritic:

	'''
	Class which implements actor-critic algorithm

	Arguments:
		alpha: the step size
		lambd: the lambda parameter
	'''
	def __init__(self, alpha, lambd):

		self.stepSize = alpha

		self.lambd = lambd

	def run_tabular(self, domain, episodeCount=100, initScale_w=0, initScale_theta=0, smallUpdCount=20, smallDecay=1):
		'''
		Runs the tabular version of Actor-critic algorithm 21

		Arguments:
			domain: the input domain to run

		Returns:
			rewardArr: the array of rewards after running episodes
		'''
		# Initialize theta
		domain.policy_param = initScale_theta*np.ones((domain.stateNum, domain.actionNum))
		# Initialize w
		domain.set_state_value_fn(initScale_w*np.ones((domain.stateNum, )))

		# Iterate over episodes
		rewardArr = np.array([])
		update_step = 0
		
		for episodeNum in range(episodeCount):

			time_step = 0
			total_reward = 0
			# Sample initial state
			state = domain.sample_initial_state()
			# Reset the e-traces
			e_trace_w = np.zeros((domain.stateNum, ))
			e_trace_theta = np.zeros((domain.stateNum, domain.actionNum))

			while(1):
				# no decay softmax
				action = domain.sample_action_softmax(1.0, state, criterion='policy_param')
				# Sample S'
				new_state = domain.sample_next_state(state, action)
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				total_reward = total_reward + pow(domain.gamma, time_step)*reward
				# find the current step size
				currStepSize = self.stepSize/np.sqrt(update_step + 1)
				# Tune these -> actor/critic step size converge to zero?
				ratioParam = np.sqrt(update_step + 1)
				criticStepSize = currStepSize 
				actorStepSize = criticStepSize
				# td-error
				td_error = reward + domain.gamma*domain.get_state_value_fn(new_state) - domain.get_state_value_fn(state)
				# print td_error
				# Update Critic
				# Update e-traces for w
				e_trace_w = domain.gamma*self.lambd*e_trace_w
				e_trace_w[domain.get_state_idx(state)] = e_trace_w[domain.get_state_idx(state)] + 1
				# v_pi(s) update
				domain.state_value = domain.state_value + criticStepSize*td_error*e_trace_w
				# Update Actor
				# Update e-traces for theta
				# e_trace_theta = domain.gamma*self.lambd*e_trace_theta
				# e_trace_theta[domain.get_state_idx(state)] = e_trace_theta[domain.get_state_idx(state)] - domain.get_policy_probs(state)
				# e_trace_theta[domain.get_state_idx(state), action] = e_trace_theta[domain.get_state_idx(state), action] + 1
				e_trace_theta = domain.gamma*self.lambd*e_trace_theta + domain.grad_ln_pi(state, action)
				# theta update
				domain.policy_param = domain.policy_param + actorStepSize*td_error*e_trace_theta
				# End of step
				state = new_state
				time_step = time_step + 1
				if (episodeNum > smallUpdCount):
					update_step = update_step + 1
				else:
					update_step = min(update_step + 1, smallDecay*(episodeNum + 1))
				# check termination of episode
				if domain.check_termination(new_state):
					break
				# termination based on time step for gridWorld
				if (time_step >= 200):
					break
			# End of episode 
			rewardArr = np.append(rewardArr, total_reward)
		return rewardArr

	def run_fn_approx(self, domain, episodeCount=100, initScale_w=0, initScale_theta=0, noUpdCount=50, beyondDecay=10):
		'''
		Functional approximation version for actor-critic

		Arguments:
			domain: the input domain
			episodeCount: the number of episodes to run
			initScale_w: the initalization scale of w
			initScale_theta: the initialization scale of theta 

		Returns:
			rewardArr: array containing rewards for all the episodes
		'''
		# Initialize theta
		domain.init_policy_wts(initScale=initScale_theta)
		# Initialize w
		domain.init_v_weights(initScale=initScale_w)

		# Iterate over episodes
		rewardArr = np.array([])
		update_step = 0
		
		for episodeNum in range(episodeCount):

			time_step = 0
			total_reward = 0
			# Sample initial state
			state = domain.sample_initial_state()
			# Reset the e-traces
			e_trace_w = np.zeros(domain.v_wts.shape)
			e_trace_theta = np.zeros(domain.policy_wts.shape)

			# start the episode
			while(1):
				# no decay softmax
				action = domain.sample_action_softmax(1.0, state, criterion='policy_param')
				# Sample S'
				new_state = domain.sample_next_state(state, action)
				# sample R
				reward = domain.sample_reward(state, action, new_state)
				total_reward = total_reward + pow(domain.gamma, time_step)*reward
				# find the current step size
				currStepSize = self.stepSize/np.sqrt(update_step + 1)
				# Tune these -> actor/critic step size converge to zero?
				ratioParam = np.sqrt(update_step + 1)
				criticStepSize = currStepSize 
				actorStepSize = criticStepSize
				# td-error
				td_error = reward + domain.gamma*domain.get_state_value_fn(new_state) - domain.get_state_value_fn(state)
				# Update Critic
				# update e_trace_w
				e_trace_w = domain.gamma*self.lambd*e_trace_w + domain.basis_expansion_v(state)
				# update w
				domain.set_v_weights(domain.v_wts + criticStepSize*td_error*e_trace_w)
				# Update Actor
				# update e_trace_theta
				# policy_probs = domain.get_policy_probs(state)
				# expState = domain.basis_expansion_v(state)
				# e_trace_theta = domain.gamma*self.lambd*e_trace_theta
				# for ii in range(domain.actionNum):
				# 	e_trace_theta[:, ii] = e_trace_theta[:, ii] - policy_probs[ii]*expState
				# e_trace_theta[:, domain.get_action_idx(action)] = e_trace_theta[:, domain.get_action_idx(action)] + expState  
				e_trace_theta = domain.gamma*self.lambd*e_trace_theta + domain.grad_ln_pi(state, action)
				# update theta
				domain.policy_wts = domain.policy_wts + actorStepSize*td_error*e_trace_theta
				# End of step
				state = new_state
				time_step = time_step + 1
				# if (episodeNum > smallUpdCount):
				# 	update_step = update_step + 1
				# else:
				# 	update_step = min(update_step + 1, smallDecay*(episodeNum + 1))
				if (episodeNum > noUpdCount):
					update_step = min(update_step + 1, beyondDecay*(episodeNum - noUpdCount))
				else:
					update_step = 0
				# check termination of episode
				if domain.check_termination(new_state, time_step):
					break
				# termination based on time step for mountain car
				if (time_step >= 6000):
					break
			# End of episode 
			rewardArr = np.append(rewardArr, total_reward)
		return rewardArr

