import numpy as np
from joblib import Parallel, delayed

class gridWorld:
	'''
	Class which implements the standard gridWorld domain

	Arguments:
		gamma: gamma value of MDP (default: 0.9)
		randomSeed: the seed for random sampling (default: 10) 
	'''
	def __init__(self, gamma=0.9, randomSeed=10, sigmaSoftmax=1.0):
		
		# Seed the random number generator for repeatability
		np.random.seed(randomSeed)

		# gridWorld - state 21 is water, obstacles are denoted by -1
		grid = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,-1,13,14], [15,16,-1,17,18], [19,20,21,22,23]])
		self.grid = np.pad(grid, [(1, 1), (1, 1)], mode='constant', constant_values=-1)

		# Actions are indices of [AU, AR, AD, AL]
		self.action_array = np.array([[-1,0], [0, 1], [1,0], [0,-1]])

		# Discount parameter
		self.gamma = gamma

		# Tabular Parametric policy
		self.stateNum = 23
		self.actionNum = 4
		policy = np.ones((self.stateNum-1, self.actionNum)) 	# update the policy matrix too
		self.policy = policy.flatten() 							# theta

		self.sigmaSoftmax = sigmaSoftmax

		# create an index for state->position
		self.positionDict = {}
		tmpArr = np.array([-1,-1])
		for ii in range(1,6):
			for jj in range(1,6):
				if (self.grid[ii][jj] != -1):
					tmpArr[0] = ii
					tmpArr[1] = jj
					self.positionDict[self.grid[ii][jj]] = np.copy(tmpArr)

		# state value function initialization: (ndarray: (stateNum, ))
		self.state_value = np.zeros((self.stateNum, ))

		# action value function initialization: (ndarray: (stateNum, actionNum))
		self.action_value = np.zeros((self.stateNum, self.actionNum))

		# policy parameters in tabular for actor critic (ndarray: (stateNum, actionNum))
		self.policy_param = np.zeros((self.stateNum, self.actionNum))

	def sample_initial_state(self):
		'''
		Sample the initial state from d_0(.)
		In the gridWorld, it's always state 1

		Returns:
			initial state = 1 in the gridWorld setting
		'''
		return 1;

	def get_state_idx(self, state):
		'''
		Returns the index of state from 0 to stateNum-1
		
		Arguments:
			state: current state

		Returns:
			stateIdx: index of the state in state array reps
		'''
		return (state-1);

	def check_termination(self, new_state):
		'''
		Checks for termination condition of episode given current transition

		Arguments:
			new_state: the next state S_t+1
		
		Returns:
			boolean -> True if episode should end after this step
		'''
		if(new_state == 23):
			return True

		return False


	def get_state_value_fn(self, state):
		'''
		Function v_pi(s) for the gridworld domain which returns the value fn. of given state

		Arguments:
			state: the state number ranging from 1 to 23 (both inclusive)

		Returns:
			v_pi(state): (int) the value function at given state
		'''
		return self.state_value[state-1]


	def set_state_value_fn(self, stateValueFn):
		'''
		Function that sets the value of v_pi(s) for the gridWorld domain

		Arguments:
			stateValueFn: (ndarray: shape=(stateNum, ))
		
		'''
		self.state_value = stateValueFn

	def set_state_value_specific(self, val, state):
		'''
		Function that sets the value of v_pi(s) for specific state

		Arguments:
			val: the value to set
			state: the state to update
		
		'''
		self.state_value[state-1] = val


	def get_policy_probs(self, state):
		'''
		Function that returns the policy probs for each action given the state i.e. pi(s,.)

		Arguments:
			state: the current state

		Returns:
			policyProbs: (ndarray: (actionNum, ))
				The probs of policy at that state i.e. softmax(polic_params(state)) 
		'''
		return self.softmax(self.policy_param[state-1])


	def set_policy_param_specific(self, val, state, action):
		'''
		Function that sets the specific values of policy parameters
		
		Arguments:
			val: the value to set
			state: req. state to update
			action: req. action to update

		'''
		self.policy_param[state-1][action] = val


	def set_action_value_specific(self, val, state, action):
		'''
		Set specifc action value to val 
		q_pi(state, action) = val

		Arguments:
			val: the value to set
			state: req. state to update
			action: req. action to update

		'''
		self.action_value[state-1][action] = val


	def get_action_value_fn(self, state, action):
		'''
		Function q_pi(s,a) for the gridWorld domain which 
		returns the action value fn. for given state and action

		Arguments:
			state: the state number ranging from 1 to 23 (both inclusive)
			action: the action index ranging from 0 to 3 (both inclusive)
		
		Returns:
			q_pi(state, action): the action value function at given state
		'''
		return self.action_value[state-1, action]


	def set_action_value_fn(self, actionValueFn):
		'''
		Function that sets the value of q_pi(s,a) for the gridWorld domain
		
		Arguments:
			actionValueFn: (ndarray: shape=(stateNum, actionNum))

		'''
		self.action_value = actionValueFn


	def sample_next_state(self, state, action):
		'''
		Function to sample next state based on gridWorld dynamics i.e. P(s,a,.)	
	
		Arguments:
			state: the current state S_t 
			action: the current action A_t

		Returns:
			new_state: the next state S_t+1
		'''
		position = self.get_position(state)
		prob = np.random.uniform()
		if (prob <= 0.8):
			new_position = position + self.action_array[action]
		else:
			if (prob <= 0.85):
				actual_action = (action + 1) % 4
				new_position = position + self.action_array[actual_action]
			else:
				if (prob <= 0.90):
					actual_action = (action - 1) % 4
					new_position = position + self.action_array[actual_action]
				else:
					new_position = position
		
		# handle obstacles and out of grid cases
		if (self.grid[new_position[0], new_position[1]] == -1):
			new_position = position

		return self.get_state_num(new_position)


	def sample_reward(self, state, action, new_state):
		'''
		Function to sample reward of current transition i.e. d_r(s,a,s',.)

		Arguments:
			state: the current state S_t 
			action: the current action A_t
			new_state: the next state S_t+1

		Returns:
			reward: the reward at current time step R_t
		'''
		# position = self.get_position(state)
		# new_position = self.get_position(new_state)
		if (new_state == 21):
			return -10
		if (new_state == 23):
			return 10
		return 0


	def softmax(self, x):
		'''
		Softmax calculating function

		Arguments:
			x: the input array to apply softmax (ndarray: (n, ))

		Returns:
			out: the softmax applied to x i.e. out=softmax(x)
		'''
		expVal = np.exp(x - np.max(x))
		softmaxVal = expVal/np.sum(expVal)
		return softmaxVal

	def sample_action(self, state, policy):
		'''
		Function to sample action from given policy i.e pi(s,.)

		Arguments:
			state: state the current state S_t 
			policy: the input policy vector (ndarray: shape=(22*4,)) state
		Returns:
			action: An integer in [0,4) which is an index into action_array
		
		'''	
		position = self.get_position(state)
		# make policy into matrix
		policyVec = np.reshape(policy, (-1, 4))
		state = self.grid[position[0], position[1]]
		# select the policy for current state
		currentPolicy = self.softmax(policyVec[state-1])
		# sample action
		rndNum = np.random.rand()
		if (rndNum <= currentPolicy[0]):
			return 0
		else:
			if (rndNum <= currentPolicy[1] + currentPolicy[0]):
				return 1
			else:
				if (rndNum <= currentPolicy[2] + currentPolicy[1] + currentPolicy[0]):
					return 2
				else:
					return 3

	
	def sample_action_uniform(self, state):
		'''
		Function to sample action based on uniform policy

		Arguments:
			state: the current state S_t (doesn't matter)

		Returns:
			action: An integer in {0,1,2,3} which is an index into action_array
		'''
		# sample a uniform number
		return np.random.randint(0, 4)


	def sample_action_softmax(self, temperature, state, criterion='action_value_fn'):
		'''
		Function to sample action based on softmax policy w.r.to q-function (or) policy parameters

		Arguments:
			state: the current state S_t
			temperature: the sigma parameter of softmax
			criterion: criterion to decide where to sample actions from {'action_value_fn', 'policy_param'}

		Returns:
			action: An integer in {0,1,2,3} which is an index into action_array
		'''
		# get corresponding q values
		if(criterion == 'action_value_fn'):
			corr_q_values = self.action_value[state-1]
		else:
			corr_q_values = self.policy_param[state-1]

		norm_q_values = corr_q_values/temperature
		exp_q_values = np.exp(norm_q_values - np.max(norm_q_values))
		prob = exp_q_values/np.sum(exp_q_values)

		rndSample = np.random.rand()

		if(rndSample <= prob[0]):
			return 0
		else:
			if(rndSample <= prob[0] + prob[1]):
				return 1
			else:
				if(rndSample <= prob[0] + prob[1] + prob[2]):
					return 2
				else:
					return 3


	def sample_action_e_greedy(self, epsilon, state):
		'''
		Function to sample action based on e-greedy policy w.r.to q-function

		Arguments:
			state: the current state S_t

		Returns:
			action: An integer in {0,1,2,3} which is an index into action_array
		'''
		rndSample = np.random.rand()
		# With epsilon prob. sample uniformly
		if(rndSample < epsilon):
			return np.random.randint(0, 4)
		# Else sample the max action
		else:
			# get the corresponding q-values for that state
			corr_q_values = self.action_value[state-1]
			# find the indices corr. to maximum value
			maxIdx = (corr_q_values == np.max(corr_q_values)).astype("int")
			# count such max
			countMax = np.sum(maxIdx)

			idxRndNum = np.random.randint(0,countMax)
			select = 0
			for ii in range(self.actionNum):
				if(maxIdx[ii] == 1):
					if(select == idxRndNum):
						return ii
					else:
						select = select + 1


	def get_state_num(self, position):
		'''
		Function to return the state number based on position

		Arguments: 
			position: the position as an index into grid

		Returns:
			state: the corresponding state number in gridWorld
		''' 
		return self.grid[position[0], position[1]]


	def get_position(self, state):
		'''
		Function to return the position based on state number

		Arguments: 
			state: the state number in gridWorld

		Returns:
			position: the corresponding position in grid
		'''
		return self.positionDict[state]


	def grad_ln_pi(self, state, action):
		'''
		Returns the gradient of log of pi(s,a) w.r.to policy parameters theta

		Arguments:
			state: current state
			action: current action
		'''
		gradVal = np.zeros(self.policy_param.shape)
		gradVal[state-1] = gradVal[state-1] - self.get_policy_probs(state)
		gradVal[state-1, action] = gradVal[state-1, action] + 1
		return gradVal


	def evaluate_policy(self, policy, episodeCount):
		'''
		Evaluates a given policy for given number of episodes. 

		Arguments:
			policy: input policy to be evaluated as vector (ndarray: (22*4, ))
			episodeCount: number of episodes to average over

		Returns:
			meanReward: the mean of all the rewards from these episodes
		'''
		reward_array = Parallel(n_jobs=4)(delayed(self.evaluate_episode)(policy, 200) for ii in range(episodeCount))
		reward_array = np.array(reward_array)
		return np.mean(reward_array), reward_array


	def evaluate_episode(self, policy, earlyTermination=-1):
		'''
		Evaluate the policy for a single episode. Used to pass for a parallel evaluation

		Arguments: 
			policyParam: input policy parameters (tabular) to be evaluated as vector (ndarray: shape=(22*4, ))
			earlyTermination: the no. of steps after which to terminate simulation

		Returns:
			total_reward: the total return G for that episode run
		'''
		total_reward = 0
		time_step = 0
		# Start state 1 always
		state = self.sample_initial_state()
		while (1):
			# Select action based on policy
			action = self.sample_action(state, policy)
			# print action
			# Select next state
			new_state = self.sample_next_state(state, action)
			# print new_position
			# Select reward
			reward = self.sample_reward(state, action, new_state)
			# Update total reward
			total_reward = total_reward + pow(self.gamma, time_step)*reward
			# End of iteration
			state = new_state
			time_step = time_step + 1
			if self.check_termination(new_state):
				break
			# Prevent an episode from running too long
			if (earlyTermination != -1):
				if (time_step >= earlyTermination):
					break
		# End of episode
		return total_reward
