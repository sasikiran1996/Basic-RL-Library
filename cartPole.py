import numpy as np
from joblib import Parallel, delayed

class cartPole:
	'''
	Class which implements the standard cartPole domain

	Arguments:
		gamma: gamma value of MDP (default: 1.0)
		randomSeed: the seed for random sampling (default: 10) 
		timeDelta: the intervals for each action (default: 0.02)
		maxTime: horizon time for cartPole (default: 20.2)
		basis: basis expansion being used for q_fn approximation (default: fourier)
		order: order of the basis (default: 3)
	'''
	def __init__(self, gamma=1.0, randomSeed=10, timeDelta=0.02, maxTime=20.2, basis='fourier', order=3, tileShift=0.33, tileWidth=0.5):
		
		# Seed the random number generator for repeatability
		np.random.seed(randomSeed)

		# State array (x, v, theta, theta_dot)
		self.stateArr = np.zeros((4,))

		# Actions are -1 - left, +1 - right -> multiplier of force in dynamics eqn. 
		self.action_array = np.array([-1,1])
		self.actionNum = 2

		# Time granule
		self.delta_T = timeDelta

		# horizon of cart pole
		self.maxTime = maxTime

		# Discount parameter
		self.gamma = gamma

		# fourier expanded combos if using fourier basis
		self.basis = basis
		if(basis == 'fourier'):
			self.basisOrder = order
			self.expanderSet = self.find_expansion(-1*np.ones((self.stateArr.shape)))
			# Initialize weights for q_fn - use the fn. 
			# self.q_wts = np.zeros(self.expanderSet.shape[0]) 
	
		if(basis == 'polynomial'):
			self.basisOrder = order
			self.expanderSet = self.find_expansion(-1*np.ones((self.stateArr.shape)))

		if(basis == 'tile'):
			self.basisOrder = order
			# expander set consists of shifts to do
			self.expanderSet = self.find_expansion(-1*np.ones((self.stateArr.shape))) - order/2
			self.tileShift = tileShift	# set shifts to be multiples of width
			self.tileWidth = tileWidth

	def sample_initial_state(self):
		'''
		Function to return the initial state

		Returns:
			numpy array with [0,0,0,0] i.e. the start of cart pole
		'''
		return np.array([0,0,0,0])

	def sample_next_state(self, state, action):
		'''
		Function to sample next state based on cartPole dynamics i.e. P(s,a,.)	
	
		Arguments:
			state: the current state S_t 
			action: the current action A_t

		Returns:
			new_state: the next state S_t+1
		'''

		# Get current state var
		x, v, theta, theta_dot = state
		
		# Calculate the double derivatives - angular acc
		theta_dd = (9.8*np.sin(theta) + np.cos(theta)*(-10*action - 0.05*(theta_dot**2)*np.sin(theta))/(1.1))/(0.5*(4.0/3.0 - 0.1*(np.cos(theta)**2)/(1.1)))
		# acceleration
		x_dd = (10*action + 0.05*((theta_dot**2)*np.sin(theta) - theta_dd*np.cos(theta)))/(1.1)

		# Compute the next state
		x_new = x + 0.02*(v)
		theta_new = theta + 0.02*(theta_dot)
		v_new = v + 0.02*(x_dd)
		theta_dot_new = theta_dot + 0.02*(theta_dd)

		# Velocity exceeds maximum
		if (v_new <= -10):
			v_new = -10
		if (v_new >= 10):
			v_new = 10
			
		# Angular velocity exceeds maximum
		if (theta_dot_new <= -np.pi):
			theta_dot_new = -np.pi
		if (theta_dot_new >= np.pi):
			theta_dot_new = np.pi

		return np.array([x_new, v_new, theta_new, theta_dot_new])


	def sample_reward(self, state, action, new_state):
		'''
		Function to sample reward of current transition i.e. d_r(s,a,s',.)
		which is always one in cartPole domain

		Arguments:
			state: the current state S_t 
			action: the current action A_t
			new_state: the next state S_t+1

		Returns:
			reward: the reward at current time step R_t
		'''
		return 1.0


	def append_one_basis(self, state):
		'''
		Function that just appends one to the given state

		Arguments:
			state: the current state to expand

		Returns:
			[state, 1.0]
		'''
		return np.append(state, 1.0);


	def normalize(self, state, minVal=0.0, maxVal=1.0):
		'''
		Normalize the state of cartPole to lie within specified limits
		x: [-3,3]
		v: [-10,10]
		theta: [-pi/2,pi/2]
		theta_dot: [-pi,pi]

		Arguments:
			minVal: the minimum after normalize
			maxVal: the maximum after normalize
		'''
		normPosition = np.zeros((4,))
		normPosition[0] = (state[0] + 3.0)/6.0
		normPosition[1] = (state[1] + 10.0)/20.0
		normPosition[2] = (state[2] + np.pi/2)/(np.pi)
		normPosition[3] = (state[3] + np.pi)/(2*np.pi)
		
		return minVal + normPosition*(maxVal - minVal)


	def polynomial_basis(self, state):
		'''
		Function that transforms the state into polynomial basis of n th order

		Arguments:
			state: the current state to expand 

		Returns:
			fourier expanded state representation phi(s)
		'''
		#first normalize the state
		normPosition = self.normalize(state)
		expanderSet = self.expanderSet
		return np.product(normPosition**expanderSet, axis=1)

	def fourier_basis(self, state):
		'''
		Function that transforms the state into fourier basis of n th order

		Arguments:
			state: the current state to expand 

		Returns:
			fourier expanded state representation phi(s)
		'''
		# first normalize the state
		normPosition = self.normalize(state)
		expanderSet = self.expanderSet
		return np.cos(np.pi*np.sum(expanderSet*normPosition, axis=1))

	def tile_basis(self, state):
		'''
		Function that transforms the state into tile basis of nth order

		Arguments:
			state: the current state to expand

		Returns:
			tile basis expanded state representation
		'''
		# first normalize the state
		normPosition = self.normalize(state)
		# shiftSet = self.expanderSet*int(self.tileShift/self.tileWidth)
		
		# # find the grid no. in original tiling
		# totalTileNum = int(1.0/self.tileWidth)
		# originalPlace = np.clip((normPosition/self.tileWidth).astype('int'), 0, totalTileNum-1)
		
		# # find in all shifted tilings
		# tileNums = originalPlace - shiftSet
		
		# alternate
		totalTileNum = int(1.0/self.tileWidth)
		totalShiftNum = self.expanderSet.shape[0]
		shiftSet = self.expanderSet*self.tileShift
		tileNums = np.floor((normPosition - shiftSet)/self.tileWidth)
		tileNums = tileNums.astype('int')
		tileCoods = np.hstack((tileNums, np.array([np.arange(totalShiftNum)]).T))
		#----------

		# find those within limits
		limitBools = ((tileNums >= 0).astype('int'))*((tileNums <= totalTileNum-1).astype('int'))
		selectIdx = np.product(limitBools, axis=1).astype('bool')
		reqCoods = tileCoods[selectIdx]

		phi_s = np.zeros((totalTileNum, totalTileNum, totalTileNum, totalTileNum, totalShiftNum))
		phi_s[reqCoods[:,0], reqCoods[:,1], reqCoods[:,2], reqCoods[:,3], reqCoods[:,4]] = 1

		return phi_s.flatten()


	def basis_expansion_q(self, state, action):
		'''
		Function that transforms the state, action pair into fourier basis of n th order

		Arguments:
			state: the current state to expand 
			action: the current action to expand {-1,+1}

		Returns:
			fourier expanded state, action representation phi(s,a)
		'''
		# convert action to 0,1
		actionIdx = (action + 1)/2
		actionIdx = int(actionIdx)
		# find fourier/polynomial exp phi(s)
		if(self.basis == 'fourier'):
			expState = self.fourier_basis(state)
		else:
			if(self.basis == 'polynomial'):
				expState = self.polynomial_basis(state)
			else:
				if(self.basis == 'tile'):
					expState = self.tile_basis(state)
				else:
					raise NotImplementedError

		otherZeros = np.zeros(expState.shape)
		# append zeros at other action
		if(actionIdx == 0):
			expPosition = np.append(expState, otherZeros)
		else:
			expPosition = np.append(otherZeros, expState)

		return expPosition


	def find_expansion(self, inArray):
		'''
		Recursive function to generate all terms of fourier series

		Arguments:
			inArray: array with first -1 being the position to recursively expand

		Returns:
			array expanded after first occurrence of -1 in inArray.
		'''
		# Find the first occurrence of -1 in inArray
		foundIdx = -1
		for ii in range(inArray.size):
			if(inArray[ii] == -1):
				foundIdx = ii
				break
		# If exists expand
		returnArr = np.array([])
		if(foundIdx != -1):
			for jj in range(self.basisOrder+1):
				inArray[foundIdx] = jj
				if(returnArr.size == 0):
					returnArr = self.find_expansion(inArray)
				else:
					returnArr = np.concatenate((returnArr, self.find_expansion(inArray)))
				inArray[foundIdx] = -1
		else:
			returnArr = np.array([inArray])
		return returnArr


	def sample_action_simple(self, position, inPolicy):
		'''
		Function to sample action from given policy i.e pi(s,.). Uses append one basis

		Arguments:
			position: the current state S_t 
			inPolicy: the input policy vector

		Returns:
			action: -1 or +1 which are left or right 		
		
		'''
		# Define action prob as a logistic function of the one appended state
		basisExpState = self.append_one_basis(position)
		intm = np.dot(inPolicy, basisExpState)
		prob_right = 1.0/(1.0 + np.exp(-1*intm))
		rndNum = np.random.rand()
		if (rndNum <= prob_right):
			return 1.0
		else: 
			return -1.0


	def sample_action_uniform(self, position):
		'''
		Function to sample action from uniform policy over all actions

		Arguments:
			position: the current state S_t (doesn't matter)

		Returns:
			action: -1 or +1 which are left or right
		
		'''
		return np.random.randint(0,2)*2 - 1


	# Note: for different basis, requires to define expanderSet in __init__
	def init_q_weights(self, initScale=0):
		'''
		Sets the weights used in function approx. of q-fn.

		Arguments: 
			initScale: scale of the weights to be set i.e. 1*scale
		'''
		if(self.basis == 'fourier' or self.basis == 'polynomial'):
			self.q_wts = initScale*np.ones((self.actionNum*self.expanderSet.shape[0],))
		else:
			if(self.basis == 'tile'):
				totalTiles1D = int(1.0/self.tileWidth) 
				self.q_wts = initScale*np.ones((self.actionNum*self.expanderSet.shape[0]*(totalTiles1D**4)), )
			else:
				raise NotImplementedError

	def get_action_value_fn(self, state, action):
		'''
		Function q_pi(s,a) for the cartPole domain which 
		returns the action value fn. for given state and action

		Arguments:
			state: the state (x,v,theta,theta_dot)
			action: the action - {-1, +1}
		
		Returns:
			q_pi(state, action): the action value function at given state
		'''
		expPosition = self.basis_expansion_q(state, action)
		return np.sum(self.q_wts*expPosition)


	def set_q_weights(self, wts):
		'''
		Function to set the weights of action value function for fn. approx.

		Arguments:
			wts: the value of weights to set 
		'''
		self.q_wts = wts


	def get_state_value_fn(self, state):
		'''
		Function to compute the value fn. at given state using fourier basis
		Uses v_wts: the weights for value fn. must be set earlier using set

		Arguments:
			state: the required state

		Returns:
			v_pi(s) = w.T*phi(s)
		'''
		expPosition = self.fourier_basis(state)
		return np.sum(self.v_wts*expPosition)


	def set_v_weights(self, wts):
		'''
		Function to set the weights of state value function for fn. approx.

		Arguments:
			wts: the value of weights to set 
		'''
		self.v_wts = wts


	def evaluate_policy_simple(self, policyParam, episodeCount):
		'''
		Evaluates a given policy for given number of episodes. Uses a threaded implementation.
		This uses append one basis

		Arguments:
			policyParam: input policy parameters (weights) to be evaluated as vector
			episodeCount: number of episodes to average over

		Returns:
			meanReward: the mean of all the rewards from these episodes
			rewardArray: the array containing all the rewards
		'''
		reward_array = Parallel(n_jobs=4)(delayed(self.evaluate_episode_simple)(policyParam) for ii in range(episodeCount))
		reward_array = np.array(reward_array)
		return np.mean(reward_array), reward_array


	def evaluate_episode_simple(self, policyParam):
		'''
		Evaluate the policy for a single episode. Used to pass for parallel evaluation. 
		This uses append one basis

		Arguments: 
			policyParam: input policy parameters (weights) to be evaluated as vector

		Returns:
			total_reward: the total return G for that episode run
		'''
		# Initialize total reward for each episode
		total_reward = 0
		time_step = 0
		# Start state 1 always
		position = np.array([0,0,0,0])
		while (1):
			# Select action based on policy
			action = self.sample_action_simple(position, policyParam)
			# print action
			# Select next state
			new_position = self.sample_next_state(position, action)
			# print new_position
			# Select reward
			reward = self.sample_reward(position, action, new_position)
			# Update total reward
			total_reward = total_reward + pow(self.gamma, time_step)*reward
			# End of iteration
			position = new_position
			time_step = time_step + 1
			# Check time limit
			if (time_step*(self.delta_T) == self.maxTime):
				break
			# Check distance bounds
			if ((new_position[0] <= -3) or (new_position[0] >= 3)):
				break
			# Pole falls down
			if ((new_position[2] <= -np.pi/2) or (new_position[2] >= np.pi/2)):
				break
			# Velocity exceeds maximum
			if (position[1] <= -10):
				position[1] = -10
			if (position[1] >= 10):
				position[1] = 10
			# Angular velocity exceeds maximum
			if (position[3] <= -np.pi):
				position[3] = -np.pi
			if (position[3] >= np.pi):
				position[3] = np.pi
		# End of episode
		return total_reward


	def sample_action_softmax(self, temperature, state):
		'''
		Function to sample action based on softmax policy w.r.to q-function

		Arguments:
			state: the current state S_t

		Returns:
			action: An integer in {-1,+1} 
		'''
		corr_q_values = np.array([self.get_action_value_fn(state, -1), self.get_action_value_fn(state, 1)])
		# print temperature
		norm_q_values = corr_q_values/temperature
		exp_q_values = np.exp(norm_q_values - np.max(norm_q_values))
		prob = exp_q_values/np.sum(exp_q_values)

		rndSample = np.random.rand()
		if(rndSample <= prob[0]):
			return -1
		else:
			return 1

	def sample_action_e_greedy(self, epsilon, state):
		'''
		Function to sample action based on e-greedy policy w.r.to q-function

		Arguments:
			state: the current state S_t

		Returns:
			action: An integer in {-1,+1} 
		'''
		rndSample = np.random.rand()
		# With epsilon prob. sample uniformly
		if(rndSample < epsilon):
			return np.random.randint(0,2)*2 - 1
		# Else sample the max action
		else:
			# get the corresponding q-values for that state
			corr_q_values = np.array([self.get_action_value_fn(state, -1), self.get_action_value_fn(state, 1)])
			# find the indices corr. to maximum value
			maxIdx = (corr_q_values == np.max(corr_q_values)).astype("int")
			# count such max
			countMax = np.sum(maxIdx)
			
			idxRndNum = np.random.randint(0,countMax)
			select = 0
			for ii in range(self.actionNum):
				if(maxIdx[ii] == 1):
					if(select == idxRndNum):
						return 2*ii-1
					else:
						select = select + 1


	def check_termination(self, new_state, time_step):
		'''
		Checks for termination condition of episode given current transition

		Arguments:
			new_state: the next state S_t+1
		
		Returns:
			boolean -> True if episode should end after this step
		'''
		# Check time limit
		if (time_step*(self.delta_T) == self.maxTime):
			return True
		# Check distance bounds
		if ((new_state[0] <= -3) or (new_state[0] >= 3)):
			return True
		# Pole falls down
		if ((new_state[2] <= -np.pi/2) or (new_state[2] >= np.pi/2)):
			return True
		return False
