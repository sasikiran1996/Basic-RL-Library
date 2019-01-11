import numpy as np
from joblib import Parallel, delayed

class mountainCar:
	'''
	Class which implements the standard mountainCar domain

	Arguments:
		gamma: gamma value of MDP (default: 1.0)
		randomSeed: the seed for random sampling (default: 10) 
		basis: basis expansion being used for q_fn approximation (default: fourier)
		order: order of the basis (default: 3)
		tileShift: the shift of tiles if using tile basis
		tileWidth: the width of each tile if using tile basis
	'''
	def __init__(self, gamma=1.0, randomSeed=10, basis='fourier', order=3, tileShift=0.33, tileWidth=0.5):
		
		# Seed the random number generator for repeatability
		np.random.seed(randomSeed)

		# State array (x, v)
		self.stateArr = np.zeros((2,))

		# Actions are {-1,0,1} -> {reverse, neutral, forward} 
		self.action_array = np.array([-1,0,1])
		self.actionNum = 3

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
			numpy array with [-0.5, 0] i.e. the start of moutainCar
		'''
		return np.array([-0.5, 0])

	def sample_next_state(self, state, action):
		'''
		Function to sample next state based on mountainCar dynamics i.e. P(s,a,.)	
	
		Arguments:
			state: the current state S_t 
			action: the current action A_t

		Returns:
			new_state: the next state S_t+1
		'''

		# Get current state var
		x, v = state
		
		# Compute the next state
		v_new = v + 0.001*action - 0.0025*np.cos(3*x)
		x_new = x + v_new

		# Position exceeds limits
		if(x_new < -1.2):
			x_new = -1.2
			v_new = 0
		if(x_new > 0.5):
			x_new = 0.5
			v_new = 0
			
		# velocity exceeds limits
		if(v_new < -0.07):
			v_new = -0.07
		if(v_new > 0.07):
			v_new = 0.07

		return np.array([x_new, v_new])


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
		return -1.0


	def normalize(self, state, minVal=0.0, maxVal=1.0):
		'''
		Normalize the state of cartPole to lie within specified limits
		x: [-1.2,0.5]
		v: [-0.07,0.07]

		Arguments:
			minVal: the minimum after normalize
			maxVal: the maximum after normalize
		'''
		normPosition = np.zeros((2,))
		normPosition[0] = (state[0] + 1.2)/1.7
		normPosition[1] = (state[1] + 0.07)/0.14
		
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


	def get_action_idx(self, action):
		'''
		Returns the index of the given action

		Arguments:
			action: the current action

		Returns:
			actionIdx: the index of the action
		'''
		return (action+1)


	def basis_expansion_v(self, state):
		'''
		Function that transforms the state into required basis of that order

		Arguments:
			state: the current state to expand 

		Returns:
			fourier expanded state, action representation phi(s)
		'''
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

		return expState


	def basis_expansion_q(self, state, action):
		'''
		Function that transforms the state, action pair into required basis of that order

		Arguments:
			state: the current state to expand 
			action: the current action to expand {-1,0,+1}

		Returns:
			fourier expanded state, action representation phi(s,a)
		'''
		# convert action to 0,1,2
		actionIdx = (action + 1)
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
		# if(actionIdx == 0):
		# 	expPosition = np.append(expState, otherZeros)
		# else:
		# 	expPosition = np.append(otherZeros, expState)
		if(actionIdx == 0):
			expPosition = np.append(expState, np.append(otherZeros, otherZeros))
		else:
			if(actionIdx == 1):
				expPosition = np.append(otherZeros, np.append(expState, otherZeros))
			else:
				expPosition = np.append(otherZeros, np.append(otherZeros, expState))

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


	def sample_action_uniform(self, position):
		'''
		Function to sample action from uniform policy over all actions

		Arguments:
			position: the current state S_t (doesn't matter)

		Returns:
			action: {-1,0,+1} - {reverse, neutral, forward}
		
		'''
		return np.random.randint(0,3) - 1


	def init_policy_wts(self, initScale=0):
		'''
		Initializes the policy parameters shape=(|phi(s)|, |A|)

		Arguments:
			initScale: scale of the weights to be set i.e. 1*scale
		'''
		if(self.basis == 'fourier' or self.basis == 'polynomial'):
			self.policy_wts = initScale*np.ones((self.expanderSet.shape[0], self.actionNum))
		else:
			if(self.basis == 'tile'):
				totalTiles1D = int(1.0/self.tileWidth) 
				self.policy_wts = initScale*np.ones((self.expanderSet.shape[0]*(totalTiles1D**4), self.actionNum))
			else:
				raise NotImplementedError


	def init_v_weights(self, initScale=0):
		'''
		Sets the weights used in function approx. of v-fn.

		Arguments:
			initScale: scale of the weights to be set i.e. 1*scale
		'''
		if(self.basis == 'fourier' or self.basis == 'polynomial'):
			self.v_wts = initScale*np.ones((self.expanderSet.shape[0],))
		else:
			if(self.basis == 'tile'):
				totalTiles1D = int(1.0/self.tileWidth) 
				self.v_wts = initScale*np.ones((self.expanderSet.shape[0]*(totalTiles1D**4), ))
			else:
				raise NotImplementedError


	# Note: for different basis, requires to define expanderSet in __init__: Done
	def init_q_weights(self, initScale=0):
		'''
		Sets the weights used in function approx. of q-fn.

		Arguments: 
			initScale: scale of the weights to be set i.e. 1*scale
		'''
		if(self.basis == 'fourier' or self.basis == 'polynomial'):
			self.q_wts = initScale*np.ones((self.actionNum*self.expanderSet.shape[0], ))
		else:
			if(self.basis == 'tile'):
				totalTiles1D = int(1.0/self.tileWidth) 
				self.q_wts = initScale*np.ones((self.actionNum*self.expanderSet.shape[0]*(totalTiles1D**4), ))
			else:
				raise NotImplementedError


	def get_action_value_fn(self, state, action):
		'''
		Function q_pi(s,a) for the cartPole domain which 
		returns the action value fn. for given state and action

		Arguments:
			state: the state (x,v,theta,theta_dot)
			action: the action - {-1, 0, +1}
		
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

	# NOTE: Done only for fourier basis
	# Changed for all basis
	def get_state_value_fn(self, state):
		'''
		Function to compute the value fn. at given state using given basis
		Uses v_wts: the weights for value fn. must be set earlier using set

		Arguments:
			state: the required state

		Returns:
			v_pi(s) = w.T*phi(s)
		'''
		expPosition = self.basis_expansion_v(state)
		return np.sum(self.v_wts*expPosition)


	def set_v_weights(self, wts):
		'''
		Function to set the weights of state value function for fn. approx.

		Arguments:
			wts: the value of weights to set 
		'''
		self.v_wts = wts


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


	def get_policy_probs(self, state):
		'''
		Returns the policy probs. at the given state pi(s,.)

		Arguments:
			state: the current state

		Returns:
			probs: the array of probs pi(s,.) for all actions
		'''
		expState = self.basis_expansion_v(state)
		corr_unnorm_val = np.array([0,0,0])
		for ii in range(3):
			corr_unnorm_val[ii] = np.sum(expState*self.policy_wts[:, ii])
		return self.softmax(corr_unnorm_val)


	def sample_action_softmax(self, temperature, state, criterion='action_value_fn'):
		'''
		Function to sample action based on softmax policy w.r.to q-function

		Arguments:
			temperature: the softmax parameter 
			state: the current state S_t
			criterion: either of {'action_value_fn' or 'policy_param'}

		Returns:
			action: An integer in {-1,0,+1} 
		'''
		if(criterion == 'action_value_fn'):
			corr_q_values = np.array([self.get_action_value_fn(state, -1), self.get_action_value_fn(state, 0), self.get_action_value_fn(state, 1)])
		else:
			expState = self.basis_expansion_v(state)
			corr_q_values = np.array([0,0,0])
			for ii in range(3):
				corr_q_values[ii] = np.sum(expState*self.policy_wts[:, ii])
		# print temperature
		norm_q_values = corr_q_values/temperature
		exp_q_values = np.exp(norm_q_values - np.max(norm_q_values))
		prob = exp_q_values/np.sum(exp_q_values)

		rndSample = np.random.rand()
		if(rndSample <= prob[0]):
			return -1
		else:
			if(rndSample <= prob[0] + prob[1]):
				return 0
			else:
				return 1


	def sample_action_e_greedy(self, epsilon, state):
		'''
		Function to sample action based on e-greedy policy w.r.to q-function

		Arguments:
			state: the current state S_t

		Returns:
			action: An integer in {-1,0,+1} 
		'''
		rndSample = np.random.rand()
		# With epsilon prob. sample uniformly
		if(rndSample < epsilon):
			return np.random.randint(0,3) - 1
		# Else sample the max action
		else:
			# get the corresponding q-values for that state
			corr_q_values = np.array([self.get_action_value_fn(state, -1), self.get_action_value_fn(state, 0), self.get_action_value_fn(state, 1)])
			# find the indices corr. to maximum value
			maxIdx = (corr_q_values == np.max(corr_q_values)).astype("int")
			# count such max
			countMax = np.sum(maxIdx)
			
			idxRndNum = np.random.randint(0,countMax)
			select = 0
			for ii in range(self.actionNum):
				if(maxIdx[ii] == 1):
					if(select == idxRndNum):
						return ii-1
					else:
						select = select + 1


	def grad_ln_pi(self, state, action):
		'''
		Returns the gradient of log of pi(s,a) w.r.to policy parameters theta

		Arguments:
			state: current state
			action: current action
		'''
		gradVal = np.zeros(self.policy_wts.shape)
		# compute pi(s,.)
		policy_probs = self.get_policy_probs(state)
		# compute phi(s)
		expState = self.basis_expansion_v(state)
		# compute gradient
		for ii in range(self.actionNum):
			gradVal[:, ii] += -policy_probs[ii]*expState
		gradVal[:, self.get_action_idx(action)] += expState
		return gradVal  
				

	def check_termination(self, new_state, time_step):
		'''
		Checks for termination condition of episode given current transition

		Arguments:
			new_state: the next state S_t+1
			time_step: the current time_step

		Returns:
			boolean -> True if episode should end after this step
		'''
		# Check distance bounds
		if (new_state[0] >= 0.5):
			return True
		# Check time limit
		if (time_step >= 6000):
			return True
		return False