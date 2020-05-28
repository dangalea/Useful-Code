import numpy as np
import random
import inspect
import copy
import pylab as pl
from IPython import display
import time
import matplotlib.pyplot as plt

class ga(object):

	def __init__(self, pop_size, max_iters, term_threshold, keep_next_iter, remove_next_iter, mutation_rate):
		'''
		Initilization method
		arg: pop_size - population size for genetic algorithm
		arg: num_iters - maximum number of iterations for the genetic algorithm to perform
		arg: term_threshold - termination threshold
		arg: keep_next_iter - percentage of best population members to keep. Must be less than or equal to 1
		arg: remove_next_iter - percentage of worst population members to replace. Must be less than or equal to 1
		arg: mutation_rate - percentage probability of each parameter of a candidate to mutate. Must be less than or equal to 1
		'''

		#Error handling for keep_next_iter
		if keep_next_iter >= 1:
			raise ValueError("Cannot keep more than the whole of the population for the next iteration. Make sure that keep_next_iter <= 1.")
		if keep_next_iter < 0:
			raise ValueError("Must keep at least none of the population for the next iteration. Make sure that keep_next_iter >= 0.")

		#Error handling for remove_next_iter
		if remove_next_iter >= 1:
			raise ValueError("Cannot remove more than the whole of the population for the next iteration. Make sure that remove_next_iter <= 1.")
		if remove_next_iter < 0:
			raise ValueError("Cannot remove more than none of the population for the next iteration. Make sure that remove_next_iter >=0 .")

		#Error handling
		if keep_next_iter + remove_next_iter >= 1:
			raise ValueError("Some percentage of the population must remain for crossover. Make sure that keep_next_iter+remove_next_iter < 1.")

		#Erro handling for mutation rate
		if 0 > mutation_rate > 1:
			raise ValueError("Mutation rate must be in the range [0, 1]")

		#Store variables for instance
		self.pop_size = pop_size
		self.max_iters = max_iters
		self.term_threshold = term_threshold
		self.keep_next_iter = keep_next_iter
		self.remove_next_iter = remove_next_iter
		self.mutation_rate = mutation_rate 

		#Initialise variables for fitting method
		self.num_pars = -1 
		self.eqn = None
		self.x = None
		self.y = None
		self.eqn = None
		self.constraints = None
		self.history = None

	def __create_initial_population(self):
		'''
		Method that creates the initial population for the genetic algorithm
		return: pop: array with population with the cost of each member in the first column
		'''

		#Initialize population array
		pop = np.zeros((self.pop_size, self.num_pars + 1))

		#Create each new member
		for member in range(self.pop_size):
			pop[member] = self.__create_single_member()
		return pop

	def __create_single_member(self):
		'''
		Method to create single member by getting a random value between the bounds specified for each parameter. At least one parameter must be non-zero.
		return: member: 1D array that describes the member. The cost of the candiadate will be the first value, while the rest will be the parameters as tested. 
		'''

		#Create array to hold member
		member = np.zeros(self.num_pars + 1)

		#For each parameter
		for par in range(self.num_pars):

			#Get the lower and upper bounds for the parameter being created
			lower_limit = self.constraints[par, 0]
			upper_limit = self.constraints[par, 1]

			#Generate random parameter that is inside the bounds specified
			val = random.random() * (upper_limit - lower_limit) + lower_limit

			#Store parameter
			member[par+1] = val

		#Check that not all parameters are 0
		if all(0 == param for param in member[1:]):

			#Get random parameter
			par = random.random() * len(member[1:])

			#Get the lower and upper bounds for the parameter being created
			lower_limit = self.constraints[par, 0]
			upper_limit = self.constraints[par, 1]

			#Get non-zero value
			val = 0
			while val == 0:
				val = random.random() * (upper_limit - lower_limit) + lower_limit	

			#Store parameter
			member[par+1] = val
	
		#Calculate cost of member
		member[0] = self.__calculate_cost(member)
		
		#return member
		return member

	def __sort_pop(self, pop):
		'''
		Method to sort a population of candidates
		arg: pop: array holding unsorted population of candidates
		arg: pop: array holding sorted population of candidates in order of increasing cost
		'''

		#Get indices of sotred pop
		i = np.argsort(pop[:, 0])

		#Sort pop according to indices
		pop = pop[i, :]

		#Return sorted population 
		return pop

	def __create_next_iter_pop(self, pop):
		'''
		Method that creates population for next iteration
		arg: pop: population of current iteration
		return: new_pop: population for next iteration
		'''
		
		#Array to hold next iteration's population
		new_pop = np.zeros_like(pop)

		#get number of member of population to keep in the next iteration
		num_members_to_keep = int(np.round(len(pop) * self.keep_next_iter))

		#keep the top percentage of old pop in the new pop
		new_pop[:num_members_to_keep] = pop[:num_members_to_keep]

		#get how many new candiates to produce
		num_new_members = int(np.round(len(pop) * self.remove_next_iter))
		
		#get how many children to produce
		num_members_to_get_children = int(len(pop) - (num_members_to_keep + num_new_members))

		#for each child to create
		for member in range(num_members_to_get_children):

			#get parents' numbers
			parent1, parent2 = np.random.random_sample((2,))
			parent1 = int(parent1 * (num_members_to_keep + num_members_to_get_children))
			parent2 = int(parent2 * (num_members_to_keep + num_members_to_get_children))
			while parent2 == parent1:
				_, parent2 = np.random.random_sample((2,))
				parent2 = int(parent2 * (num_members_to_keep + num_members_to_get_children))	
			
			#get cross point
			cross_point = int(np.round(random.random() * self.num_pars))
			while cross_point == 0 or (self.num_pars != 1 and cross_point == self.num_pars):
				cross_point = int(np.round(random.random() * self.num_pars))
			
			#produce child
			if random.random() < 0.5:
				new_pop[num_members_to_keep+member, 1:cross_point+1] = pop[parent1, 1:cross_point+1]
				new_pop[num_members_to_keep+member, cross_point+1:] = pop[parent2, cross_point+1:]
			else:
				new_pop[num_members_to_keep+member, 1:cross_point+1] = pop[parent2, 1:cross_point+1]
				new_pop[num_members_to_keep+member, cross_point+1:] = pop[parent1, cross_point+1:]

			#for each parameter
			for param in range(self.num_pars):

				#if to be mutated
				if random.random() < self.mutation_rate:

					#Get the lower and upper bounds for the parameter being created
					lower_limit = self.constraints[param, 0]
					upper_limit = self.constraints[param, 1]

					#Get non-zero value
					param_val = 0
					while param_val == 0:
						param_val = random.random() * (upper_limit - lower_limit) + lower_limit	

					#Store parameter
					new_pop[num_members_to_keep+member, param+1] = param_val

			#calculate cost for child
			new_pop[num_members_to_keep+member, 0] = self.__calculate_cost(new_pop[num_members_to_keep+member])			
		
		#Create new members
		for member in range(num_new_members):
			new_pop[num_members_to_keep + num_members_to_get_children + member] = 0
			new_pop[num_members_to_keep + num_members_to_get_children + member] = self.__create_single_member()
		
		#Return new pop
		return new_pop
	
	def __calculate_cost(self, member):
		'''
		Method to calculate the cost between the member given and the target
		arg: member: array having the parameters to test out
		return: cost: cost between target and candidate
		'''

		#Get profile from parameters of candidate
		target_calc = self.__calc_profile(member[1:])

		#Calculate cost
		cost = 0
		for i in range(len(target_calc)):
			cost += (target_calc[i] - self.target[i])**2
		cost = np.sqrt(cost/len(target_calc))
		
		#Return cost
		return cost

	def __save(self, save_path, top):
		'''
		Method to save the candidate provided
		args: save_path: path to save plot and numpy file. Must include name but not extension
		args: top: candidate to save
		return: None
		'''
		
		#Calculate profile from candidate given
		calc_target = self.__calc_profile(top[1:])

		#plot and save figure
		plt.figure()
		plt.plot(range(len(self.target)), self.target, label="desired")
		plt.plot(range(len(calc_target)), calc_target, label="fitted")
		plt.legend()
		plt.title(str(inspect.getsource(self.eqn)) + ": " + str(top))
		plt.savefig(save_path + ".png")

		#save profie
		np.save(save_path + ".npy", calc_target)

	def __calc_profile(self, values):
		'''
		Method to get the profile from the parameters provided
		arg: values: parameter choices
		return: profile: array holding profile generated
		'''

		#Parameters for candiate
		args = list(values)
		
		#Append variables
		for var in self.vars:
			args.append(var)

		#Format inputs
		args = tuple(args)

		#Generate profile
		profile = self.eqn(*args)

		#Return profile
		return profile

	def __plot_timestep(self, history, best_cand, iter_num):
		'''
		Method to plot the progress up to the current timestep
		arg: history: 1D array holding the cost values for each iteration
		arg: best_cand: candidate to be plotted
		arg: iter_num: iteration number
		return: None
		'''

		#Clear figure
		pl.clf()
				
		#plot history
		pl.subplot(121)
		pl.yscale('log')				
		pl.plot(history[:iter_num])

		#plot target
		pl.subplot(122)
		pl.plot(range(len(self.target)), self.target, color='r', label='desired')
		
		#calculate candidate profile
		calc_target = self.__calc_profile(best_cand[1:])

		#plot candidate profile
		pl.plot(range(len(self.target)), calc_target, color='b', label='fitted')
		pl.title("Iter " + str(iter_num+1) + ": " + str(best_cand))
		pl.legend()

		#finalise plot
		display.clear_output(wait=True)
		if iter != self.max_iters-1:
			display.display(pl.gcf())

	def fit(self, target, variables, eqn, constraints, plot=False, save_path=None):
		'''
		Genetic Algorithm fitting method
		arg: target values to be fitted to. Must be 1D array.
		arg: variables: list of arrays which are variables in the equation to fit. Each array in list must have the same length as the target array
		arg: eqn: equation to be fitted. When defining the equation, it needs to be done such that first come the coefficients to be fitted in the order of the constraints defined and then come the variables in the order they were defined
		arg: constraints: array of constraints having two columns (for the lower and upper bound of each of the coefficents to be fitted
		factors
		arg: plot: True if progress is to be plotted every 10 iterations. False if cost and best of the population is to be outputted every 10 iterations
		arg: save_path: Path to save progress and final fitted values. Needs to have file name except extenstion (i.e. .npy or .png)
		return: parameters of best candidate after fitting
		'''
		
		#Error Handling
		#if len(x) != len(y):
		#	raise ValueError("Length of x and y variables do not match")
		if constraints.shape[1] != 2:
			raise ValueError("Each contraint entry needs to have two values only: a lower bound and an upper bound")

		#Store inputs in object
		self.target = target
		self.vars = variables
		self.eqn = eqn
		self.constraints = constraints
		self.num_pars = eqn.__code__.co_argcount - len(variables)

		#Error Handling
		if len(constraints) != self.num_pars:
			self.target = None
			self.vars = None
			self.eqn = None
			self.constraints = None
			self.num_pars = -1
			raise ValueError("Number of constraints need to be the same as the number of parameters of the function to be fitted")

		#Initialise array to keep the cost of the best candidate of each iteration
		if plot == True:
			history = np.zeros(self.max_iters)

		#Create the initial population
		pop = self.__create_initial_population()

		#Sort the initial population
		pop = self.__sort_pop(pop)

		#Get the cost of the best candidate
		best_cost = pop[0, 0]

		#Counter to help in the termination condition
		same_cost_iter = 0

		#For each of the iterations, up to max_iters
		for iter in range(self.max_iters):

			#If want to save and iteration number is a facotr of 10	
			if save_path != None and iter % 10 == 0:

				#Calculate the values of the profiles
				calc_target = self.__calc_profile(pop[0, 1:])

				#Save
				self.__save(save_path, pop[0])

			#Update variables for termination check
			if pop[0, 0] == best_cost:
				same_cost_iter += 1
			else:
				best_cost = pop[0, 0]
				same_cost_iter = 0

			#Plot progress if wanted
			if plot == True:
				history[iter] = pop[0, 0]
				calc_target = self.__calc_profile(pop[0, 1:])
				self.__plot_timestep(history, pop[0], iter)

			#Else print best candidate
			else:
				print("Iter " + str(iter) + ": " + str(pop[0]))

			#Termination check
			if pop[0, 0] < self.term_threshold or same_cost_iter >= int(0.1 * self.max_iters):
				break
		
			#Create population for next iteration
			new_pop = self.__create_next_iter_pop(pop)

			#Sort new population
			new_pop = self.__sort_pop(new_pop)

			#Get new population ready for next iteration
			pop = copy.deepcopy(new_pop)

		#Print best candidate after fitting		
		print(pop[0])

		#Print profile generated by parameter choices
		print(self.__calc_profile(pop[0, 1:]))
		
		#Reset object
		self.target = None
		self.vars = None
		self.eqn = None
		self.constraints = None

		#Return
		return pop[0, 1:]
