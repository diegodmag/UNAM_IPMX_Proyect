
import math 
import numpy as np 
import matplotlib.pyplot as plt
import random as rnd 
import copy
import time as tm 
import sys

import queen_rep as qrep 

class GeneticAlg: 
	'''
	Class modeling the steps and operators of a genetic algorithm to resolve the n-queens problem 

	Attributes : 
	queens : int 
		Number of the n-queen problem
	pop_size : int 
		Size of the population
	current_pop : list :  Queen_Solution
		The current population of the current iteration 
	offspring : list :  Queen_Solution
		The next generation of individuals 
	sel_proportion : float  
		Proportion of the population to be selected by roullete selection, the rest is selected by elitism
		We setted between (.6 , .8)  
	cross_prob : float 
		Probability of crossover  (.8 , .9)
	mut_prob : float 
		Probability of select a individual from the population in order to be mutated (.1 , .2)
	'''

	def __init__(self, n_q, pop_s, p_sel,cross_p, mut_p, t):

		self.n_queens = n_q 
		self.pop_size = pop_s
		self.current_pop = np.array([])
		#self.offspring = np.array([])
		self.max_time = t  
		self.sel_proportion = p_sel
		self.cross_prob = cross_p
		self.mut_prob = mut_p
		self.optimal = (n_q*(n_q-1))/2

	def get_the_best(self):
		return sorted(self.current_pop, key = lambda solution : -solution.fitness)[0]		

	def show_pop(self):
		for ind in self.current_pop:
			print(ind)

	def init_population(self):
		'''
		Generate the initial population and asigns to the current population 
		Each individual is a permutation from the numbers 0 to n_queens (exclusive)
		'''
		init_pop = []

		for i in range(self.pop_size):
			init_pop.append(qrep.Queen_Solution(np.random.permutation(self.n_queens)))


		self.current_pop=np.array(init_pop)
		[ind.evaluate() for ind in self.current_pop]


	def selection_rl(self):
		'''
		Selection  by Roulette

		Returns : 
		select_pop : list : Queen_Solution
			The selected individuals to be parents 
		'''

		#Evaluate each individual
		[ind.evaluate() for ind in self.current_pop]
		#The total sum of fitness
		fit_sum = sum([ind.fitness for ind in self.current_pop])
		#Generate the probabilities array
		probs = [ind.fitness/fit_sum for ind in self.current_pop]

		#Selection by roulette
		return np.array(rnd.choices(self.current_pop, weights=probs, k=int(self.pop_size*self.sel_proportion)))

	def selection_elitism(self):

		'''
		Selection by elitism, select the first 1-sel_proportion individuals with the best fitness value 


		Returns: 
		elite : list : Queen_Solution
			The best individuals of the current population 
		'''
		[ind.evaluate() for ind in self.current_pop]
		return np.array(sorted(self.current_pop, key = lambda solution : -solution.fitness)[:self.pop_size-int(self.pop_size*self.sel_proportion)])		


	def crossover(self, p_1, p_2):
		'''
		Crossover operator for permutations  
	
		Args: 
		p1 : Queen_Solution
			first parent
		p2 : Queen_Solution
			second parent 

		Returns: 
		s_1 : Queen_Solution
			first son 
		s_2 : Queen_Solution
			second son 
		'''
		if (rnd.random() < self.cross_prob):
			#The crossover happends
			#First we generate the two points of crossover 
			cross_p_1 = math.floor(self.n_queens/4)
			cross_p_2 = math.floor(3*(self.n_queens/4))

			ra = range(self.n_queens)

			#Initialize the son's chromosomes
			son_1_chromosome = np.array([-1 for x in ra])
			son_2_chromosome = np.array([-1 for x in ra])

			#Set the first parents information 
			son_1_chromosome[cross_p_1:cross_p_2] = p_2.chromosome[cross_p_1:cross_p_2]
			son_2_chromosome[cross_p_1:cross_p_2] = p_1.chromosome[cross_p_1:cross_p_2]

			cont_select = cross_p_2
			cont_insert = cross_p_2

			while(-1 in son_1_chromosome):
				if p_1.chromosome[cont_select%self.n_queens] not in son_1_chromosome:
					son_1_chromosome[cont_insert%self.n_queens] = p_1.chromosome[cont_select%self.n_queens]
					cont_select = cont_select+1 
					cont_insert = cont_insert+1 
				else : 
					cont_select = cont_select+1 

			cont_select = cross_p_2
			cont_insert = cross_p_2

			while(-1 in son_2_chromosome):
				if p_2.chromosome[cont_select%self.n_queens] not in son_2_chromosome:
					son_2_chromosome[cont_insert%self.n_queens] = p_2.chromosome[cont_select%self.n_queens]
					cont_select = cont_select+1 
					cont_insert = cont_insert+1 
				else : 
					cont_select = cont_select+1 


			return qrep.Queen_Solution(np.array(son_1_chromosome)),qrep.Queen_Solution(np.array(son_2_chromosome)) 		

		else:
			return copy.deepcopy(p_1), copy.deepcopy(p_2)

	def crossover_pop(self,population):
		'''

		*The population must be a even number 

		'''
		offspring = []
		for i in range(0,len(population),2): 
			s_1, s_2 = self.crossover(population[i],population[i+1])
			offspring.append(s_1)
			offspring.append(s_2)

		return np.array(offspring)



	def mutate_individual(self,individual):
		'''
		Mutate an individual by swapping two random elements within the chromosome
		
		Args:
		individual : Queen_Solution
			individual mutated
		
		'''
		index_1, index_2 = rnd.sample(range(len(individual.chromosome)), 2)
		individual.chromosome[index_1], individual.chromosome[index_2] = individual.chromosome[index_2],individual.chromosome[index_1] 



	def mutation_simple(self):
		'''
		Mutation operator, for each individual check if the probability of mutation is less than a random 
		float, if is then it execute the funtion "mutate_individual". 
		The mutation only works when the offspring is ready (not empty)

		'''
		for ind in self.current_pop:
			if(rnd.random() < self.mut_prob):
				self.mutate_individual(ind)

	def execution(self):
		'''
		Whole execution of the genetic algortihm 
		
		Returns : 
			time : float 
				Total execution time 
			iteration / execution : int 
				Total time generation 
		'''
		start = tm.time()
		
		total_iterations = 0 
		ga.init_population()

		#while (tm.time() - start < self.max_time) and (self.get_the_best().fitness != self.optimal):
		#while tm.time() < self.max_time or self.get_the_best().fitness != self.optimal:
		#while(tm.time() < self.max_time):
		while(self.get_the_best().fitness != self.optimal):
			#Roullete Selection 
			roulette_selected = self.selection_rl() 
			#Crossover  
			roulette_offspring = self.crossover_pop(roulette_selected)
			#Elitism Selection 
			elite = self.selection_elitism() 
			#Union of the Selected individuals
			offspring = np.concatenate((roulette_offspring,elite),axis=0) 
			#Generational Replacement 
			self.current_pop = offspring
			#Mutate
			self.mutation_simple()
			total_iterations = total_iterations+1 


		end = tm.time()		
		return (end-start), total_iterations

	def visualize_board(self,arr):
  		# Tamaño del tablero
	    n = len(arr)

	    # Genera una matriz de ceros de tamaño n x n para representar el tablero
	    tablero = np.zeros((n, n))

	    # Marca las posiciones de las reinas en el tablero
	    for i in range(n):
	        fila = arr[i] - 1  # resta 1 ya que las filas se cuentan desde 1, pero los índices de la lista comienzan desde 0
	        tablero[fila][i] = 1

	    # Crea una figura y un eje
	    fig, ax = plt.subplots(figsize=(8, 8))

	    # Define el mapa de colores para el tablero
	    cmap = plt.get_cmap('binary')
	    cmap.set_bad(color='brown')
	    cmap.set_over(color='yellow')
	    cmap.set_under(color='gray')

	    # Dibuja el tablero de ajedrez
	    ax.imshow(tablero, cmap=cmap, extent=[-0.5, n-0.5, -0.5, n-0.5])

	    # Dibuja las reinas
	    for i in range(n):
	        fila = arr[i] - 1
	        ax.text(i, fila, '♛', ha='center', va='center', fontsize=30, color='black')

	    # Configura los ejes
	    ax.set_xticks(np.arange(n))
	    ax.set_yticks(np.arange(n))
	    ax.set_xticklabels(np.arange(1, n+1))
	    ax.set_yticklabels(np.arange(1, n+1)[::-1])
	    ax.tick_params(length=0)
	    ax.grid(True)
	    ax.set_aspect('equal')
	    ax.set_title('Tablero de Ajedrez con '+str(self.n_queens)+' Reinas', fontsize=20, fontweight='bold')

	    # Muestra la figura
	    plt.show()


def boxplot(sample_1):
		fig, ax = plt.subplots()
		bp = ax.boxplot([sample_1], showfliers=False)
		plt.show()

def rep_iter(total_rep, genetic_al):
	'''
	Execute the algortihm "total_rep" times

	Args: 
	total_rep : int 
		The total number of repetitions to execute the genetic 
	genetic_al : GeneticAlg 
		The 
	'''
	#Save the bests fitness 
	data = []
	#Save the times 
	times = []
	#Save the number of generations 
	gens = [] 
	for i in range(total_rep): 
		tmp_time, tmp_generations = genetic_al.execution()
		data.append(genetic_al.get_the_best().fitness)
		times.append(tmp_time)
		gens.append(tmp_generations)

	#AVG Time
	avg_time = sum(times)/len(times)
	avg_genes = sum(gens)/len(gens)
	avg_fitness = sum(data)/len(data)
	#print("AVG Tiempo : " + str(avg_time))
	print("AVG Geeneraciones: "+str(avg_genes))
	print(data)
	print("AVG Fitness: "+str(avg_fitness))
	print("Best possible fitness: "+str(genetic_al.get_the_best().max_conflics))

	#boxplot(data)

if __name__ == '__main__':

	

	number_q,popultation_size,prob_cross,prob_mut,time = int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])

	ga = GeneticAlg(number_q,popultation_size,.8,prob_cross,prob_mut,time)
	
	#rep_iter(10,ga)
	ga.execution()
	print(ga.get_the_best())	
	ga.visualize_board(ga.get_the_best().chromosome)
	

	



	




