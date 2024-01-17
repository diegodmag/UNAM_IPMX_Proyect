import numpy as np 
import matplotlib.pyplot as plt
import time as tm 
import sys
import permutation_based_problems as perproblems 
import selection_operators as selop 
import crossover_operators as crossop
import mutation_operators as mutop
import generational_replacement_operators as genreplacement 
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
	tournament_size : int 
		Sample's size to consider for a selection 
	'''

	def __init__(self, per_size, pop_s, p_sel,t_s,cross_p, mut_p, t):

		self.permutation_size = per_size 
		self.pop_size = pop_s
		self.current_pop = np.array([])
		#self.offspring = np.array([])
		self.max_time = t  
		#Ya no se usa la proporcion de seleccion
		self.sel_proportion = p_sel
		self.cross_prob = cross_p
		self.mut_prob = mut_p
		self.tournament_size = t_s 

		#Se declara asi para que se puedan usar sus metodos 
		self.permutation_based_problem = perproblems.NQueens([0])
		self.optimal = self.permutation_based_problem.optimal
		
		#Tal vez aqui se debe determinar que tipo de operador vamos a usar 
		self.selection_operator = selop.Tournament(3)
		#self.selection_operator = selop.Roulette()
		#Recibe de parametro la probabilidad de cruzarlos 
		#self.crossover_operator = crossop.Basic(self.cross_prob)
		#self.crossover_operator = crossop.PMX(self.cross_prob)
		self.crossover_operator = crossop.IMPX(self.cross_prob)
		self.mutation_operator = mutop.SimpleSwap(self.mut_prob)
		#Replacement 
		self.generation_replacement = genreplacement.ElitismMuPlusLambda(self.pop_size)
		
		
	def get_the_best(self):
		#return sorted(self.current_pop, key = lambda solution : -solution.fitness)[0]
		'''
		Obtiene el mejor individuo de la población (el de menor fitness)
		La función sorted los ordena de menor a mayor, por lo que el elemento en la posición 0 es el de menor fitness
		'''
		return sorted(self.current_pop, key = lambda solution : solution.fitness)[0]		

	def show_pop(self):
		for ind in self.current_pop:
			print(ind)

	def init_population(self):
		'''
		Generate the initial population and asigns to the current population 
		Each individual is a permutation from the numbers 0 to permutation_size (exclusive)
		'''
		init_pop = []

		for i in range(self.pop_size):
			#init_pop.append(qrep.Queen_Solution(np.random.permutation(self.permutation_size)))
			init_pop.append(self.permutation_based_problem.get_instance(np.random.permutation(self.permutation_size)))
		self.current_pop=np.array(init_pop)
				
	
	#Tenemos que hacer una ejecucion individual de la cual obtengamos varios datos 
	#Tal vez guardarlos en un .csv 
	#Luego hacemos una que repita varias 


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
		#for i in range(2):

		while True :
			[ind.evaluate() for ind in self.current_pop]
			if(self.get_the_best().fitness == self.optimal):
				break
			else:
				#SELECTION
				selected = self.selection_operator.select(self.current_pop, self.pop_size)
				
			
			pop_chromosomes = self.crossover_operator.cross_population(selected, self.pop_size)
			tournament_offspring = self.permutation_based_problem.get_population(pop_chromosomes)
			offspring = self.generation_replacement.replace(self.current_pop, tournament_offspring)
			self.current_pop = offspring
			self.mutation_operator.mutate_population(self.current_pop)
			total_iterations = total_iterations+1
				
		end = tm.time()		
		#Una ultima evaluacion de la ultima generacion 
		
		[ind.evaluate() for ind in self.current_pop]
		
		
		return (end-start), total_iterations

def boxplot(sample_1):
		fig, ax = plt.subplots()
		bp = ax.boxplot([sample_1], showfliers=False)
		plt.show()


	
# Este es el que ejecuta el algoritmo varias veces 
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
	#print(data)
	print("AVG Fitness: "+str(avg_fitness))
	print("Best possible fitness: "+str(genetic_al.get_the_best().max_conflics))
		
	#boxplot(data)

if __name__ == '__main__':

	# Condicion de termino : Alcanza optimo o maximo de generaciones 

	number_q,popultation_size,prob_cross,prob_mut,time = int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])
	
	
	#El 3 es el tamanio del torneo 
	ga = GeneticAlg(number_q,popultation_size,.8,3,prob_cross,prob_mut,time)
	
	ga.execution()
	print(ga.get_the_best())
	#ga.get_the_best().output_plot()
	#plot_queens(ga.get_the_best(),outputName) 

	
	



	




