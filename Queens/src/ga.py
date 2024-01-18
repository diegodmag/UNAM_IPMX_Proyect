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
		self.sel_proportion = p_sel
		self.tournament_size = t_s 
		self.cross_prob = cross_p
		self.mut_prob = mut_p
		self.max_time = t  
		self.current_pop = np.array([])
		
		#Necesitamos un parametro para determinar las generaciones 


		#Problema con representacion de la solucion en permutacion 
		self.permutation_based_problem = perproblems.NQueens([0])
		#El optimo de ese problema
		self.optimal = self.permutation_based_problem.optimal
		
		#Operadores de SELECCION 
		self.selection_operator = selop.Tournament(3)
		#self.selection_operator = selop.Roulette()
		#Operadores de CRUZA  
		#self.crossover_operator = crossop.Basic(self.cross_prob)
		#self.crossover_operator = crossop.PMX(self.cross_prob)
		self.crossover_operator = crossop.IMPX(self.cross_prob)
		#Operadores de MUTACION
		self.mutation_operator = mutop.SimpleSwap(self.mut_prob)
		#Operadores de REEMPLAZO 
		self.generation_replacement = genreplacement.ElitismMuPlusLambda(self.pop_size)
		
		
	def get_the_best(self, population):
		'''
		Obtiene el mejor individuo de una poblacion dada  (el de menor fitness)
		Params : 
			population : list[Object]
				Unas poblacion de invididuos con atributo fitness
		La función sorted los ordena de menor a mayor, por lo que el elemento en la posición 0 es el de menor fitness
		'''
		return sorted(population, key = lambda ind : ind.fitness)[0]		

	def get_avg_fitness(self, population):
		'''
		Obtiene el promedio de fitness dado una poblacion 
		Params : 
			population : list[Object]
				Unas poblacion de invididuos con atributo fitness
		'''
		return np.mean(np.array([ind.fitness for ind in population]))
		
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
			ind = self.permutation_based_problem.get_instance(np.random.permutation(self.permutation_size))
			ind.evaluate()
			init_pop.append(ind)
			
		self.current_pop=np.array(init_pop)

				
	#Tenemos que hacer una ejecucion individual de la cual obtengamos varios datos 
	#Tal vez guardarlos en un .csv 
	#Luego hacemos una que repita varias 

	# por cada iteracion debemos obtener :
	# best (mejor solucion)
	# avg (promedio fitness)
	# avg_hijos (promedio hijos)
	# best_hijo (mejor)

	#Necesitamos la mejor solucion 
	#El promedio del fitness 
	#Mejor hijo -> Es decir obtener el mejor de la cruza 
	#Promedio de fitness de hijos 

	def individual_execution(self):
		#Evaluamos a los individuos de la poblacion actual 
		[ind.evaluate() for ind in self.current_pop]
		#SELECCION
		selected = self.selection_operator.select(self.current_pop, self.pop_size)
		#CRUZA 1 -> OBTENEMOS UNA POBLACION DE PERMUTACIONES
		primitive_offspring_chromosomes = self.crossover_operator.cross_population(selected, self.pop_size)
		#CRUZA 2 -> GENERACION DE POBLACION BASADA EN LOS CROMOSOMAS 
		primitive_offspring_population = self.permutation_based_problem.get_population(primitive_offspring_chromosomes)
		[ind.evaluate() for ind in primitive_offspring_population]
		best_son = self.get_the_best(primitive_offspring_population) #MEJOR HIJO
		offspring_avg_fitness = self.get_avg_fitness(primitive_offspring_population) #PROMEDIO DE FITNESS DE HIJOS
		print("Mejor hijo :"+str(best_son.fitness))
		print("AVG Fitness Hijos :"+str(offspring_avg_fitness))
		#REEMPLAZO GENERACIONAL 
		self.current_pop = self.generation_replacement.replace(self.current_pop, primitive_offspring_population)
		#MUTACION 
		self.mutation_operator.mutate_population(self.current_pop)
		[ind.evaluate() for ind in primitive_offspring_population]
		best_current_pop = self.get_the_best(self.current_pop) #MEJOR DE LA EJECUCION 
		current_pop_avg_fitness = self.get_avg_fitness(self.current_pop) #PROMEDIO DE FITNESS 
		print("Mejor de iteracion :"+str(best_current_pop.fitness))
		print("AVG Fitness Iteracion :"+str(current_pop_avg_fitness))


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
		cont = 1
		#CONDICIONES DE TERMINO IMPORTANTES -> Por generacion y por alcanzar el optimo 
		while True:
			if(self.get_the_best(self.current_pop).fitness == self.optimal):
				#Aqui hay que revisar la condicion de paro (falta la condición por generacion maxima alcanzada)
				#print(self.get_the_best(self.current_pop))
				break 
			else : 
				#print("Iteracion :"+str(total_iterations))
				self.individual_execution()
				print(">>>>>>>>>>>>>>>>>>>>>")
				
		end = tm.time()		
			
		return (end-start), total_iterations
	
#Este hay que moverlo a uno de puras metricas 
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
		data.append(genetic_al.get_the_best(genetic_al.current_pop).fitness)
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
	print("Best possible fitness: "+str(genetic_al.get_the_best(genetic_al.current_pop).max_conflics))
		
	#boxplot(data)

if __name__ == '__main__':

	# Condicion de termino : Alcanza optimo o maximo de generaciones 

	number_q,popultation_size,prob_cross,prob_mut,time = int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])
	
	
	#El 3 es el tamanio del torneo 
	ga = GeneticAlg(number_q,popultation_size,.8,3,prob_cross,prob_mut,time)
	
	ga.execution()
	#print(ga.get_the_best(ga.current_pop))
	#ga.get_the_best().output_plot()
	#plot_queens(ga.get_the_best(),outputName) 

	
	



	




