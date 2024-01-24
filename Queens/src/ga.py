import numpy as np 
import permutation_based_problems as perproblems 
import selection_operators as selop 
import crossover_operators as crossop
import mutation_operators as mutop
import generational_replacement_operators as genreplacement 

class GeneticAlg: 
	'''
	Class modeling the steps and operators of a genetic algorithm to resolve the n-queens problem 

	Attributes : 
	permutation_size : int 
		Number of the n-permutation problem
	pop_size : int 
		Size of the population
	cross_prob : float 
		Probability of crossover  (.8 , .9)
	mut_prob : float 
		Probability of select a individual from the population in order to be mutated (.1 , .2)
	current_pop : list :  Queen_Solution
		The current population of the current iteration  
	max_generations : int 
		Max number of generations for the algorithm 
	max_time : int 
		Max execution time for the algorithm

	#-> Por definir : permutation_based_problem (Por defecto el de las n-queens)
	optimal : int/float 
		The optimal value for the permutation based problem
	selection_operator : SelectionOp
		Selection Operators 
			0 -> Roulette
			1 -> Tournament
	tournament_size : int 
		Sample's size to consider for a selection 
	crossover_operator : CrossoverOp
		Crossover Operators
			0 -> Basic
			1 -> PMX
			2 -> IMPX
	mutation_operator : MutationOp
		Mutation Operators 
			0 -> SimpleSwap
	generation_replacement : class GenerationalReplacement:
		Generational Replacement Operator 
			0 -> ElitismMuPlusLambda
 	
	'''

	def __init__(self,per_size, pop_s,cross_p, mut_p,max_gen ,t, sel_op,t_s,croos_op, mut_op, gen_rep_op):

		
		self.permutation_size = per_size 
		self.pop_size = pop_s
		self.cross_prob = cross_p
		self.mut_prob = mut_p
		self.current_pop = np.array([])
		self.max_generations = max_gen
		self.max_time = t  
		
		#Problema con representacion de la solucion en permutacion 
		self.permutation_based_problem = perproblems.NQueens([0])
		#El optimo de ese problema
		self.optimal = self.permutation_based_problem.optimal
		self.tournament_size = t_s 
		
		self.selection_operator = self.set_selection_op(sel_op)

		self.crossover_operator = self.set_crossover_op(croos_op)
		
		self.mutation_operator = self.set_mutation_op(mut_op)

		self.generation_replacement = self.set_generational_replacement_op(gen_rep_op)

		
	def set_selection_op(self,sel_op):
		#Operadores de SELECCION 
		if(sel_op==0):
			return selop.Roulette()
		elif(sel_op==1):
			return selop.Tournament(self.tournament_size)
		else:
			#Por defecto toma el 3-torneo
			return selop.Tournament(3)			

	def set_crossover_op(self,croos_op): 
		#Operadores de CRUZA  
		if(croos_op==0):
			return crossop.Basic(self.cross_prob)
		elif(croos_op==1):
			return crossop.PMX(self.cross_prob)
		elif(croos_op==2):
			return crossop.IMPX(self.cross_prob)
		else:
			#Por defecto toma PMX
			return crossop.PMX(self.cross_prob)
		
	def set_mutation_op(self, mut_op):
		#Operadores de MUTACION
		if(mut_op==0):
			return mutop.SimpleSwap(self.mut_prob)
		else: 
			#Por defecto toma SimpleSwap
			return mutop.SimpleSwap(self.mut_prob)
	
	def set_generational_replacement_op(self,gen_rep_op):
		#Operadores de REEMPLAZO 
		if(gen_rep_op==0):
			return genreplacement.ElitismMuPlusLambda(self.pop_size)
		else:
			#Por defecto toma el Reemplazo Generacional
			return genreplacement.ElitismMuPlusLambda(self.pop_size) 

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

	def simple_individual_execution(self):
	
		[ind.evaluate() for ind in self.current_pop]
		selected = self.selection_operator.select(self.current_pop, self.pop_size)
		primitive_offspring_chromosomes = self.crossover_operator.cross_population(selected, self.pop_size)
		primitive_offspring_population = self.permutation_based_problem.get_population(primitive_offspring_chromosomes)
		[ind.evaluate() for ind in primitive_offspring_population]
		self.current_pop = self.generation_replacement.replace(self.current_pop, primitive_offspring_population)
		self.mutation_operator.mutate_population(self.current_pop)
		[ind.evaluate() for ind in self.current_pop] 

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
		best_son = self.get_the_best(primitive_offspring_population).fitness #MEJOR HIJO
		offspring_avg_fitness = self.get_avg_fitness(primitive_offspring_population) #PROMEDIO DE FITNESS DE HIJOS
		#REEMPLAZO GENERACIONAL 
		self.current_pop = self.generation_replacement.replace(self.current_pop, primitive_offspring_population)
		#MUTACION 
		self.mutation_operator.mutate_population(self.current_pop)
		[ind.evaluate() for ind in self.current_pop]
		best_current_pop = self.get_the_best(self.current_pop).fitness #MEJOR DE LA EJECUCION 
		current_pop_avg_fitness = self.get_avg_fitness(self.current_pop) #PROMEDIO DE FITNESS 
		
		
		return best_son, offspring_avg_fitness, best_current_pop, current_pop_avg_fitness
	

	def execution(self):
		'''
		Whole execution of the genetic algortihm 
		
		Returns : 
			time : float 
				Total execution time 
			iteration / execution : int 
				Total time generation 
		'''
		#while (tm.time() - start < self.max_time) and (self.get_the_best().fitness != self.optimal):
		#while tm.time() < self.max_time or self.get_the_best().fitness != self.optimal:
		#while(tm.time() < self.max_time):
		#for i in range(2):
		
		#Contenedores de datos 
		best_son_data = []
		offspring_avg_fitness_data =[]
		best_current_pop_data = []
		current_pop_avg_fitness_data = []
		generations_data = []
		self.init_population()
		generations = 0
		#CONDICIONES DE TERMINO IMPORTANTES -> Por generacion y por alcanzar el optimo 
		while True:
			#if(self.get_the_best(self.current_pop).fitness == self.optimal or generations == self.max_generations):
			if(generations == self.max_generations):	
				#Aqui hay que revisar la condicion de paro (falta la condición por generacion maxima alcanzada)
				#print(self.get_the_best(self.current_pop))
				break 
			else : 
				#print("Iteracion :"+str(total_iterations))
				best_son, offspring_avg_fitness, best_current_pop, current_pop_avg_fitness =  self.individual_execution()
				generations+=1
				best_son_data.append(best_son)
				offspring_avg_fitness_data.append(offspring_avg_fitness)
				best_current_pop_data.append(best_current_pop)
				current_pop_avg_fitness_data.append(current_pop_avg_fitness)
				generations_data.append(generations)
				#Aqui podemos guardar los datos o regresarselos a alguien 
				#print(">>>>>>>>>>>>>>>>>>>>>")
		
		return generations_data, best_son_data, offspring_avg_fitness_data, best_current_pop_data, current_pop_avg_fitness_data
	
	#BORRAR >>>>>
	def simple_execution(self):
		'''
		Whole execution of the genetic algortihm 
		
		Returns : 
			time : float 
				Total execution time 
			iteration / execution : int 
				Total time generation 
		'''
		self.init_population()
		generations = 0
		#CONDICIONES DE TERMINO IMPORTANTES -> Por generacion y por alcanzar el optimo 
		while True:
			#if(self.get_the_best(self.current_pop).fitness == self.optimal or generations == self.max_generations):
			if(generations == self.max_generations):	
				break 
			else : 
				
				self.simple_individual_execution()
				generations+=1
				
				
		return self.get_the_best(self.current_pop)

	
	




