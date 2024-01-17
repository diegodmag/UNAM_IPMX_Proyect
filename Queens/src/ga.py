import math 
import numpy as np 
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random as rnd 
import copy
import time as tm 
import sys
import os 
import queen_rep as qrep 
import selection_operators as selop 
import crossover_operators as crossop
import mutation_operators as mutop
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

	def __init__(self, n_q, pop_s, p_sel,t_s,cross_p, mut_p, t):

		self.n_queens = n_q 
		self.pop_size = pop_s
		self.current_pop = np.array([])
		#self.offspring = np.array([])
		self.max_time = t  
		#Ya no se usa la proporcion de seleccion
		self.sel_proportion = p_sel
		self.cross_prob = cross_p
		self.mut_prob = mut_p
		self.tournament_size = t_s 

		#Tal vez aqui se debe determinar que tipo de operador vamos a usar 
		self.selection_operator = selop.Tournament(3)
		#self.selection_operator = selop.Roulette()
		#self.crossover_operator = crossop.Basic()
		#self.crossover_operator = crossop.PMX()
		#Recibe de parametro la probabilidad de cruzarlos 
		self.crossover_operator = crossop.IMPX(self.cross_prob)
		self.mutation_operator = mutop.SimpleSwap(self.mut_prob)
		#self.optimal = (n_q*(n_q-1))/2
		self.optimal=0 
		#self.optimal = (n_q*(n_q-1))/2

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
		Each individual is a permutation from the numbers 0 to n_queens (exclusive)
		'''
		init_pop = []

		for i in range(self.pop_size):
			init_pop.append(qrep.Queen_Solution(np.random.permutation(self.n_queens)))


		self.current_pop=np.array(init_pop)
		#[ind.evaluate() for ind in self.current_pop]
		#[ind.evaluate_min() for ind in self.current_pop]


	#COMIENZAN OPERADORES DE SELECCION 

	def selection_rl(self):
		'''
		Selection  by Roulette

		Returns : 
		select_pop : list : Queen_Solution
			The selected individuals to be parents 
		'''

		#Evaluate each individual
		#[ind.evaluate() for ind in self.current_pop]
		#[ind.evaluate_min() for ind in self.current_pop]
		#The total sum of fitness
		fit_sum = sum([ind.fitness for ind in self.current_pop])
		#Generate the probabilities array
		probs = [ind.fitness/fit_sum for ind in self.current_pop]
		probs = np.array(probs)


		#Selection by roulette
		#Vamos a seleccionar tantos como poblacion haya 
		#return np.array(rnd.choices(self.current_pop, weights=probs, k=int(self.pop_size*self.sel_proportion)))
		return np.random.choice(self.current_pop, self.pop_size, p=probs)


	def selection_elitism(self):
		#ESTE NO ESTA FUNCIONAL, LA SELECCION NO DEBE CONSIDERAR AL self.sel_proportion
		'''
		Selection by elitism, select the first 1-sel_proportion individuals with the best fitness value 
		Returns: 
		elite : list : Queen_Solution
			The best individuals of the current population 
		'''
		#[ind.evaluate() for ind in self.current_pop]
		#[ind.evaluate_min() for ind in self.current_pop]
		#return np.array(sorted(self.current_pop, key = lambda solution : -solution.fitness)[:self.pop_size-int(self.pop_size*self.sel_proportion)])
		return np.array(sorted(self.current_pop, key = lambda solution : solution.fitness)[:self.pop_size-int(self.pop_size*self.sel_proportion)])		

	def selection_tournament(self):
		
		#Cuantas veces vamos a realizar un torneo = k=int(self.pop_size*self.sel_proportion))
		
		# 1 : np.random.choice(self.current_pop,self.tournament_size,False) -> Obtenemos 3 soluciones de entre todas las que hay en la poblacion
		# 2 : self.get_best_among(np.random.choice(self.current_pop,self.tournament_size,False)) -> De esas soluciones obtenemos la que menor fitnees tiene (la mejor)
		# 3 : [self.get_best_among(np.random.choice(self.current_pop,self.tournament_size,False)) for i in range(int(self.pop_size*self.sel_proportion))] 
		# 	-> Realizamos paso 1 y 3 una cantidad int(self.pop_size*self.sel_proportion) de veces  
		# 4 : Hacemos un np.array de esa lista 

		# IMPORTANTE >>> SE PUEDE DAR EL CASO QUE LA K DEL TORNEO SEA 0 o 1 y eso puede provocar porblemas
		#[ind.evaluate_min() for ind in self.current_pop] <----- Esto no afecta 
		
		selection = []

		for i in range(self.pop_size):
			#Esto es sin reemplazo ->participants = np.random.choice(self.current_pop,self.tournament_size,False)
			participants = np.random.choice(self.current_pop,self.tournament_size)
			chosen =self.get_best_among(participants)
			selection.append(chosen)
		
		return np.array(selection)
		#return np.array([self.get_best_among(np.random.choice(self.current_pop,self.tournament_size,False)) for i in range(int(self.pop_size*self.sel_proportion))])
	

	def get_best_among(self,pop_sample):
		'''
		Obtiene la mejor solucion (minimización) dado un arreglo de soluciones 	
		'''
		#Obtenemos los fitness de todas las soluciones de la muestra de la poblacion y las almacenamos 
		fitness_arr = np.array([solution.fitness for solution in pop_sample])
		#Obtenemos el indice del fitness menor 
		min_index = np.argmin(fitness_arr)
		#Regresamos ese objeto 
		return pop_sample[min_index]	
	
	#COMIENZAN OPERADORES DE CRUZA

	#Necesitamos un metodo para generar una cantidad de representaciones de reinas 
	#Por cada permutacion recibida 
	def get_queens_population(self, chromosomes): 
		queen_pop = []
		for chromosome in chromosomes:
			queen_pop.append(qrep.Queen_Solution(np.array(chromosome)))
		return np.array(queen_pop)


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


	def pmx_mapping_relation_generation(self,subs_str_1,subs_str_2):
		'''
		Funcion que recibe dos subcadenas y genera una relacion de mapeo entre ellas
		regresa dos diccionarios 
		'''
		map_rel_R = {}
		map_rel_L = {}
		
		for i in range(len(subs_str_1)):
			#if subs_str_1[i] != subs_str_2[i]: #Esta condicion es para evitar datos tipo x:x, ACTUALIZACION : SE TIENE QUE QUITAR 
			map_rel_R[subs_str_2[i]]= subs_str_1[i]
			map_rel_L[subs_str_1[i]]= subs_str_2[i]

		return map_rel_R, map_rel_L

	def crossover_pmx(self, p_1, p_2):
		
		#Generar los puntos de corte 
		all_cut_points = np.arange(self.n_queens)
		cut_points = sorted(np.random.choice(all_cut_points,2,replace=False))
		#Este se puede dar el caso de solo se intercambien dos elementos 
		cp_1 , cp_2= cut_points[0],cut_points[1]
	
		rel_R, rel_L = self.pmx_mapping_relation_generation(p_1.chromosome[cp_1:cp_2],p_2.chromosome[cp_1:cp_2])
		
		#Generamos dos copias de los cromosomas de los padres 
		primitive_offspring_1_chromosome = p_1.chromosome.copy()
		primitive_offspring_2_chromosome = p_2.chromosome.copy()

		#Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte 
		primitive_offspring_1_chromosome[cp_1:cp_2] = p_2.chromosome[cp_1:cp_2]
		primitive_offspring_2_chromosome[cp_1:cp_2] = p_1.chromosome[cp_1:cp_2]

		for i in range(0,cp_1):
			while primitive_offspring_1_chromosome[i] in rel_R:
				primitive_offspring_1_chromosome[i] = rel_R[primitive_offspring_1_chromosome[i]] 
			while primitive_offspring_2_chromosome[i] in rel_L: 
				primitive_offspring_2_chromosome[i] = rel_L[primitive_offspring_2_chromosome[i]]
		
		for i in range(cp_2,len(primitive_offspring_1_chromosome)):
			while primitive_offspring_1_chromosome[i] in rel_R:
				primitive_offspring_1_chromosome[i] = rel_R[primitive_offspring_1_chromosome[i]] 
			while primitive_offspring_2_chromosome[i] in rel_L: 
				primitive_offspring_2_chromosome[i] = rel_L[primitive_offspring_2_chromosome[i]]

	
		return qrep.Queen_Solution(np.array(primitive_offspring_1_chromosome)),qrep.Queen_Solution(np.array(primitive_offspring_2_chromosome))
	
	def crossover_pmx_debugged(self, p_1, p_2):
		
		#Generar los puntos de corte 
		all_cut_points = np.arange(self.n_queens)
		cut_points = sorted(np.random.choice(all_cut_points,2,replace=False))
		#Este se puede dar el caso de solo se intercambien dos elementos 
		cp_1 , cp_2= cut_points[0],cut_points[1]
		print("Cutting points :"+str(cp_1)+" ,"+str(cp_2))

		rel_R, rel_L = self.pmx_mapping_relation_generation(p_1.chromosome[cp_1:cp_2],p_2.chromosome[cp_1:cp_2])
		print("Mapping Relationship R :"+str(rel_R))
		print("Mapping Relationship L :"+str(rel_L))

		#Generamos dos copias de los cromosomas de los padres 
		
		primitive_offspring_1_chromosome = p_1.chromosome.copy()
		primitive_offspring_2_chromosome = p_2.chromosome.copy()

		print("Parent 1 :"+str(primitive_offspring_1_chromosome))
		print("Parent 2 :"+str(primitive_offspring_2_chromosome))

		#Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte 
		primitive_offspring_1_chromosome[cp_1:cp_2] = p_2.chromosome[cp_1:cp_2]
		primitive_offspring_2_chromosome[cp_1:cp_2] = p_1.chromosome[cp_1:cp_2]

		print("Substring 1 :"+str(primitive_offspring_1_chromosome[cp_1:cp_2]))
		print("Substring 2 :"+str(primitive_offspring_2_chromosome[cp_1:cp_2]))

		print("Primitive Offsrping 1 : "+str(primitive_offspring_1_chromosome))
		print("Primitive Offsrping 2 : "+str(primitive_offspring_2_chromosome))


		for i in range(0,cp_1):
			while primitive_offspring_1_chromosome[i] in rel_R:
				primitive_offspring_1_chromosome[i] = rel_R[primitive_offspring_1_chromosome[i]] 
			while primitive_offspring_2_chromosome[i] in rel_L: 
				primitive_offspring_2_chromosome[i] = rel_L[primitive_offspring_2_chromosome[i]]
		
		for i in range(cp_2,len(primitive_offspring_1_chromosome)):
			while primitive_offspring_1_chromosome[i] in rel_R:
				primitive_offspring_1_chromosome[i] = rel_R[primitive_offspring_1_chromosome[i]] 
			while primitive_offspring_2_chromosome[i] in rel_L: 
				primitive_offspring_2_chromosome[i] = rel_L[primitive_offspring_2_chromosome[i]]

	
		return qrep.Queen_Solution(np.array(primitive_offspring_1_chromosome)),qrep.Queen_Solution(np.array(primitive_offspring_2_chromosome))

	def crossover_impx_debugged(self, p_1, p_2):

		#Paso 1
		n = len(p_1.chromosome)
		all_cut_points = np.arange(self.n_queens)
		cut_points = sorted(np.random.choice(all_cut_points,2,replace=False))
		#Este se puede dar el caso de solo se intercambien dos elementos 
		cp_1 , cp_2= cut_points[0],cut_points[1]
		print("Cutting points :"+str(cp_1)+" ,"+str(cp_2))

		#Generamos dos copias de los cromosomas de los padres 	
		primitive_offspring_1_chromosome = p_1.chromosome.copy()
		primitive_offspring_2_chromosome = p_2.chromosome.copy()

		print("Parent 1 :"+ str(primitive_offspring_1_chromosome))
		print("Parent 2 :"+ str(primitive_offspring_2_chromosome))

		subs_1 = p_2.chromosome[cp_1:cp_2]
		subs_2 = p_1.chromosome[cp_1:cp_2]
		
		print("Substring 1:"+str(subs_1))
		print("Substring 2:"+str(subs_2))

		#Paso 2
		#Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte 
		primitive_offspring_1_chromosome[cp_1:cp_2] = p_2.chromosome[cp_1:cp_2]
		primitive_offspring_2_chromosome[cp_1:cp_2] = p_1.chromosome[cp_1:cp_2]

		print("Primitive Offspring 1 :"+ str(primitive_offspring_1_chromosome))
		print("Primitive Offspring 2 :"+ str(primitive_offspring_2_chromosome))

		#Paso 3 y 4: Generamos e inicializamos la exchange list  (O(m))
		m = len(p_2.chromosome[cp_1:cp_2]) 
		exchange_list = np.zeros((m,3), dtype=int)
		
		for i in range(m): 
			exchange_list[i,0] = subs_1[i]
			exchange_list[i,1] = subs_2[i]
			exchange_list[i,2] = 1

		print("Initial Exchange List :")
		print(exchange_list)
		

		#Paso 5,6 y 7 
		#Generacion de la guide list 
		#Nuestra implementacion tiene que usar -1 en vez de 0 por que el 0 se ocupa en nuestra representacion 
		guide_list = np.full(n,-1,dtype=int)
		for i in range(m):
			guide_list[exchange_list[i,0]]=exchange_list[i,1]
		
		print("Initial Guide List")
		print(guide_list)
		
		#Generacion L1, L2 y L1+L2 
		l1 = np.zeros(n,dtype=int)
		l2 = np.zeros(n,dtype=int)
		for i in range(m): 
			l1[exchange_list[i,0]] = 1 
			l2[exchange_list[i,1]] = 1
		print("l1 :"+str(l1))
		print("l2 :"+str(l2))

		l1_l2 = l1+l2 
		print("L1 + L2 :"+str(l1_l2))
		
		
		#Actualizar la exchange list haciendo los nodos intermedios 0 
		for i in range(m):
			index = exchange_list[i,0] 
			if l1_l2[index] == 2 : 
				exchange_list[i,2] = 0
				
		
		print("Updated Exchange List (Removing mid nodes) :")
		print(exchange_list)

		#Actualizar exchange list eliminando nodos intermedios y conectado los nodos con camino directo 
		for i in range(m):
			if exchange_list[i,2] == 1 : 
				#El reemplazo actual
				current_replacement = exchange_list[i,1]
				#El reemplazo final 
				final_replacement = -1
				count = 0  
				while current_replacement != -1:#Al menos se cumple la primera vez 
					count = count +1 
					final_replacement = current_replacement
					current_replacement = guide_list[final_replacement] 
					 #Aqui es donde la busqueda "salta" en la guide_list buscando 
				
				exchange_list[i,1] = final_replacement
				

		print("Updated Exchange List (New Paths)")
		print(exchange_list)

		#Ahora actualizamos la guide list 
		for i in range(m):
			if(exchange_list[i,2]==1):
				guide_list[exchange_list[i,0]] =  exchange_list[i,1]
			else:
				guide_list[exchange_list[i,0]] = -1
		
		print("Updated Guide list")
		print(guide_list)

		###PASO  8  
		for i in range(cp_1):

			#Primero queremos el gen en el padre (excluyendo los genes entre los puntos de corte)
			#Para ese gen buscamos si su valor en la guide list es distinto de -1
			#Si lo es, entonces lo sustituimos por el nuevo valor actualizado de la exchange list
			value = guide_list[p_1.chromosome[i]]
			if value != -1:
				primitive_offspring_1_chromosome[i] = value
		
		for i in range(cp_2,len(primitive_offspring_1_chromosome)):
			#primitive_offspring_1_chromosome
			value = guide_list[p_1.chromosome[i]]
			if value != -1:
				primitive_offspring_1_chromosome[i] = value

		print("Offspring 1 : Legalized :")
		print(str(primitive_offspring_1_chromosome))

		f = np.full(n,-1,dtype=int)
		for i in range(n): 
			f[primitive_offspring_1_chromosome[i]]=p_1.chromosome[i]

		for i in range(n):
			primitive_offspring_2_chromosome[i]=f[p_2.chromosome[i]]
		
		print("Offspring 2 : Legalized :")
		print(str(primitive_offspring_2_chromosome))

		return qrep.Queen_Solution(np.array(primitive_offspring_1_chromosome)),qrep.Queen_Solution(np.array(primitive_offspring_2_chromosome))
	
	def crossover_impx(self, p_1, p_2):

		#Paso 1
		n = len(p_1.chromosome)
		all_cut_points = np.arange(self.n_queens)
		cut_points = sorted(np.random.choice(all_cut_points,2,replace=False))
		#Este se puede dar el caso de solo se intercambien dos elementos 
		cp_1 , cp_2= cut_points[0],cut_points[1]

		#Generamos dos copias de los cromosomas de los padres 	
		primitive_offspring_1_chromosome = p_1.chromosome.copy()
		primitive_offspring_2_chromosome = p_2.chromosome.copy()

		subs_1 = p_2.chromosome[cp_1:cp_2]
		subs_2 = p_1.chromosome[cp_1:cp_2]
		
		#Paso 2
		#Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte 
		primitive_offspring_1_chromosome[cp_1:cp_2] = p_2.chromosome[cp_1:cp_2]
		primitive_offspring_2_chromosome[cp_1:cp_2] = p_1.chromosome[cp_1:cp_2]

		#Paso 3 y 4: Generamos e inicializamos la exchange list  (O(m))
		m = len(p_2.chromosome[cp_1:cp_2]) 
		exchange_list = np.zeros((m,3), dtype=int)
		
		for i in range(m): 
			exchange_list[i,0] = subs_1[i]
			exchange_list[i,1] = subs_2[i]
			exchange_list[i,2] = 1

		#Paso 5,6 y 7 
		#Generacion de la guide list 
		#Nuestra implementacion tiene que usar -1 en vez de 0 por que el 0 se ocupa en nuestra representacion 
		guide_list = np.full(n,-1,dtype=int)
		for i in range(m):
			guide_list[exchange_list[i,0]]=exchange_list[i,1]
		
		#Generacion L1, L2 y L1+L2 
		l1 = np.zeros(n,dtype=int)
		l2 = np.zeros(n,dtype=int)
		for i in range(m): 
			l1[exchange_list[i,0]] = 1 
			l2[exchange_list[i,1]] = 1
		
		l1_l2 = l1+l2 
		
		#Actualizar la exchange list haciendo los nodos intermedios 0 
		for i in range(m):
			index = exchange_list[i,0] 
			if l1_l2[index] == 2 : 
				exchange_list[i,2] = 0
				
		#Actualizar exchange list eliminando nodos intermedios y conectado los nodos con camino directo 
		for i in range(m):
			if exchange_list[i,2] == 1 : 
				#El reemplazo actual
				current_replacement = exchange_list[i,1]
				#El reemplazo final 
				final_replacement = -1
				count = 0  
				while current_replacement != -1:#Al menos se cumple la primera vez 
					count = count +1 
					final_replacement = current_replacement
					current_replacement = guide_list[final_replacement] 
					 #Aqui es donde la busqueda "salta" en la guide_list buscando 
				
				exchange_list[i,1] = final_replacement
		
		#Ahora actualizamos la guide list 
		for i in range(m):
			if(exchange_list[i,2]==1):
				guide_list[exchange_list[i,0]] =  exchange_list[i,1]
			else:
				guide_list[exchange_list[i,0]] = -1
		
		###PASO  8  
		for i in range(cp_1):

			#Primero queremos el gen en el padre (excluyendo los genes entre los puntos de corte)
			#Para ese gen buscamos si su valor en la guide list es distinto de -1
			#Si lo es, entonces lo sustituimos por el nuevo valor actualizado de la exchange list
			value = guide_list[p_1.chromosome[i]]
			if value != -1:
				primitive_offspring_1_chromosome[i] = value
		
		for i in range(cp_2,len(primitive_offspring_1_chromosome)):
			#primitive_offspring_1_chromosome
			value = guide_list[p_1.chromosome[i]]
			if value != -1:
				primitive_offspring_1_chromosome[i] = value

		f = np.full(n,-1,dtype=int)
		for i in range(n): 
			f[primitive_offspring_1_chromosome[i]]=p_1.chromosome[i]

		for i in range(n):
			primitive_offspring_2_chromosome[i]=f[p_2.chromosome[i]]
		
		return qrep.Queen_Solution(np.array(primitive_offspring_1_chromosome)),qrep.Queen_Solution(np.array(primitive_offspring_2_chromosome))


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

	def crossover_pop_2(self,popultation):
		'''
		Esta tecnica de offpring permite que un individuo se cruce con sigo mismo
		'''
		offspring = []
		for i in range(int(self.pop_size/2)): 
			parents = np.random.choice(popultation, 2)
			s_1, s_2 = parents[0],parents[1] 
			offspring.append(s_1)
			offspring.append(s_2)

		return np.array(offspring)

	def crossover_pop_pmx(self, population):
		offspring = []
		for i in range(int(self.pop_size/2)): 
			parents = np.random.choice(population, 2)
			s_1, s_2 = self.crossover_pmx(parents[0],parents[1]) 
			offspring.append(s_1)
			offspring.append(s_2)

		return np.array(offspring)

	def crossover_pop_ipmx(self, population):
		offspring = []
		for i in range(int(self.pop_size/2)): 
			parents = np.random.choice(population, 2)
			s_1, s_2 = self.crossover_impx(parents[0],parents[1]) 
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

	def generational_replacement_elitism_mu_plus_lambda(self, offspring):
		'''
		offspring : needs to be a np.array 
		'''

		generarion_pool = np.hstack((self.current_pop,offspring)) 
		
		#Ordenamos el conjunto de la anterior poblacion y la nueva generacion y tomamos a los mejores individuos. 
		return sorted(generarion_pool, key = lambda solution : solution.fitness)[:self.pop_size] 


		

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
			[ind.evaluate_min() for ind in self.current_pop]
			if(self.get_the_best().fitness == self.optimal):
				break
			else:
				#SELECTION
				selected = self.selection_operator.select(self.current_pop, self.pop_size)
				#tournament_selected = selop.SelectionOp()
				#tournament_selected = self.selection_tournament()
			#Crossover  
			#roulette_offspring = self.crossover_pop(roulette_selected)
			#tournament_offspring = self.crossover_pop_2(tournament_selected)
			#tournament_offspring = self.crossover_pop_pmx(tournament_selected)
			
			#IMPORTANTE, EL CROSSOVER ESTA CONDICIONADO A UNA PROBA
			#ESTUDIAR ESTO 
			# if(rnd.random() < self.cross_prob):
			# 	#tournament_offspring = self.crossover_pop_ipmx(selected)
			# 	#Aqui nos van a regresar un arreglo de permutaciones 
			# 	pop_chromosomes = self.crossover_operator.cross_population(selected, self.pop_size)
			# 	tournament_offspring = self.get_queens_population(pop_chromosomes)
			#pop_chromosomes = self.crossover_operator.cross_population(selected, self.pop_size)
			pop_chromosomes = self.crossover_operator.cross_population(selected, self.pop_size)
			tournament_offspring = self.get_queens_population(pop_chromosomes)
			
			#tournament_offspring = self.crossover_pop_ipmx(selected)
			#crossover_pop_pmx
			#Elitism Selection  ---> El elitismo debe ser entre padres e hijos 
			#elite = self.selection_elitism() 
			#Union of the Selected individuals
			#offspring = np.concatenate((roulette_offspring,elite),axis=0)
			#offspring = np.concatenate((tournament_offspring,elite),axis=0) 
			offspring = self.generational_replacement_elitism_mu_plus_lambda(tournament_offspring)
			#Generational Replacement 
			self.current_pop = offspring
			#Mutate
			self.mutation_operator.mutate_population(self.current_pop)
			total_iterations = total_iterations+1
				
		end = tm.time()		
		#Una ultima evaluacion de la ultima generacion 
		[ind.evaluate_min() for ind in self.current_pop]
		return (end-start), total_iterations


#Generado con GPT-3.5

def draw_recursive_diagonals(ax, x, y, direction_x, direction_y, n):
    if x < 0 or x >= n or y < 0 or y >= n:
        return
    ax.plot([x, x + direction_x], [y, y + direction_y], color='red', linewidth=2)
    draw_recursive_diagonals(ax, x + direction_x, y + direction_y, direction_x, direction_y, n)

def plot_queens(solution, outputName):
    n = len(solution.chromosome)
    chessboard = np.zeros((n, n), dtype=int)

    # Llenar el tablero con 1s en las posiciones de las reinas
    for row, col in enumerate(solution.chromosome):
        chessboard[row][col] = 1

    fig, ax = plt.subplots(figsize=(8, 8))

    # Dibujar el tablero de ajedrez
    for i in range(n):
        for j in range(n):
            is_white = (i + j) % 2 == 0
            color = 'white' if is_white else 'black'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
            queen_color = 'black' if is_white else 'white'  # Color opuesto al fondo
            if chessboard[i][j] == 1:
                queen_symbol = u'\u265B'  # Símbolo Unicode para la reina de ajedrez (♛)
                ax.text(j + 0.5, i + 0.5, queen_symbol, fontsize=24, ha='center', va='center', color=queen_color)

                # Dibujar líneas diagonales desde la reina hacia las esquinas
                draw_recursive_diagonals(ax, j, i, 1, 1, n)
                draw_recursive_diagonals(ax, j, i, -1, -1, n)
                draw_recursive_diagonals(ax, j, i, 1, -1, n)
                draw_recursive_diagonals(ax, j, i, -1, 1, n)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
	
    plt.savefig(get_path_for_output(outputName))
	#plt.savefig(get_path_for_output('Test'))
	


def boxplot(sample_1):
		fig, ax = plt.subplots()
		bp = ax.boxplot([sample_1], showfliers=False)
		plt.show()

def get_path_for_output(figure_name):


	current_adress = os.path.dirname(os.path.abspath(__file__))

	output_adress = os.path.join(current_adress,'..','output')

	if not os.path.exists(output_adress):
		os.makedirs(output_adress)

	return os.path.join(output_adress,figure_name)

	

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

	#Tarea 1 : Dejarlo como el algoritmo genetico generico LISTO 

	# Parametrizar la k del torneo, default 3 
	# Parametrizar el operador de cruza (cuando haya mas de uno ) y la probabilidad 
	# Parametrizar probabilidad de mutacion 
	# Parametrizar operador de reemplazo RemplazoGeneracional / Elitista mejores entre (hijos+padres)

	# Condicion de termino : Alcanza optimo o maximo de generaciones 

	number_q,popultation_size,prob_cross,prob_mut,time = int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])
	
	#Hay que definir un formato para guardar el plot de la representacion visual de la reina.
	outputName=str('Board'+str(number_q))

	#El 3 es el tamanio del torneo 
	ga = GeneticAlg(number_q,popultation_size,.8,3,prob_cross,prob_mut,time)
	
	ga.execution()
	print(ga.get_the_best())
	plot_queens(ga.get_the_best(),outputName) 

	
	#ga.execution()
	#print(ga.get_the_best())
	#ga.init_population()

	#rep_iter(10,ga)
	#time, iters = ga.execution()
	# print("Tiempo de ejecucion "+str(time))
	# print("Iteraciones usadas "+ str(iters))
	# print(ga.get_the_best())
	#plot_queens(ga.get_the_best().chromosome)
	
	
	# print("---------------")

	# listaSols = sorted(ga.current_pop, key = lambda solution : solution.fitness)
	# child1 = listaSols[0]
	# child2 = listaSols[-1]

	# offs_1,offs_2=  ga.crossover_impx(child1,child2)
	# print(offs_1)
	# print("----------")
	# print(offs_2)
	# i_start_time = tm.time()
	# i_cpu_start = tm.process_time()
	

	# permutation1= [2, 1, 9, 7, 8, 3, 6, 10, 5, 4]
	# permutation2 = [10, 1, 8, 4, 5, 9, 3, 6, 7, 2]
	
	#offspring_1,offspring_2  =  ga.crossover_impx(child1,child2)
	# print("Legalized offspring")
	# print(offspring)
	# offspring_1,offspring_2 = ga.crossover_pmx(child1,child2)

	# print(offspring_1.chromosome)
	# print(offspring_2.chromosome)


	# i_end_time = tm.time()
	# i_cpu_end = tm.process_time()

	# total_time = i_end_time -i_start_time  
	# cpu_used = i_cpu_end - i_cpu_start

	# print("Tiempo usado por PMX :"+str(total_time))



	# substr_1 = child1.chromosome[3:6]
	# print("Hijo 1 :"+str(child1.chromosome))
	# print("Subcadena 1 : "+str(substr_1))
	
	# substr_2 = child2.chromosome[3:6]
	# print("Hijo 1 :"+str(child2.chromosome))
	# print("Subcadena 1 : "+str(substr_2))
	
	# mapRel, ocurRel = ga.pmx_generate_mapping_relationship(substr_1,substr_2)
	# print("Relacion de mapeo :"+str(mapRel))
	# print("Relacion de ocurrencias :"+str(ocurRel))

	# #Vamos a buscar el reemplazo de un numero que esté en la relacion de mapeo 
	# numero_a_reemplazar = 0
	# for i in range(len(ocurRel)):
	# 	if ocurRel[i] == 1 : 
	# 		numero_a_reemplazar = i
	# 		print("Vamos a buscar reemplazar a "+str(numero_a_reemplazar))
	# 		break; 
	
	# end = ga.pmx_replacement(numero_a_reemplazar,mapRel,0)
	# print("Reemplazo encontrado :"+str(end))


	# print("Prueba de reemplazo")
	# map_rel_R, map_rel_L  = ga.pmx_mapping_relation_generation([4,2,13,10,8], [6,13,10,12,5])
	# print("Map Rel Right "+str(map_rel_R))
	# print("Map Rel Left "+str(map_rel_L))
	# #print
	# # (map_REL) 		

	# print("Esta es la lista tabu")
	# tabu = [0 for i in range(14)]
	# print(tabu)

	# num = 4
	# print("Se quiere reemplazar: "+str(num))

	# replace = ga.find_replacement_pmx(num,map_rel_R,map_rel_L,tabu)
	# print("Reemplazo :"+str(replace))
	# replace = ga.find_replacement(num,map_REL,tabu)

	
	#def find_replacement(self,elemnt,mapp_rel, tabu):

	#pmx_mapping_relation_generation
	#Esto regresa el reemplazo
	#def pmx_replacement(self,elemt, mapp_rel, end):

	# return np.array(mapping_r), np.array(mapping_ocurrences)
	# pmx_generate_mapping_relationship(self, subs_str_1,subs_str_2 )

	# print(child1.chromosome)
	# print(child2.chromosome)
	#ga.crossover_pmx(child1, child2)
	#ga.crossover_ipmx(child1, child2)
	
	# for sol in listaSols: 
	# 	print(sol)

	#aplot_queens(ga.get_the_best())
	#print(ga.get_best_among(ga.current_pop))
	#print(ga.get_the_best().)			
	#ga.visualize_board(ga.get_the_best().chromosome)
	

	



	




