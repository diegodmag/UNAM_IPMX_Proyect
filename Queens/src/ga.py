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

	def __init__(self, n_q, pop_s, p_sel,t_s,cross_p, mut_p, t):

		self.n_queens = n_q 
		self.pop_size = pop_s
		self.current_pop = np.array([])
		#self.offspring = np.array([])
		self.max_time = t  
		self.sel_proportion = p_sel

		self.tournament_size = t_s 

		self.cross_prob = cross_p
		self.mut_prob = mut_p
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
		Obtiene la mejor solucion (minimizaciónSS) dado un arreglo de soluciones 	
		'''
		#Obtenemos los fitness de todas las soluciones de la muestra de la poblacion y las almacenamos 
		fitness_arr = np.array([solution.fitness for solution in pop_sample])
		#Obtenemos el indice del fitness menor 
		min_index = np.argmin(fitness_arr)
		#Regresamos ese objeto 
		return pop_sample[min_index]	
	

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

	def crossover_pmx(self, p_1, p_2):
		'''

		'''
		#Generar los puntos de corte 
		all_cut_points = np.arange(self.n_queens)
		cut_points = sorted(np.random.choice(all_cut_points,2,replace=False))
		#Este se puede dar el caso de solo se intercambien dos elementos 
		cp_1 , cp_2= cut_points[0],cut_points[1]



		
		#Generar la relacion de mapeo de las cadenas
		p_1.chromosome[cp_1:cp_2]
		p_2.chromosome[cp_1:cp_2]
		#Necesitamos almacenar los intercambios realizados en una relacion de mapeo 
		#Para esta primera versión elegimos un diccionario por que el acceso a los elementos es O(1)
		
		mapping_relation ={}
		for i in range(len(p_2.chromosome[cp_1:cp_2])):
			mapping_relation[p_1.chromosome[cp_1:cp_2][i]] = p_2.chromosome[cp_1:cp_2][i] 
			mapping_relation[p_2.chromosome[cp_1:cp_2][i]] = p_1.chromosome[cp_1:cp_2][i] 	

		#Generamos dos copias de los cromosomas de los padres 
		primitive_offspring_1_chromosome = p_1.chromosome.copy()
		primitive_offspring_2_chromosome = p_2.chromosome.copy()
		
		#Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte 
		primitive_offspring_1_chromosome[cp_1:cp_2] = p_2.chromosome[cp_1:cp_2]
		primitive_offspring_2_chromosome[cp_1:cp_2] = p_1.chromosome[cp_1:cp_2]

		#Ahora hay que arreglar los chromosomas 

		#ESTA IMPLEMENTACION FUE PUESTA EN PAUSA,
		#values_1, count_1 = np.unique(primitive_offspring_1_chromosome,return_counts=True)
		#values_2, count_2 = np.unique(primitive_offspring_2_chromosome,return_counts=True)

		#Obtenemos las ocurrencias de cada numero en los cromosomas primitivos :

	
		ocurences_1 = np.array([0 for i in range(self.n_queens)])
		ocurences_2 = np.array([0 for i in range(self.n_queens)])

		#A cada numero en la permutacion le corresponde un índice en el arreglo de ocurrencias
		#Cada vez que ocurra el numero "x" el índice x del arreglo de ocurrencias incrementa en 1 
		#Si un numero aparace 2 veces entonces el ocurrences[x] =2 
		for i in range(self.n_queens):
			ocurences_1[primitive_offspring_1_chromosome[i]] +=1
			ocurences_2[primitive_offspring_2_chromosome[i]] +=1 						

		
		#Sustituir los numeros con ocurrencia 2 usando la relacion de mapeo 
		# >>>> En proceso
		keys = mapping_relation.keys()

		print("Crhomosomas de los padres")
		print("Padre 1 : ")
		print(str(p_1.chromosome))
		print("Padre 2 :")
		print(str(p_2.chromosome))
		print("Puntos de corte  P1 :"+str(cp_1)+" P2 :"+str(cp_2))
		print("Subcadenas consideradas : ")
		print("Subchain 1 : ")
		print(str(p_1.chromosome[cp_1:cp_2]))
		print("Subchain 2 : ")
		print(str(p_2.chromosome[cp_1:cp_2]))
		print("Relacion de intercambio : "+str(mapping_relation))
		print("Chromosomas primitivos")
		print("Primitivo 1")
		print(str(primitive_offspring_1_chromosome))
		print(ocurences_1)


		for i in range(self.n_queens):
			if(primitive_offspring_1_chromosome[i] in mapping_relation):
				if(ocurences_1[primitive_offspring_1_chromosome[i]] == 2):
					print("Caso en el que va a cambiar a un elemento repetido por primera vez")
					primitive_offspring_1_chromosome[i] = mapping_relation[primitive_offspring_1_chromosome[i]]
					ocurences_1[primitive_offspring_1_chromosome[i]]=-1
				elif(ocurences_1[primitive_offspring_1_chromosome[i]]==-1):
					print("Caso en el que no cambia un elemento que ya fue cambiado")
					continue
				else:
					primitive_offspring_1_chromosome[i] = mapping_relation[primitive_offspring_1_chromosome[i]]
				
	

		# for i in range(self.n_queens):
		# 	pass
		# 	#ocurences_1[primitive_offspring_1_chromosome[i]] +=1
		# 	if(ocurences_1[primitive_offspring_1_chromosome[i]] == 2): #Entonces el elemento aparece mas de una vez
		# 		print(primitive_offspring_1_chromosome[i])
		# 		ocurences_1[primitive_offspring_1_chromosome[i]]=1
		# 		primitive_offspring_1_chromosome[i] = mapping_relation[primitive_offspring_1_chromosome[i]]
				#Ahora sustituimos ese elemento usando la relacion de mapeo
				# print("Elemento :"+ str(primitive_offspring_1_chromosome[i]))
				# print("Ocurrencias antes de restar:"+str(ocurences_1[primitive_offspring_1_chromosome[i]]))

				#primitive_offspring_1_chromosome[i] = mapping_relation[primitive_offspring_1_chromosome[i]]
				#ocurences_1[primitive_offspring_1_chromosome[i]] =ocurences_1[primitive_offspring_1_chromosome[i]]-1
				# ocurences_1[primitive_offspring_1_chromosome[i]]  =1
				# print("Ocurrencias despues de restar:"+str(ocurences_1[primitive_offspring_1_chromosome[i]]))
			# print("Se tiene que cambiar "+ str(primitive_offspring_1_chromosome[i])+" ?")
			# print("Ocurrencias :"+str(dic_ocur_1[primitive_offspring_1_chromosome[i]]))
			# print(dic_ocur_1[primitive_offspring_1_chromosome[i]]==2)


			# if dic_ocur_1[primitive_offspring_1_chromosome[i]] == 2: #Esto quiere decir que el elemento está duplicado 
			# 	#Ahora sustituimos ese elemento usando la relacion de mapeo 
			# 	primitive_offspring_1_chromosome[i] = mapping_relation[primitive_offspring_1_chromosome[i]]
			# 	print("Antes de restar :"+str(dic_ocur_1[primitive_offspring_1_chromosome[i]]))
			# 	dic_ocur_1[primitive_offspring_1_chromosome[i]] -=1
			# 	print("Despues de restar :"+str(dic_ocur_1[primitive_offspring_1_chromosome[i]])) 
			
			# if dic_ocur_2[primitive_offspring_2_chromosome[i]] == 2: #Esto quiere decir que el elemento está duplicado 
			# 	#Ahora sustituimos ese elemento usando la relacion de mapeo 
			# 	primitive_offspring_2_chromosome[i] = mapping_relation[primitive_offspring_2_chromosome[i]]
			# 	dic_ocur_2[primitive_offspring_2_chromosome[i]] = dic_ocur_2[primitive_offspring_2_chromosome[i]]-1 

		#Debug 
		
		#print(values_1)
		#print(count_1)
		# print("Primitivo 2")
		# print(str(primitive_offspring_2_chromosome))
		# print(ocurences_2)


		print("Cadenas arregladas ")
		print("Final 1")
		print(str(primitive_offspring_1_chromosome))
		#print("Final 2")
		#print(str(primitive_offspring_2_chromosome))



		#Generar las cadenas primitivas 
		
		#Arreglar las cadenas primitivas 
		
	def crossover_ipmx(self, p_1, p_2):
		'''
		Implementacion del Improved Partially Mapped Crossover 

		'''

		#Generar los puntos de corte 
		all_cut_points = np.arange(self.n_queens)
		cut_points = sorted(np.random.choice(all_cut_points,2,replace=False))
		#Este se puede dar el caso de solo se intercambien dos elementos 
		cp_1 , cp_2= cut_points[0],cut_points[1]

		#Generamos dos copias de los cromosomas de los padres 
		primitive_offspring_1_chromosome = p_1.chromosome.copy()
		primitive_offspring_2_chromosome = p_2.chromosome.copy()
		
		#Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte 
		primitive_offspring_1_chromosome[cp_1:cp_2] = p_2.chromosome[cp_1:cp_2]
		primitive_offspring_2_chromosome[cp_1:cp_2] = p_1.chromosome[cp_1:cp_2]

		#Se genera la lista de intercambio ExchangeList -- Equivalente a la mapping relationship del PMX 
		
		#DEBUG 
		print("Crhomosomas de los padres")
		print("Padre 1 : ")
		print(str(p_1.chromosome))
		print("Padre 2 :")
		print(str(p_2.chromosome))
		print("Puntos de corte  P1 :"+str(cp_1)+" P2 :"+str(cp_2))
		print("Subcadenas consideradas : ")
		print("Subchain 1 : ")
		print(str(p_1.chromosome[cp_1:cp_2]))
		print("Subchain 2 : ")
		print(str(p_2.chromosome[cp_1:cp_2]))	



		mapping_relation = []
		for i in range(len(p_2.chromosome[cp_1:cp_2])):
			mapping_relation.append([p_1.chromosome[cp_1:cp_2][i],p_2.chromosome[cp_1:cp_2][i],1])
			
		exchange_list = np.array(mapping_relation)
		print("Tabla de intercambio")
		print(exchange_list )
		#Guide List 
		guide_list = [0 for i in range(self.n_queens)]
		
		# Para llenar la guide list tenemos que considerar como indices los numeros de la primer columna de la exchange_list 
		# Para llenar los valores de la guide list tenemos 

		#Aqui hay un prolema, la implementacion del paper se basa en que el 0 no esta en la permutacion 
		#Por lo que consideramos -1 como el 0 
		for tuple in exchange_list:
			guide_list[tuple[0]] = tuple[1] 
		print("Lista de Guia")
		print(guide_list)


		#Generamos las listas L1 y L2 
		l1 = [0 for i in range(self.n_queens)]
		l2 = [0 for i in range(self.n_queens)]

		#Ahora vamos a llenar l1 considerando la primer columna de la exchange list y a l2 con las segunda columna
		for e in exchange_list[:,0]: 
			l1[e] =1 

		for e in exchange_list[:,1]: 
			l2[e] =1 

		print("Listas auxiliares")
		print("L1 :" + str(l1))
		print("L2 :" + str(l2))




		#Generamos la suma de ambas listas 
		l1_plus_l2 = [x+y for x,y in zip(l1,l2)]

		print("Suma de las listas")
		print("L12:" + str(l1_plus_l2))

		#Cada entrada de l1_plus_l2 que sea igual a 2  indica que es un nodo intermedio, en ese caso actualizamos la lista de intercambio
		#Actualizamos la lista de intercambio
		for tuple in exchange_list:
			if l1_plus_l2[tuple[0]] == 2:
				tuple[2] = 0  
		print("Tabla de intercambio actualizada")
		print(exchange_list)

		#Y luego actualizamos la lista guia en base a la nueva lista de intercambio 
		#Updated Guide List 
		for tuple in exchange_list:
			if(tuple[2] == 0):
				guide_list[tuple[0]] = 0  
		
		print("Lista de Guia Actualizada")
		print(guide_list)


		#Debug 
		
		
		#Columna 1 = exchange_list[:0]
		#Columna 2 = exchange_list[:1]
		#Columna 3 = exchange_list[:2] #Esta es la que nos dara innformacion sobre si existen o no un camino directo  
			
			
		
		
		
		pass	 

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
			parents = np.random.choice(self.current_pop, 2)
			s_1, s_2 = parents[0],parents[1] 
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
				tournament_selected = self.selection_tournament()
			#Crossover  
			#roulette_offspring = self.crossover_pop(roulette_selected)
			tournament_offspring = self.crossover_pop_2(tournament_selected)
			#Elitism Selection  ---> El elitismo debe ser entre padres e hijos 
			#elite = self.selection_elitism() 
			#Union of the Selected individuals
			#offspring = np.concatenate((roulette_offspring,elite),axis=0)
			#offspring = np.concatenate((tournament_offspring,elite),axis=0) 
			offspring = self.generational_replacement_elitism_mu_plus_lambda(tournament_offspring)
			#Generational Replacement 
			self.current_pop = offspring
			#Mutate
			self.mutation_simple()
			total_iterations = total_iterations+1
				

		

		# while(self.get_the_best().fitness != self.optimal):
		# 	#Roullete Selection 
		# 	#roulette_selected = self.selection_rl(
		# 	[ind.evaluate_min() for ind in self.current_pop]
		# 	tournament_selected = self.selection_tournament()
		# 	#Crossover  
		# 	#roulette_offspring = self.crossover_pop(roulette_selected)
		# 	tournament_offspring = self.crossover_pop(tournament_selected)
		# 	#Elitism Selection  ---> El elitismo debe ser entre padres e hijos 
		# 	elite = self.selection_elitism() 
		# 	#Union of the Selected individuals
		# 	#offspring = np.concatenate((roulette_offspring,elite),axis=0)
		# 	offspring = np.concatenate((tournament_offspring,elite),axis=0) 
		# 	#Generational Replacement 
		# 	self.current_pop = offspring
		# 	#Mutate
		# 	self.mutation_simple()
		# 	total_iterations = total_iterations+1 


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

def plot_queens(solution):
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

	#El 3 es el tamanio del torneo 
	ga = GeneticAlg(number_q,popultation_size,.8,3,prob_cross,prob_mut,time)
	
	#rep_iter(10,ga)
	time, iters = ga.execution()
	print("Tiempo de ejecucion "+str(time))
	print("Iteraciones usadas "+ str(iters))
	print(ga.get_the_best())
	#plot_queens(ga.get_the_best().chromosome)
	
	
	# print("---------------")

	listaSols = sorted(ga.current_pop, key = lambda solution : solution.fitness)
	child1 = listaSols[0]
	child2 = listaSols[-1]
	#ga.crossover_pmx(child1, child2)
	#ga.crossover_ipmx(child1, child2)
	
	# for sol in listaSols: 
	# 	print(sol)

	#aplot_queens(ga.get_the_best())
	#print(ga.get_best_among(ga.current_pop))
	#print(ga.get_the_best().)			
	#ga.visualize_board(ga.get_the_best().chromosome)
	

	



	




