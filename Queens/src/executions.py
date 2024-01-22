import ga 
import sys
import time  
import matplotlib.pyplot as plt
import pandas as pd
import os 
import csv
import numpy as np 
class Metrics:

	def __init__(self):
		self.genetic_algo = None
	
	def get_params(self):
		
		permutation_size = int(sys.argv[1])
		population_size = int(sys.argv[2])
		crossover_probability = float(sys.argv[3])
		mutation_probability = float(sys.argv[4])
		max_generations = int(sys.argv[5])
		max_time = int(sys.argv[6])
		selection_operator = int(sys.argv[7])
		tournament_size = int(sys.argv[8])
		crossover_operator = int(sys.argv[9])
		mutation_operator = int(sys.argv[10])
		generational_replacement_operator = int(sys.argv[11])    

		#Inicializamos el algoritmo 
		self.genetic_algo = ga.GeneticAlg(permutation_size,population_size,crossover_probability,mutation_probability,max_generations,max_time,selection_operator,tournament_size,crossover_operator,mutation_operator,generational_replacement_operator)
	
	def execute_algorithm(self):
		self.genetic_algo.execution()
	
	#Una funcion que ejecute el operador de cruza con dos cromosomas random y regrese el tiempo 
	#que tardó realizando la cruza. 
	# Tenemos que obtener el self.crossover_operator.cross y pasarle dos 
	def get_crossover_time(self):
		#Se tiene que realizar primero el get_params 
		#Inicializamos poblacion para obtener cromosomas randoms 
		individuals = np.random.choice(self.genetic_algo.current_pop,2)
		#Se realiza la cruza 
		print(">>>>")
		time_start = time.time()
		offs_1,offs_2 = self.genetic_algo.crossover_operator.cross(individuals[0],individuals[1])
		time_end = time.time()
		print("offspring 1 "+str(offs_1))
		print("offspring 2 "+str(offs_2))
		print(">>>>")
		total_time = time_end-time_start
		print(total_time)
		return total_time
	# parents = np.random.choice(population,2)
    #         if(rnd.random() < self.cross_prob):    
    #             offspring_1, offspring_2 = self.cross(parents[0],parents[1])
    #             offspring.append(offspring_1)
    #             offspring.append(offspring_2)
	
	def register_time(self,iterations):
		#De uno a n vamos incrementar las reinas ? 
		#Buscamos realizar 30 veces el get crossover time y registrarlo en el .csv 
		self.genetic_algo.init_population()
		for i in range(8,21):
			data = []
			data.append(str(i))
			j = 0
			self.genetic_algo.permutation_size = i 
			for i in range(iterations):
				data.append(self.get_crossover_time())
			##Ahora hay que registrarlo en el .csv
			file = get_path_for_output("output/crossoverOp/PMXTimes.txt") 
			writte_txt_data(file,data)
			j+=1
			

#Este hay que moverlo a uno de puras metricas 
def boxplot(sample_1):
		fig, ax = plt.subplots()
		bp = ax.boxplot([sample_1], showfliers=False)
		plt.show()

#Funcion para encontrar la ruta de un archivo
def get_path_for_output(output_file_name):
		current_address = os.path.dirname(os.path.abspath(__file__))
		current_address = os.path.dirname(current_address)
		output_address = os.path.join(current_address,output_file_name)
		return output_address

def writte_txt_data(file_path, data):

	with open(file_path, 'a') as file:
		line = f"{data[0]},{','.join(map(str,data[1:]))}\n"

		file.write(line)

def get_data_from_txt(file_path):
	
	file_path = get_path_for_output(file_path)
	results = []
	with open(file_path, 'r') as file : 
		for line in file:
			data = line.strip().split(',')
			value = int(data[0])
			times = list(map(float,data[1:])) 
			results.append([value]+times)
	
	return results


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

    #get_params_execute()
	# Condicion de termino : Alcanza optimo o maximo de generaciones 
	metrics = Metrics()
	metrics.get_params()
	#metrics.register_time(5)
	# cross_time = metrics.get_crossover_time()
	print(get_data_from_txt("output/crossoverOp/PMXTimes.txt"))
	
	#metrics.execute_algorithm()
	#number_q,popultation_size,prob_cross,prob_mut,time = int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])
	
	
	#El 3 es el tamanio del torneo 
	#ga = GeneticAlg(number_q,popultation_size,.8,3,prob_cross,prob_mut,time)
	
	#ga.execution()
	#print(ga.get_the_best(ga.current_pop))
	#ga.get_the_best().output_plot()
	#plot_queens(ga.get_the_best(),outputName) 

	








