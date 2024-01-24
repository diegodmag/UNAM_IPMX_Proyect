import ga 
import sys
import time  
import matplotlib.pyplot as plt
import random
import os 
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
		a,x,y,z,w = self.genetic_algo.execution()
		print(a)
		print(x)
		print(y)
		print(z)
		print(w)
	
	#Una funcion que ejecute el operador de cruza con dos cromosomas random y regrese el tiempo 
	#que tardó realizando la cruza. 
	# Tenemos que obtener el self.crossover_operator.cross y pasarle dos 
	def get_crossover_time(self):
		#Se tiene que realizar primero el get_params 
		#Inicializamos poblacion para obtener cromosomas randoms 
		individuals = np.random.choice(self.genetic_algo.current_pop,2)
		#Se realiza la cruza 
		time_start = time.time()
		offs_1,offs_2 = self.genetic_algo.crossover_operator.cross(individuals[0],individuals[1])
		time_end = time.time()
		total_time = time_end-time_start
		return total_time
	
	def register_crossover_time(self,min_per_size,max_per_size,iterations,file_name):
		'''
		Esta funcion registra en 
		
		'''
		
		#De uno a n vamos incrementar las reinas ? 
		#Buscamos realizar 30 veces el get crossover time y registrarlo en el .csv 
		file = "output/crossoverdata/"+str(file_name)
		self.genetic_algo.init_population()
		for i in range(min_per_size,max_per_size+1):
			data = []
			data.append(str(i))
			j = 0
			self.genetic_algo.permutation_size = i
			self.genetic_algo.init_population() 
			for i in range(iterations):
				data.append(self.get_crossover_time())
			##Ahora hay que registrarlo en el .csv
			file = get_path_for_file(file) 
			writte_txt_data(file,data)
			j+=1
			
	def register_ga_individual_execution(self,file_name):
		'''
		Esta funcion ejecuta el algoritmo evolutivo y escribe los datos para la ejecucion individual 
		a lo largo de las generaciones 
		'''
		generations,best_sons,avg_offspring,best_all,avg_fitness = self.genetic_algo.execution()
		file = "output/gaindividualexecutions/"+str(file_name)
		#Hay que hacer el formaro de cada linea 
		total_data = []
		for i in range(len(generations)):
			line = [generations[i], best_sons[i], avg_offspring[i],best_all[i], avg_fitness[i]]
			total_data.append(line)
		
		for line in total_data:
			writte_txt_data(file,line)
		# writte_txt_data(file,generations,best_sons)
		# writte_txt_data(file,best_sons)
		# writte_txt_data(file,avg_offspring)
		# writte_txt_data(file,best_all)
		# writte_txt_data(file,avg_fitness)

		#Estaria bien guardar informacion del Operador de Cruza, El operador de Seleccion, Operador Mutacion, Operador de seleccion
		
#Este hay que moverlo a uno de puras metricas 
def boxplot(sample_1):
		fig, ax = plt.subplots()
		bp = ax.boxplot([sample_1], showfliers=False)
		plt.show()

#Funcion para encontrar la ruta de un archivo
def get_path_for_file(output_file_name):
		'''
		Obtiene la ruta de un archivo a partir de un nombre
		'''
		current_address = os.path.dirname(os.path.abspath(__file__))
		current_address = os.path.dirname(current_address)
		output_address = os.path.join(current_address,output_file_name)
		return output_address

def writte_txt_data(file_path, data):
	with open(file_path, 'a') as file:
		line = f"{data[0]},{','.join(map(str,data[1:]))}\n"
		file.write(line)


def get_data_from_txt(file_path,output_dir):
	'''
	Obtiene la data de un archivo .txt donde cada linea está separada solo por ','
	'''
	file_path = "output/"+str(output_dir)+"/"+str(file_path)
	file_path = get_path_for_file(file_path)
	results = []
	with open(file_path, 'r') as file : 
		for line in file:
			data = line.strip().split(',')
			value = int(data[0])
			times = list(map(float,data[1:])) 
			results.append([value]+times)
	
	return results

def get_data_from_txt_individuals(file_path,output_dir):
	'''
	Obtiene los datos de una ejecucion individual en formato [gen, best_individual_offspring, avg_ofspring, best, avg_fitness]
	'''
	file_path = "output/"+str(output_dir)+"/"+str(file_path)
	file_path = get_path_for_file(file_path)

	gens_data = []
	best_ind_offspring_data = []
	avg_offspring_data = []
	best_ind_data=[]
	avg_fitness_data = []
	with open(file_path,'r') as file : 
		for line in file : 
			data = line.strip().split(',')
			# gen = int(data[0])
			# best_ind_offspring = int(data[1])
			# avg_offspring = float(data[2])
			# best_ind = int(data[3])
			# avg_fitness = float(data[0])
			gens_data.append(int(data[0]))
			best_ind_offspring_data.append(int(data[1]))
			avg_offspring_data.append(float(data[2]))
			best_ind_data.append(float(data[3]))
			avg_fitness_data.append(float(data[4]))

	return [gens_data,best_ind_offspring_data,avg_offspring_data,best_ind_data,avg_fitness_data] 
	

def generate_avg_data(data):
	'''
	'''
	#Considerando que nuestra data son arreglos bidimencionales 
	data_1 = []
	avg_data = []

	for results in data : 
		#Aqui ya vamos por linea 
		#Primero guardamos el valor el cual siempre esta en el primer indice 
		data_1.append(results[0])
		avg = np.array(results[1:])
		avg = np.mean(avg)
		avg_data.append(avg)

	return data_1,avg_data

def gen_basic_graph(data_1,data_2,file_path):
	title = file_path
	file_path = "output/crossovervisuals/"+str(file_path)
	file_path = get_path_for_file(file_path)
	plt.figure(figsize=(10, 8))
	plt.plot(data_1,data_2)
	plt.xlabel('Tamaño permutacion')
	plt.ylabel('Tiempo promedio')
	#Srive para mostrar todos los puntos 
	plt.xticks(data_1)
	plt.title('Tiempo promedio para realizar la cruza de '+str(title))
	plt.savefig(file_path, dpi=300)

def gen_vs_graph(data_1,data_2,file_path,data_name_1, data_name_2):
	title = file_path
	file_path = "output/crossovervisuals/"+str(file_path)
	file_path = get_path_for_file(file_path)
	plt.figure(figsize=(10, 8))
	plt.plot(data_1[0],data_1[1],label=data_name_1,marker='o', linestyle='--', color='blue')
	plt.plot(data_2[0],data_2[1],label=data_name_2,marker='s', linestyle='--', color='orange')
	plt.xlabel('Tamaño permutacion')
	plt.ylabel('Tiempo promedio')
	#Srive para mostrar todos los puntos 
	plt.xticks(data_1[0])
	plt.legend()
	plt.title('Tiempo promedio para realizar la cruza de '+str(title))
	plt.savefig(file_path, dpi=300)

def get_ind_exe_graph(data,data_names,colors,file_path):
	'''
	La grafica de ejecuciones individuales, se considera como data[0] las generaciones 
	'''
	
	title = file_path
	file_path = "output/gaindividualexecutionsvisuals/"+str(file_path)
	file_path = get_path_for_file(file_path)
	plt.figure(figsize=(10, 8))
	for i in range(1,len(data)):
		#Generamos un color random 
		random_color = (random.random(), random.random(), random.random())
		plt.plot(data[0],data[i],label=data_names[i],linestyle='--', color=colors[i])

	plt.xlabel('Generacion')
	plt.ylabel('Fitness')
	#Srive para mostrar todos los puntos 
	#plt.xticks(data[0])
	plt.legend()
	plt.title('Evolucion del valor fitness de una ejecucion individual'+str(title))
	plt.savefig(file_path, dpi=300)

#plt.savefig(self.get_path_for_file(str('Board'+str(len(self.chromosome)))))

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

def graph_generation(file_name):
	raw_data = get_data_from_txt(file_name+".txt")
	d_1, d_2 = generate_avg_data(raw_data)
	file_name_generic = file_name
	gen_basic_graph(d_1,d_2,file_name_generic)

def graph_vs_generation(file_name1,file_name2):
	raw_data_1 = get_data_from_txt(file_name1+".txt")
	raw_data_2 = get_data_from_txt(file_name2+".txt")
	d_1_1, d_1_2 = generate_avg_data(raw_data_1)
	d_2_1, d_2_2 = generate_avg_data(raw_data_2)
	file_name_generic = file_name1+"vs"+file_name2
	gen_vs_graph([d_1_1,d_1_2], [d_2_1,d_2_2],file_name_generic,file_name1,file_name2)
	#gen_basic_graph(d_1,d_2,file_name_generic)

if __name__ == '__main__':
 
	metrics = Metrics()
	metrics.get_params()
	#metrics.execute_algorithm()
	file_name_txt = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"IndExecution.txt"
	file_name_graph = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"IndExecution"
	metrics.register_ga_individual_execution(file_name_txt)

	data = get_data_from_txt_individuals(file_name_txt,"gaindividualexecutions")
	colors = ['black', 'red', 'blue', 'green', 'purple']
	get_ind_exe_graph(data,["Generations", "Best of Offspring", "Avg Offspring", "Best", "Avg Fitness"],colors, file_name_graph)
	#print(data,["Generations", "Best of Offspring", "Avg Offspring", "Best", "Avg Fitness"],file_name_graph)
	#Se registran 
	#metrics.register_crossover_time(15, 35, 30,file_name+".txt")
	#graph_generation(file_name)
	#graph_vs_generation("IMPXTimes","PMXTimes")







