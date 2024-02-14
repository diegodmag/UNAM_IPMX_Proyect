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
	
	def simple_execution(self):
		
		print(self.genetic_algo.simple_execution())

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
		Esta funcion registra en un archivo los tiempos que toma un algoritmo por tamanio de permutacion 
		
		'''
		
		#De uno a n vamos incrementar las reinas ? 
		#Buscamos realizar 30 veces el get crossover time y registrarlo en el .csv 
		file = "output/crossoverdata/"+str(file_name)
		#self.genetic_algo.init_population()
		#Hay que guardar en la primer file los parametros de la ejecucion 
		#Que me interesa saber de los operadores de cruza ?? 
			
		#Aqui se mete la informacion basica 
		for i in range(min_per_size,max_per_size+1):
			#CMABIAMOS LA SEMILLA 
			new_sed = int(time.time()) 
			np.random.seed(new_sed)
			self.genetic_algo.init_population()
			#La poblacion se reinicia en cada iteracion 
			data = []
			data.append(str(i))
			data.append(new_sed)
			j = 0
			self.genetic_algo.permutation_size = i
			self.genetic_algo.init_population() 
			for i in range(iterations):
				data.append(self.get_crossover_time())
			##Ahora hay que registrarlo en el .csv
			#Ahora hay que registrar la semilla 
			file = get_path_for_file(file) 
			writte_txt_data(file,data)
			j+=1
	

	def register_ga_avg_execution(self,file_name,iterations):
		
		file = "output/executionsforavg/"+str(file_name)
		file_seeds = "output/executionsforavgseedsperiterations/"+str(file_name)
		
		#Por cada iteracion se guardan todos los datos obtenidos en cada generacion

		total_best_sons_data = []
		total_avg_offspring_data = []
		total_best_all_data=[]
		total_avg_fitness_data=[]
		seeds_per_iteration = []
		#Total de iteraciones 
		#Por cada iteracion hay que guardar los datos obtenidos 
		for i in range(iterations):
			#En cada iteracion se cambia la semilla 
			new_sed = int(time.time()) 
			np.random.seed(new_sed)
			# Realizamos una ejecucion 
			generations, best_sons_ind_data, avg_offspring_ind_data, best_all_ind_data,avg_fitness_ind_data = self.genetic_algo.execution()
			total_best_sons_data.append(best_sons_ind_data)
			total_avg_offspring_data.append(avg_offspring_ind_data)
			total_best_all_data.append(best_all_ind_data)
			total_avg_fitness_data.append(avg_fitness_ind_data)
			seeds_per_iteration.append(new_sed)

		#Otenemos los arreglos de promedio por cada generacion  
		generations = [i for i in range(1, self.genetic_algo.max_generations+1)]
		mean_generation_best_sons_data = np.mean(total_best_sons_data,axis=0)
		mean_generation_avg_offspring_data = np.mean(total_avg_offspring_data,axis=0)
		mean_generation_best_all_data = np.mean(total_best_all_data,axis=0)
		mean_generation_avg_fitness_data = np.mean(total_avg_fitness_data,axis=0)

		#Tenemos que generar los datos ordenados por generacion 

		#Ahora hay que REGISTRARLO
		#Generamos la primer linea la cual lleva informacion de las x iteraciones 
		selection_operator_line = str(self.genetic_algo.selection_operator.__class__.__name__)
		cross_operator_line = str(self.genetic_algo.crossover_operator.__class__.__name__)
		mutation_operator_line = str(self.genetic_algo.mutation_operator.__class__.__name__)
		replacement_operator_line = str(self.genetic_algo.generation_replacement.__class__.__name__)
		
		#Tambien guardamos cuantas iteraciones se realizaron  
		params_line=[self.genetic_algo.permutation_size, self.genetic_algo.pop_size, self.genetic_algo.cross_prob, self.genetic_algo.mut_prob,self.genetic_algo.max_generations,self.genetic_algo.max_time,selection_operator_line, self.genetic_algo.tournament_size,cross_operator_line, mutation_operator_line,replacement_operator_line,iterations ]   
		#Escribimos la linea de parametros en el documento 
		writte_txt_data(file, params_line)
		#Guardamos las semillas 
		writte_txt_data(file_seeds, seeds_per_iteration)
		#Procedemos a escribir el promedio de datos POR GENERACION 
		avg_data = []
		for i in range(self.genetic_algo.max_generations):
			#Por cada generacion, de con i in 1...n  
			# 	El promedio del mejor hijo (es decir obtenido de la cruza) de las n iteracion en la i-esima generacion 
			# 	El promedio del los n-esimso promedios del offpring obtenido en la i-esima generacion  
			# 	El promedio del mejor individuo (despues de realizar el reemplazo generacional) de las n iteracion en la i-esima generacion
			# 	El promedio del los n-esimso promedios de la poblacion (despues de realizar el reemplazo generacional) obtenido en la i-esima generacion 
			line= [generations[i],mean_generation_best_sons_data[i],mean_generation_avg_offspring_data[i],mean_generation_best_all_data[i],mean_generation_avg_fitness_data[i]]
			avg_data.append(line)
		#Finalmente procedmos a registrar los datos de las ejecuciones promedio 
		for line in avg_data:
			writte_txt_data(file,line)


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
		

		#La primer linea deben ser los parametros 
		selection_operator_line = str(self.genetic_algo.selection_operator.__class__.__name__)
		cross_operator_line = str(self.genetic_algo.crossover_operator.__class__.__name__)
		mutation_operator_line = str(self.genetic_algo.mutation_operator.__class__.__name__)
		replacement_operator_line = str(self.genetic_algo.generation_replacement.__class__.__name__)
		params_line=[self.genetic_algo.permutation_size, self.genetic_algo.pop_size, self.genetic_algo.cross_prob, self.genetic_algo.mut_prob,self.genetic_algo.max_generations,self.genetic_algo.max_time,selection_operator_line, self.genetic_algo.tournament_size,cross_operator_line, mutation_operator_line,replacement_operator_line ]
		writte_txt_data(file,params_line)
		for line in total_data:
			writte_txt_data(file,line)
		
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
		#Hay que leer la primer linea la cual lleva los parametros del algoritmo 

		#Leemos la primer linea que es la que carga info sobre la ejecucion 

		for line in file:
			data = line.strip().split(',')
			value = int(data[0])
			times = list(map(float,data[2:])) #Aqui debemos ignorar la ultima entrada por que es la semilla 
			#times = list(map(float,data[1:-1])) 
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
		#Hay que leer la primera linea como los parametro

		#Leemos la primer linea de forma particular 
		params_Line = file.readline().strip().split(',')
		
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

	return [params_Line,gens_data,best_ind_offspring_data,avg_offspring_data,best_ind_data,avg_fitness_data] 
	
def get_data_from_txt_mean_evolution(file_path,output_dir):
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
		#Hay que leer la primera linea como los parametro

		#Leemos la primer linea de forma particular 
		params_Line = file.readline().strip().split(',')
		
		for line in file : 
			data = line.strip().split(',')
			# gen = int(data[0])
			# best_ind_offspring = int(data[1])
			# avg_offspring = float(data[2])
			# best_ind = int(data[3])
			# avg_fitness = float(data[0])
			gens_data.append(int(data[0]))
			best_ind_offspring_data.append(float(data[1]))
			avg_offspring_data.append(float(data[2]))
			best_ind_data.append(float(data[3]))
			avg_fitness_data.append(float(data[4]))

	return [params_Line,gens_data,best_ind_offspring_data,avg_offspring_data,best_ind_data,avg_fitness_data] 

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

def gen_vs_graph(data_1,data_2,data_3,file_path,data_name_1, data_name_2,data_name_3):
	title = file_path
	file_path = "output/crossovervisuals/"+str(file_path)
	file_path = get_path_for_file(file_path)
	plt.figure(figsize=(10, 8))
	plt.plot(data_1[0],data_1[1],label=data_name_1,marker='o', linestyle='--', color='blue')
	plt.plot(data_2[0],data_2[1],label=data_name_2,marker='s', linestyle='--', color='orange')
	plt.plot(data_3[0],data_3[1],label=data_name_3,marker='D', linestyle='--', color='green')
	plt.xlabel('Tamaño permutacion')
	plt.ylabel('Tiempo promedio')
	#Srive para mostrar todos los puntos 
	plt.xticks(data_1[0])
	plt.legend()
	plt.title('Tiempo promedio para realizar la cruza de '+str(title))
	plt.savefig(file_path, dpi=300)

def get_ind_exe_graph(info,data,data_names,colors,file_path):
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
		plt.annotate(f'({data[0][-1]},{data[i][-1]})', xy = (data[0][-1],data[i][-1]), xytext=(data[0][-1]-i*2,data[i][-1]+(i+.5)),arrowprops=dict(color=colors[i], arrowstyle='->') )
	
	# plt.annotate(f'({ultimo_x1:.2f}, {ultimo_y1:.2f})', xy=(ultimo_x1, ultimo_y1), xytext=(ultimo_x1, ultimo_y1 + 0.5),
    #          arrowprops=dict(facecolor='black', arrowstyle='->'))
	
	plt.xlabel('Generacion')
	plt.ylabel('Fitness')
	#Srive para mostrar todos los puntos 
	#plt.xticks(data[0])
	plt.legend()

	title_font = {
    'fontsize': 6,          # Tamaño de la fuente
    'fontweight': 'bold',    # Peso de la fuente
    'color': 'blue',         # Color del texto
    'verticalalignment': 'bottom',  # Alineación vertical
	}

	#info = "Tamaño permutación : {}\nTamaño población : {}\nProb.Crossover : {}\nProb.Mutación : {}\nGeneraciones Max : {}\nTiempo Max {}\nSelección : {}\k-Torneo : {}\nCrossover : {}\nMutación : {}\nReemplazo : {}".format(info[0],info[1],info[2],info[3],info[4],info[5],info[6],info[7],info[8],info[9], info[10])
	info_1 = "Tamaño permutación : {}\nTamaño población : {}\nProb.Crossover : {}\nProb.Mutación : {}\nGeneraciones Max : {}".format(info[0],info[1],info[2],info[3],info[4])
	info_2 = "Tiempo Max {}\nSelección : {}\k-Torneo : {}\nCrossover : {}\nMutación : {}\nReemplazo : {}".format(info[5],info[6],info[7],info[8],info[9], info[10])
	#"Tiempo Max {}\nSelección : {}\k-Torneo : {}\nCrossover : {}\nMutación : {}\nReemplazo : {}".format(info[5],info[6],info[7],info[8],info[9], info[10])
	
	plt.title('Evolucion del valor fitness de una ejecucion individual '+str(title),y=1.2)
	plt.text(x=.15, y=.95, s=info_1, fontsize=9, ha='left', va='center', transform=plt.gcf().transFigure)
	plt.text(x=.35, y=.95, s=info_2, fontsize=9, ha='left', va='center', transform=plt.gcf().transFigure)
	plt.savefig(file_path,bbox_inches='tight')

def get_avg_exe_graph(info,data,data_names,colors,file_path):
	'''
	La grafica de ejecuciones individuales, se considera como data[0] las generaciones 
	'''
	
	title = file_path
	file_path = "output/executionsforavggraphics/"+str(file_path)
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

	title_font = {
    'fontsize': 6,          # Tamaño de la fuente
    'fontweight': 'bold',    # Peso de la fuente
    'color': 'blue',         # Color del texto
    'verticalalignment': 'bottom',  # Alineación vertical
	}

	#info = "Tamaño permutación : {}\nTamaño población : {}\nProb.Crossover : {}\nProb.Mutación : {}\nGeneraciones Max : {}\nTiempo Max {}\nSelección : {}\k-Torneo : {}\nCrossover : {}\nMutación : {}\nReemplazo : {}".format(info[0],info[1],info[2],info[3],info[4],info[5],info[6],info[7],info[8],info[9], info[10])
	info_1 = "Tamaño permutación : {}\nTamaño población : {}\nProb.Crossover : {}\nProb.Mutación : {}\nGeneraciones Max : {}".format(info[0],info[1],info[2],info[3],info[4])
	info_2 = "Tiempo Max {}\nSelección : {}\k-Torneo : {}\nCrossover : {}\nMutación : {}\nReemplazo : {}".format(info[5],info[6],info[7],info[8],info[9], info[10])
	#"Tiempo Max {}\nSelección : {}\k-Torneo : {}\nCrossover : {}\nMutación : {}\nReemplazo : {}".format(info[5],info[6],info[7],info[8],info[9], info[10])
	title_info = "Evolucion promedio {} con {} iteraciones".format(str(title), str(info[11]))
	plt.title(title_info,y=1.2)
	plt.text(x=.15, y=.95, s=info_1, fontsize=9, ha='left', va='center', transform=plt.gcf().transFigure)
	plt.text(x=.35, y=.95, s=info_2, fontsize=9, ha='left', va='center', transform=plt.gcf().transFigure)
	plt.savefig(file_path,bbox_inches='tight')

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
	raw_data = get_data_from_txt(file_name+".txt","crossoverdata")
	d_1, d_2 = generate_avg_data(raw_data)
	file_name_generic = file_name
	gen_basic_graph(d_1,d_2,file_name_generic)

def graph_vs_generation(file_name1,file_name2, file_name3):
	raw_data_1 = get_data_from_txt(file_name1+".txt","crossoverdata")
	raw_data_2 = get_data_from_txt(file_name2+".txt","crossoverdata")
	raw_data_3 = get_data_from_txt(file_name3+".txt","crossoverdata")
	d_1_1, d_1_2 = generate_avg_data(raw_data_1)
	d_2_1, d_2_2 = generate_avg_data(raw_data_2)
	d_3_1, d_3_2 = generate_avg_data(raw_data_3)
	file_name_generic = file_name1+"vs"+file_name2+"vs"+file_name2 
	gen_vs_graph([d_1_1,d_1_2], [d_2_1,d_2_2],[d_3_1,d_3_2],file_name_generic,file_name1,file_name2,file_name3)
	#gen_basic_graph(d_1,d_2,file_name_generic)



#METODOS PARA AUTOMATIZAR EL PROCESO DE GUARDAR LOS DATOS EN UN .TXT Y GENERAR LAS GRAFICAS 
#>>>>>>>>>>>>>>>>>

def generate_croossover_time_and_graphic(file, iterations,min_per_size,max_per_size): 
	#A este se le tiene que cambiar la semilla aleatoria  
	
	metrics = Metrics()
	metrics.get_params()

	file_basic = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"CrossTime"+str(min_per_size)+"-"+str(max_per_size)
	file_name_txt = file_basic+".txt"
	metrics.register_crossover_time(min_per_size,max_per_size,iterations,file_name_txt)

	graph_generation(file_basic)


def generate_ind_txt_and_graphic(file):
	'''
		Metodo para generar un .txt de datos de una ejecucion individual y despues leer los datos
		y generar una grafica 
	
	'''
	#A este se le tiene que cambiar la semilla aleatoria
	metrics = Metrics()
	metrics.get_params()

	colors = ['black', 'red', 'blue', 'green', 'purple']

	file+="IndExecution"

	file_name_txt = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+str(file)+".txt"

	metrics.register_ga_individual_execution(file_name_txt)

	data = get_data_from_txt_individuals(file_name_txt,"gaindividualexecutions")
	file_name_graph = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+file
	get_ind_exe_graph(data[0],data[1:],["Generations", "Best of Offspring", "Avg Offspring", "Best", "Avg Fitness"],colors, file_name_graph)


def generate_avg_txt_and_graphic(file, iterations):
	'''
		Metodo para generar un .txt de datos de una evolucion promedio y despues leer los datos
		y generar una grafica 
	
	'''
	#A este se le tiene que cambiar la semilla aleatoria
	metrics = Metrics()
	metrics.get_params()

	colors = ['black', 'red', 'blue', 'green','purple']

	file+="MeanEvolution" 

	file_name_txt = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+str(file)+".txt"

	metrics.register_ga_avg_execution(file_name_txt,iterations)

	data = get_data_from_txt_mean_evolution(file_name_txt,"executionsforavg")

	file_name_graph = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+file
	get_avg_exe_graph(data[0],data[1:],["Generations", "Best of Offspring", "Avg Offspring", "Best", "Avg Fitness"],colors, file_name_graph)
	
	

if __name__ == '__main__':
 
	# TOMA DE PARAMETROS POR DEFECTO
	metrics = Metrics()
	metrics.get_params()

	# PRUEBAS DE CROSSOVER 
	metrics.genetic_algo.init_population()
	s_1 = metrics.genetic_algo.current_pop[0]
	s_2 = metrics.genetic_algo.current_pop[1]
	print(s_1)
	print(s_2)
	print("CROSSOVER")
	metrics.genetic_algo.crossover_operator.cross(s_1,s_2)

	# metrics.simple_execution()
	
	
	#EN REVISION >>>
	
	#File output
	#file_name_txt = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"IndExecutionTest.txt"
	#file_name_txt = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"IndExecution.txt"

	#Individual Execution 
	#Se registran las ejecuciones individuales en una file 
	#metrics.register_ga_individual_execution(file_name_txt)
	
	#Se leen los datos de esa file 
	#data = get_data_from_txt_individuals(file_name_txt, "gaindividualexecutions")
	#Generamos la grafica 
	#file_name_graph = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"IndExecution"
	# colors = ['black', 'red', 'blue', 'green', 'purple']
	# #get_ind_exe_graph(data[0],data[1:],["Generations", "Best of Offspring", "Avg Offspring", "Best", "Avg Fitness"],colors, file_name_graph)
	# file = "MeanEvolutionTest"

	#GENERAR DATOS Y GRAFICAS DE TIEMPOS DE CROSSOVER 
	#generate_croossover_time_and_graphic("", 40,15,50)
	
	#PRUEBAS PARA PERMUTACIONES GRANDES 
	#generate_croossover_time_and_graphic("", 50,100,150)


	#CONTRASTE DE CROSSOVERS 
	#graph_vs_generation("BasicCrossTime","IMPXCrossTime","PMXCrossTime")
	#PERMUTACIONES GRANDES
	#graph_vs_generation("BasicCrossTime100-150","IMPXCrossTime100-150","PMXCrossTime100-150")

	#Queens/output/crossoverdata/BasicCrossTime100-150.txt
	#Queens/output/crossoverdata/IMPXCrossTime100-150.txt
	#Queens/output/crossoverdata/PMXCrossTime100-150.txt
	#GENERAR DATOS Y GRAFIFCAS DE EJECUCIONES INDIVIDUALES 
	#generate_ind_txt_and_graphic(""); 

	#GENERAR DATOS Y GRAFICAS EVOLUCION PROMEDIO 
	#generate_avg_txt_and_graphic("", 50)

	
	
	
	

	#metrics.execute_algorithm()
	#file_name_txt = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"IndExecution.txt"
	# file_name_graph = str(metrics.genetic_algo.crossover_operator.__class__.__name__)+"IndExecution"

	#data = get_data_from_txt_individuals(file_name_txt,"gaindividualexecutions")
	#print(data,["Generations", "Best of Offspring", "Avg Offspring", "Best", "Avg Fitness"],file_name_graph)
	#Se registran 
	#metrics.register_crossover_time(15, 35, 30,file_name+".txt")
	#graph_generation(file_name)
	#graph_vs_generation("IMPXTimes","PMXTimes")
	pass