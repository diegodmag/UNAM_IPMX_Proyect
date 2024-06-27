

# Parametros del algoritmo genetico

PERMUTATIONSIZE ?= 8
POPULATIONSIZE ?= 500

CROSSOVER ?= IPMX
CROSSNUM ?= 0

GENERATIONS ?=50

CROSSPROBA ?=1
MUTATIONPROBA ?=0

# Parametros de la generacion y visualizacion de los datos 

# Nombres de crossovers 
CROSSOVERSNAMES = PMX IPMX PMXCastudil Ordered

export CROSSOVERSNAMES

# Seleccion de crossovers 

ifeq ($(CROSSOVER), Basic)
	CROSSNUM = 0
else ifeq ($(CROSSOVER), PMX)
	CROSSNUM = 1
else ifeq ($(CROSSOVER), IPMX)
	CROSSNUM = 2
else ifeq ($(CROSSOVER), Ordered)
	CROSSNUM = 3
else ifeq ($(CROSSOVER), PMXCastudil)
	CROSSNUM = 5
else ifeq ($(CROSSOVER), Uniform)
	CROSSNUM = 6
else ifeq ($(CROSSOVER),Cycle)
	CROSSNUM = 7
else
	CROSSNUM = 1
endif

#Parametros para las operaciones 

#Para los datos de los crossovers y evolucion promedio
REPETITIONS ?= 5 
#Para los datos de los crossovers
MINPERMUTATIONS ?= 10 
MAXPERMUTATIONS ?= 40 

crossover_test : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 0 0 0 0
crossover_times : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 1 $(REPETITIONS) $(MINPERMUTATIONS) $(MAXPERMUTATIONS) 
compare_crossover_times : 
#make compare_crossover_times CROSSOVERSNAMES='PMXCastudil , IPMX' REPETITIONS=100 MINPERMUTATIONS=970 MAXPERMUTATIONS=1000
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 2 $(REPETITIONS) $(MINPERMUTATIONS) $(MAXPERMUTATIONS)
individual_execution : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 3 0 0 0 
mean_evolution : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 4 $(REPETITIONS) 0 0
crossover_specific:
	python3 src/executions.py 13 $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 5 0 0 0
compare_best_fitness_mean_between_crossovers:
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 6 $(REPETITIONS) 0 0
avg_data_specific_times:
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 7 $(REPETITIONS) 0 0
boxplot_times_information:
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 8 $(REPETITIONS) 0 0
#Para hacer pruebas dejar corriendo esto 
fitness_information:
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 9 $(REPETITIONS) 0 0
get_pdf_avg_graph : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) $(CROSSPROBA) $(MUTATIONPROBA) $(GENERATIONS) 5 1 3 $(CROSSNUM) 0 0 0 10 $(REPETITIONS) 0 0

# mean_evolution_executions:
# 	@for PERMUTATIONSIZE in 300 400 500 600 700 800 900 1000 1500 2000 3000; do \
# 		for CROSSOVER in PMX IPMX Ordered PMXCastudil; do \
# 			make mean_evolution PERMUTATIONSIZE=$$PERMUTATIONSIZE CROSSOVER=$$CROSSOVER REPETITIONS=30; \
# 		done; \
# 	done

all_times_information : 
	@for PERMUTATIONSIZE in 200; do \
		for CROSSOVER in Ordered; do \
			make times_information PERMUTATIONSIZE=$$PERMUTATIONSIZE CROSSOVER=$$CROSSOVER REPETITIONS=30; \
		done; \
	done

all_fitness_information:
	@for PERMUTATIONSIZE in 1000; do \
		for CROSSOVER in PMX IPMX; do \
			make fitness_information PERMUTATIONSIZE=$$PERMUTATIONSIZE CROSSOVER=$$CROSSOVER REPETITIONS=30; \
		done; \
	done

mean_evolution_executions_rest:
	@for PERMUTATIONSIZE in 1000; do \
		for CROSSOVER in PMXCastudil Ordered; do \
			make mean_evolution PERMUTATIONSIZE=$$PERMUTATIONSIZE CROSSOVER=$$CROSSOVER REPETITIONS=30; \
		done; \
	done

	@for PERMUTATIONSIZE in 1500; do \
		for CROSSOVER in PMX IPMX Ordered PMXCastudil; do \
			make mean_evolution PERMUTATIONSIZE=$$PERMUTATIONSIZE CROSSOVER=$$CROSSOVER REPETITIONS=30; \
		done; \
	done

individual_execution_samples:
	@for PERMUTATIONSIZE in 600 700 800 900; do \
		for CROSSOVER in PMX IPMX Ordered PMXCastudil Uniform; do \
			make individual_execution PERMUTATIONSIZE=$$PERMUTATIONSIZE CROSSOVER=$$CROSSOVER; \
		done; \
	done

individual_execution_samples_low:
	@for PERMUTATIONSIZE in 10 20 30 40 50 60 70 80 90; do \
		for CROSSOVER in PMX IPMX Ordered PMXCastudil Uniform; do \
			make individual_execution PERMUTATIONSIZE=$$PERMUTATIONSIZE CROSSOVER=$$CROSSOVER; \
		done; \
	done


# repetitions = int(sys.argv[14])
# 		min_permutations = int(sys.argv[15])
# 		max_permutations = int(sys.argv[16])


# run: 
# 	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) .8 .1 50 5 1 3 $(CROSSNUM) 0 0 0 0

# show_names:
# 	# Para que esto funcione hay que hacer make show_names CROSSOVERSNAMES="1 2 3"
# 	@for name in $(CROSSOVERSNAMES); do \
# 		echo $$name; \
# 	done

# elif(operation==1):
# 		print("Se obtienen datos de tiempo promedio de crossover")
# 	elif(operation==2):
# 		print("Se comparan datos de crossovers")
# 	elif(operation==3):
# 		print("Ejecucion individual de algoritmo genetico")
# 	elif(operation==4):
# 		print("Ejecucion promedio de algoritmo genetico")

