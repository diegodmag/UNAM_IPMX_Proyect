

# Parametros del algoritmo genetico

PERMUTATIONSIZE ?= 8
POPULATIONSIZE ?= 100

CROSSOVER ?= IPMX
CROSSNUM ?= 0

# Parametros de la generacion y visualizacion de los datos 

# Nombres de crossovers 
CROSSOVERSNAMES = PMX IPMX PMXCastudil


# Seleccion de crossovers 

ifeq ($(CROSSOVER), Basic)
	CROSSNUM = 0
else ifeq ($(CROSSOVER), PMX)
	CROSSNUM = 1
else ifeq ($(CROSSOVER), IPMX)
	CROSSNUM = 2
else ifeq ($(CROSSOVER), Ordered)
	CROSSNUM = 3
else ifeq ($(CROSSOVER), PMX_Castudil)
	CROSSNUM = 5
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
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) .8 .1 50 5 1 3 $(CROSSNUM) 0 0 0 0 0 0 0
crossover_times : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) .8 .1 50 5 1 3 $(CROSSNUM) 0 0 0 1 $(REPETITIONS) $(MINPERMUTATIONS) $(MAXPERMUTATIONS) 
compare_crossover_times : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) .8 .1 50 5 1 3 $(CROSSNUM) 0 0 0 2 $(REPETITIONS) $(MINPERMUTATIONS) $(MAXPERMUTATIONS)
individual_execution : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) .8 .1 50 5 1 3 $(CROSSNUM) 0 0 0 3 0 0 0 
mean_evolution : 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) .8 .1 50 5 1 3 $(CROSSNUM) 0 0 0 4 $(REPETITIONS) 0 0

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

