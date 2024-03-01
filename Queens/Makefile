

# Parametros del algoritmo genetico

PERMUTATIONSIZE ?= 8
POPULATIONSIZE ?= 100

CROSSOVER ?= IPMX
CROSSNUM ?= 0

# Parametros de la generacion y visualizacion de los datos 

OPERATION ?= IND


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



run: 
	python3 src/executions.py $(PERMUTATIONSIZE) $(POPULATIONSIZE) .8 .1 50 5 1 3 $(CROSSNUM) 0 0

