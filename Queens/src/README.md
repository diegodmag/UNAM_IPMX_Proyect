# Contenido del directorio 

Scripts con la lógica para un algoritmo genético así como lectura, escritura y visualización de datos. 

## Archivos

### `permutation_based_problems.py`

- Descripción: Contiene a la clase padre PermutationBasedProblem de la cual heredan las clases 
  que modelan un problema cuyo chromosoma tiene representación en permutación. La segunda clase 
  es NQueens, que modela instancias del problema de las n-reinas. 

### `selection_operators.py`

- Descripción: Contiene a la clase padre SelectionOp de la cual heredan las clases que modelan 
  un operador de selección para problemas cuyo chromosoma tiene representación en permutación. 
  
### `crossover_operators.py`

- Descripción: Contiene a la clase padre CrossoverOp de la cual heredan las clases que modelan 
  un operador de cruza para problemas cuyo chromosoma tiene representación en permutación. 

### `generational_replacement_operators.py`

- Descripción: Contiene a la clase padre GenerationalReplacement de la cual heredan las clases que modelan 
  un operador de reemplazo generacional para problemas cuyo chromosoma tiene representación en permutación. 

### `mutation_operators.py`

- Descripción: Contiene a la clase padre MutationOp de la cual heredan las clases que modelan 
  un operador de mutación para problemas cuyo chromosoma tiene representación en permutación.  

### `ga.py`

- Descripción: Contiene la estructa generica de un algoritmo genetico.  

### `executions.py`

- Descripción: Contiene clases y metodos para lectura, escritura y visualizacion de distintos 
  datos del algoritmo genetico y de los operadores. 


## Ejecucion de algoritmo 

Se ejecuta en WSL 2 Ubuntu 20.04.6 LTS

### Requiere : 
- Python 3.8
- numpy : `pip install numpy`
- matplotlib : `pip install matplotlib` o `conda install -c conda-forge matplotlib`
- pandas : `pip install pandas` o `conda install pandas`

Estando dentro del directorio `Queens` ejecutar : 

`python3 src/executions.py 8 100 .8 .1 50 5 1 3 1 0 0`

donde 

`python3 src/executions.py permutation_size[0] population_size[1]  crossover_probability[2] mutation_probability[3] max_generations[4] max_time[5] selection_operator[6] tournament_size[7] crossover_operator[8] mutation_operator[9] generational_replacement_operator[10]`

### Consideraciones : 

#### Selection Operator: [6] 
- 0 :Seleccion por ruleta
- 1 :Seleccion por k-torneo[7]
#### Crossover Operator : [8]
- 0 :Cruza básica de cromosomas con representacion de permutacion vista en clase
- 1 :Partially Mapped Crossover
- 2 :Improved Partially Mapped Crossover
#### Mutation Operator : [9]
- 0 :Mutacion de intercambio aleatorio de dos genes
#### Generational Replacement [10]: 
- 0 :Elitismo Mu+Lambda 




