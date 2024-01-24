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



