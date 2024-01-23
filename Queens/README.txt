## Utilizado 

SO : ubuntu 22.04.04 
Python 3.10.9
Bibliotecas utilizadas : 
        - numpy
        - matplotlib

## Instrucciones para NUMPY: 
    
    pip install numpy

## Instrucciones para MATPLOTLIB 
Se intstala tanto con pip como con miniconda : 
    
    pip install matplotlib

    ó 

    conda install -c conda-forge matplotlib

# Ejecucion del programa : 

Ejemplo : 

    python3  src/ga.py 8 100 .8 .1 5 

El anterior ejemplo es ejecutar el algoritmo genetico de 8 reinas con una poblacion de 100, probabiliidad de cruza de .8, probabilidad de mutacion de .1 y con un maximo de 5 segundos. 

Estando dentro del directorio Queens/ , ejecutar : 

    python3 src/executions.py 8 100 .8 .1 50 5 1 3 1 0 0 

    donde : 
    python3 src/executions.py   permutation_size:int  population_size:int  crossover_probability:float(.8-.9)
                                mutation_probability:float (.1-.2)  max_number_of_generations:int  max_time(seconds):int 
                                selection_operator:int , tournament_selection_size(default 3), crossover_operator:int
                                mutation_operator:int, generational_replacement_operator:int 
    
    Por lo que el ejemplo significa 

    python3 src/executions.py   8           100        .8           .1          50             5            1             3            1         0             0
                            Permutacion  Poblacion Prob.Crossover Prob.Mut  Generaciones  TiempoMaximo  Op.Seleccion Tamaño.Torneo  Op.Cruza  Op.Mutacion  Op.Reemplazo

    Consideraciones sobre operadores : 

    selection_operator :
        0 -> Seleccion por ruleta 
        1 -> Seleccion por torneo 
    tournament_selection_size :
        Es necesario especificar un tamaño del torneo aunque no se la seleccion por tournament_selection_size

    crossover_operator:
        0 -> Cruza básica de cromosomas con representacion de permutacion vista en clase 
        1 -> Partially Mapped Crossover 
        2 -> Improved Partially Mapped Crossover
    
    mutation_operator : 
        0 -> Mutacion de intercambio aleatorio de dos genes 
    
    generation_replacement :
        0 -> Elitismo Mu+Lambda 
    
