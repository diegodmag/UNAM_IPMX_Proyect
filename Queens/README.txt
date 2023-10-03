Utilizado 

SO : ubuntu 22.04.04 
Python 3.10.9
Bibliotecas utilizadas : 
        - numpy
        - matplotlib

Instrucciones para NUMPY: 
    
    pip install numpy

Instrucciones para MATPLOTLIB 
Se intstala tanto con pip como con miniconda : 
    
    pip install matplotlib

    ó 

    conda install -c conda-forge matplotlib

Ejecucion del programa : 

Estando dentro del directorio de la tarea '316032708', el script en donde se ejecutan los algoritmos es src/ga.py y el comando es 

    python3 src/ga.py x:int y:int z:float w:float v:int 

Donde 
    x : Es el numero de reinas 
    y : El tamaño de la poblacion 
    z : Probabilidad de cruza (.8,.9)
    w : Probabilidad de mutacion (.1,.2)
    v : Tiempo maximo de ejecucion 

Ejemplo : 

    python3  src/ga.py 8 100 .8 .1 5 

El anterior ejemplo es ejecutar el algoritmo genetico de 8 reinas con una poblacion de 100, probabiliidad de cruza de .8, probabilidad de mutacion de .1 y con un maximo de 5 segundos. 