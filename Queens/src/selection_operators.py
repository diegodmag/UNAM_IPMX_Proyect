from abc import ABC, abstractmethod
import numpy as np 
 

class SelectionOp(ABC):
    '''
    Class modeling different types of selection operators
    '''
    
    @abstractmethod
    def select(self,population, pop_size):
        '''
        Funcion que dada una poblacion, selecciona un conjunto de individuos 

        Params: 
            population: list[Object]
                La lista de la cual se seleccionan los individuos 
            pop_size : int 
                Tamanio de la poblacion 
        '''
class Roulette(SelectionOp): 
    
    #POR ALGUNA RAZON SE ESTA CICLANDO EL ALGORITMO GENETICO CON ESTE OPERADOR 

    def select(self, population, pop_size):

        fitness_sum = sum([ind.fitness for ind in population]); 
        probabilities = np.array([ind.fitness/fitness_sum for ind in population])

        return np.random.choice(population,pop_size,p=probabilities); 
        

class Tournament(SelectionOp): 

    def __init__(self, k):
        self.k_size = k; 

    def select(self, population, pop_size):
        
        selection = []
        for i in range(pop_size):
            #Esto es sin reemplazo ->participants = np.random.choice(self.current_pop,self.tournament_size,False)
            participants = np.random.choice(population,self.k_size); 
            chosen_one = self.get_best_among(participants); 
            selection.append(chosen_one)
        
        return np.array(selection)
    
            
    def get_best_among(self,pop_sample):

        #Obtenemos los fitness de todas las soluciones de la muestra de la poblacion y las almacenamos 
        fitnees_arr = np.array([ind.fitness for ind in pop_sample])
        #Obtenemos el indice del fitness menor
        min_index = np.argmin(fitnees_arr)
        #Regresamos ese objeto
        return pop_sample[min_index]

