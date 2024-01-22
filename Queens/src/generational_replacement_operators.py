from abc import ABC, abstractmethod
import numpy as np 

class GenerationalReplacement(ABC):
    '''
    Super clase que modela a los operadores de reemplazo generacional 

    '''

    def __init__(self, replacement_size):
        self.replacement_size = replacement_size 

    @abstractmethod
    def replace(self,old_population, offspring):
        '''
        Metodo para realizar un reemplazo generacional 
        '''
        pass

class ElitismMuPlusLambda(GenerationalReplacement):

    def replace(self,old_population, offspring):
        
        generation_pol = np.hstack((old_population,offspring))
        return sorted(generation_pol, key = lambda solution : solution.fitness)[:self.replacement_size]





