from abc import ABC, abstractmethod
import random as rnd 

class MutationOp(ABC):
    '''
    Class modeling different types of mutation operators

    Attributes : 
    self.mutate_prob : float 
        La probabilidad de mutar a un individuo 

    '''
    def __init__(self, mutate_prob) :
        self.mutate_prob = mutate_prob; 
    
    @abstractmethod
    def mutate(self,individual):
        pass 
    
    def mutate_population(self, population):
        '''
        Funcion para mutar una poblacion
        
        Params:
            population: list[Object]
        '''
        for ind in population:
            if(rnd.random() < self.mutate_prob):
                self.mutate(ind)

class SimpleSwap(MutationOp): 
    
    '''
        Se intercambian dos elementos 
    
    '''
    def mutate(self,individual):
        index_1, index_2 = rnd.sample(range(len(individual.chromosome)),2)
        temp = individual.chromosome[index_1]
        individual.chromosome[index_1] = individual.chromosome[index_2]
        individual.chromosome[index_2] = temp 
        
    