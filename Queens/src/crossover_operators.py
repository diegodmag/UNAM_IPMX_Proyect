from abc import ABC, abstractmethod
import math 
import numpy as np 
import matplotlib.pyplot as plt
import random as rnd 
import copy
import time as tm 
import sys

class CrossoverOp(ABC):
    '''
    Class modeling different types of selection operators

    Attributes
    
    Necesitamos la representacion del problema para acceder a algo ? 

    '''
    
    @abstractmethod
    def cross(self,parent1, parent2):
        pass

    @abstractmethod
    def cross_population(self, population):
        pass 

class Basic(CrossoverOp):
    
    def cross(self,parent1,parent2):

        n = len(parent1.chromosome)
        
        cp_1 = math.floor(n/4)
        cp_2 = math.floor(3*(n/4))

        offspring_chromosome_1 = np.array([-1 for x in range(n)])
        offspring_chromosome_2 = np.array([-1 for x in range(n)])

        offspring_chromosome_1[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        offspring_chromosome_2[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        #Contadores auxiliares 
        cont_select = cp_2
        cont_insert = cp_2

        while(-1 in offspring_chromosome_1):
            if(parent1.chromosome[cont_select%n] not in offspring_chromosome_1): 
                offspring_chromosome_1[cont_insert%n] = parent1.chromosome[cont_select%n]
                cont_select+=1 
                cont_insert+=1
            else:
                cont_select+=1

        cont_select = cp_2
        cont_insert = cp_2

        while(-1 in offspring_chromosome_2):
            if(parent2.chromosome[cont_select%n] not in offspring_chromosome_2): 
                offspring_chromosome_2[cont_insert%n] = parent2.chromosome[cont_select%n]
                cont_select+=1 
                cont_insert+=1
            else:
                cont_select+=1

        return np.array(offspring_chromosome_1), np.array(offspring_chromosome_2)
    
    def cross_population(self, population, pop_size):
        
        offspring = []
        for i in range(int(pop_size/2)):
            #Realizamos tantas interaciones como la mitad de la poblacion
            #Ya que por cada iteracion obtenemos dos hijos 
            parents = np.random.choice(population,2)
            offspring_1, offspring_2 = self.cross(parents[0],parents[1])
            offspring.append(offspring_1)
            offspring.append(offspring_2)
        
        return np.array(offspring)





       
        

