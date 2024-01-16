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
    def debugged_cross(self,parent1, parent2):
        pass

    @abstractmethod
    def cross_population(self, population):
        pass 

    def cross_population(self, population, pop_size):

        offspring = []
        for i in range(int(pop_size/2)):
            #Realizamos tantas interaciones como la mitad de la poblacion
            #Ya que por cada iteracion obtenemos dos hijos 
            parents = np.random.choice(population,2)
            #offspring_1, offspring_2 = self.cross(parents[0],parents[1])
            offspring_1, offspring_2 = self.cross(parents[0],parents[1])
            offspring.append(offspring_1)
            offspring.append(offspring_2)
        
        return np.array(offspring)

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
    
    def debugged_cross(self,parent1, parent2):
        pass

class PMX(CrossoverOp): 
    
    def cross(self,parent1, parent2):
        
        n = len(parent1.chromosome)
        
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]
        
        #Generamos relacion de mapeo
        rel_R, rel_L = self.mapping_relation_generation(parent1.chromosome[cp_1:cp_2],parent2.chromosome[cp_1:cp_2])
        
        #Copias de los padres 
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()

        #Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte
        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        for i in range(0,cp_1):
            while primitive_offspring_1_chromosome[i] in rel_R:
                primitive_offspring_1_chromosome[i]=rel_R[primitive_offspring_1_chromosome[i]]
            while primitive_offspring_2_chromosome[i] in rel_L:
                primitive_offspring_2_chromosome[i]=rel_L[primitive_offspring_2_chromosome[i]]
        
        for i in range(cp_2,n):
            while primitive_offspring_1_chromosome[i] in rel_R:
                primitive_offspring_1_chromosome[i]=rel_R[primitive_offspring_1_chromosome[i]]
            while primitive_offspring_2_chromosome[i] in rel_L:
                primitive_offspring_2_chromosome[i]=rel_L[primitive_offspring_2_chromosome[i]]

        return np.array(primitive_offspring_1_chromosome), np.array(primitive_offspring_2_chromosome)
    
    def debugged_cross(self,parent1, parent2):

        print("Parent 1:"+str(parent1.chromosome))
        print("Parent 2:"+str(parent2.chromosome))

        n = len(parent1.chromosome)
        
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]
        print("Cutting points :"+str(cp_1)+" ,"+str(cp_2))
        
        #Generamos relacion de mapeo
        rel_R, rel_L = self.mapping_relation_generation(parent1.chromosome[cp_1:cp_2],parent2.chromosome[cp_1:cp_2])
        
        print("Mapping Relationship R :"+str(rel_R))
        print("Mapping Relationship L :"+str(rel_L))

        #Copias de los padres 
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()

        #Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte
        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        print("Primitive Offsrping 1 : "+str(primitive_offspring_1_chromosome))
        print("Primitive Offsrping 2 : "+str(primitive_offspring_2_chromosome))

        for i in range(0,cp_1):
            while primitive_offspring_1_chromosome[i] in rel_R:
                primitive_offspring_1_chromosome[i]=rel_R[primitive_offspring_1_chromosome[i]]
            while primitive_offspring_2_chromosome[i] in rel_L:
                primitive_offspring_2_chromosome[i]=rel_L[primitive_offspring_2_chromosome[i]]
        
        for i in range(cp_2,n):
            while primitive_offspring_1_chromosome[i] in rel_R:
                primitive_offspring_1_chromosome[i]=rel_R[primitive_offspring_1_chromosome[i]]
            while primitive_offspring_2_chromosome[i] in rel_L:
                primitive_offspring_2_chromosome[i]=rel_L[primitive_offspring_2_chromosome[i]]

        return np.array(primitive_offspring_1_chromosome), np.array(primitive_offspring_2_chromosome)

    def mapping_relation_generation(self, substr_1, substr_2):
        
        map_rel_R={}
        map_rel_L={}

        for i in range(len(substr_1)):
            map_rel_R[substr_2[i]] = substr_1[i]
            map_rel_L[substr_1[i]] = substr_2[i]

        return map_rel_R,map_rel_L

       
        

