from abc import ABC, abstractmethod
import math 
import numpy as np 
import random as rnd 
import copy
import time
class CrossoverOp(ABC):
    '''
    Class modeling different types of crossover operators

    Attributes : 
    self.cross_prob : float 
        La probabilidad de cruzar a dos individuos seleccionados 
    
    '''
    def __init__(self,cross_prob):
        self.cross_prob = cross_prob
        
    @abstractmethod
    def cross(self,parent1, parent2):
        '''
        Funcion que ejecuta la cruza de dos padres 
        
        Params:
            parent1, parent2 : Object
                Instancias de una solucion con representacion de permutacion

        Returns:
            offspring_1, offspring_2 : list[int] 
                Dos permutaciones legalizadas 
        '''
        pass

    @abstractmethod
    def timed_cross(self,parent1, parent2):
        '''
        Funcion que ejecuta la cruza de dos padres 
        
        Params:
            parent1, parent2 : Object
                Instancias de una solucion con representacion de permutacion

        Returns:
            offspring_1, offspring_2 : list[int] 
                Dos permutaciones legalizadas 
        '''
        pass    
        
    @abstractmethod
    def debugged_cross(self,parent1, parent2):
        '''
        Funcion que ejecuta la cruza de dos padres 
        
        Params:
            parent1, parent2 : Object
                Instancias de una solucion con representacion de permutacion

        Returns:
            offspring_1, offspring_2 : list[int] 
                Dos permutaciones legalizadas 
        '''
        pass

    def cross_population(self, population, pop_size):
        '''
        Funcion para ejecutar la cruza sobre una poblacion dada 

        Params:
            population : list[Object]
                Lista con soluciones cuya representacion es una permutacion 
            pop_size: int 
                tamanio de la poblacion 
        
        Returns: 
            offspring : list[list[int]]
                Lista con las permutaciones (cromosomas) para generar la poblacion 

        '''
        offspring = []
        for i in range(int(pop_size/2)):
            #Realizamos tantas interaciones como la mitad de la poblacion
            #Ya que por cada iteracion obtenemos dos hijos 
            parents = np.random.choice(population,2)
            if(rnd.random() < self.cross_prob):    
                offspring_1, offspring_2 = self.cross(parents[0],parents[1])
                offspring.append(offspring_1)
                offspring.append(offspring_2)
            else:
                #Si la probabilidad es mayor entonces se regresan dos copias de los padres 
                offspring_1, offspring_2 = copy.copy(parents[0].chromosome),copy.copy(parents[1].chromosome)
            
        return np.array(offspring)

    #Tambien necesitamos un metodo que le cambie la semilla 
    

class Basic(CrossoverOp):

    '''
    Operador de cruza básico visto en clase 
    
    '''

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
    
    def timed_cross(self,parent1,parent2):

        time_start = time.time()

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

        time_end = time.time()  

        return time_end-time_start
          
    def debugged_cross(self,parent1, parent2):
        pass

#Ordered Crossover 
#Determinar los puntos de
# 

class Ordered(CrossoverOp):


    def cross(self, parent1, parent2):
        n = len(parent1.chromosome)

        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]

        #Las copias del  
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()
        
        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        #Necesitamos borrar los elementos de P2 que ya estan en PO1 
        #Para eso requerimos saber cuales son 
        marked_1 = np.full(n,-1,dtype=int)
        marked_2 = np.full(n,-1,dtype=int)

        #Marcamos los elementos que ya estan en la cadena 
        for gen in primitive_offspring_1_chromosome[cp_1:cp_2]: 
            marked_1[gen] = 1
        for gen in primitive_offspring_2_chromosome[cp_1:cp_2]: 
            marked_2[gen] = 1
         
        p1_copy = parent1.chromosome.copy()
        p2_copy = parent2.chromosome.copy()

        #Eliminamos a los marcados 
        for i in range(n):
            if(marked_1[p1_copy[i]]==1):
                p1_copy[i]=-1
        for i in range(n):
            if(marked_2[p2_copy[i]]==1):
                p2_copy[i]=-1
            
        #LEGALIZING SECOND OFFSPRING 1 
        cont = 0 
        cont_p = 0 
        while cont < cp_1:
            if(p1_copy[cont_p] != -1):
                primitive_offspring_1_chromosome[cont]=p1_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1
        cont = cp_2 
        while cont < n:
            if(p1_copy[cont_p] != -1):
                primitive_offspring_1_chromosome[cont]=p1_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1

    #LEGALIZING SECOND OFFSPRING 2 
        cont = 0 
        cont_p = 0 
        while cont < cp_1:
            if(p2_copy[cont_p] != -1):
                primitive_offspring_2_chromosome[cont]=p2_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1
        cont = cp_2 
        while cont < n:
            if(p2_copy[cont_p] != -1):
                primitive_offspring_2_chromosome[cont]=p2_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1

        return np.array(primitive_offspring_1_chromosome), np.array(primitive_offspring_2_chromosome)
    
    def timed_cross(self, parent1, parent2):
        time_start = time.time()
        n = len(parent1.chromosome)

        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]

        #Las copias del  
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()
        
        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        #Necesitamos borrar los elementos de P2 que ya estan en PO1 
        #Para eso requerimos saber cuales son 
        marked_1 = np.full(n,-1,dtype=int)
        marked_2 = np.full(n,-1,dtype=int)

        #Marcamos los elementos que ya estan en la cadena 
        for gen in primitive_offspring_1_chromosome[cp_1:cp_2]: 
            marked_1[gen] = 1
        for gen in primitive_offspring_2_chromosome[cp_1:cp_2]: 
            marked_2[gen] = 1
         
        p1_copy = parent1.chromosome.copy()
        p2_copy = parent2.chromosome.copy()

        #Eliminamos a los marcados 
        for i in range(n):
            if(marked_1[p1_copy[i]]==1):
                p1_copy[i]=-1
        for i in range(n):
            if(marked_2[p2_copy[i]]==1):
                p2_copy[i]=-1
            
        #LEGALIZING SECOND OFFSPRING 1 
        cont = 0 
        cont_p = 0 
        while cont < cp_1:
            if(p1_copy[cont_p] != -1):
                primitive_offspring_1_chromosome[cont]=p1_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1
        cont = cp_2 
        while cont < n:
            if(p1_copy[cont_p] != -1):
                primitive_offspring_1_chromosome[cont]=p1_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1

    #LEGALIZING SECOND OFFSPRING 2 
        cont = 0 
        cont_p = 0 
        while cont < cp_1:
            if(p2_copy[cont_p] != -1):
                primitive_offspring_2_chromosome[cont]=p2_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1
        cont = cp_2 
        while cont < n:
            if(p2_copy[cont_p] != -1):
                primitive_offspring_2_chromosome[cont]=p2_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1

        time_end = time.time()  

        return time_end-time_start

    def debugged_cross(self, parent1, parent2):
        
        n = len(parent1.chromosome)

        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]

        #Las copias del  
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()
        
        print("PARENT 1", primitive_offspring_1_chromosome)
        print("PARENT 2", primitive_offspring_2_chromosome)


        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        print("SUBCADENA 1",primitive_offspring_1_chromosome[cp_1:cp_2] )
        print("SUBCADENA 2",primitive_offspring_2_chromosome[cp_1:cp_2] )


        print("PRIMITIVE OFFSPRING 1", primitive_offspring_1_chromosome)
        print("PRIMITIVE OFFSPRING 2", primitive_offspring_2_chromosome)

        #Necesitamos borrar los elementos de P2 que ya estan en PO1 
        #Para eso requerimos saber cuales son 
        marked_1 = np.full(n,-1,dtype=int)
        marked_2 = np.full(n,-1,dtype=int)

        #Marcamos los elementos que ya estan en la cadena 
        for gen in primitive_offspring_1_chromosome[cp_1:cp_2]: 
            marked_1[gen] = 1
        for gen in primitive_offspring_2_chromosome[cp_1:cp_2]: 
            marked_2[gen] = 1
         
        print("MARKED LIST 1", marked_1)
        print("MARKED LIST 2", marked_2)

        p1_copy = parent1.chromosome.copy()
        p2_copy = parent2.chromosome.copy()

        #Eliminamos a los marcados 
        for i in range(n):
            if(marked_1[p1_copy[i]]==1):
                p1_copy[i]=-1
        for i in range(n):
            if(marked_2[p2_copy[i]]==1):
                p2_copy[i]=-1
            
        print("UPDATED PARENT", p1_copy)
        print("UPDATED PARENT", p2_copy)

        #LEGALIZING SECOND OFFSPRING 1 
        cont = 0 
        cont_p = 0 
        while cont < cp_1:
            if(p1_copy[cont_p] != -1):
                primitive_offspring_1_chromosome[cont]=p1_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1
        cont = cp_2 
        while cont < n:
            if(p1_copy[cont_p] != -1):
                primitive_offspring_1_chromosome[cont]=p1_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1

    #LEGALIZING SECOND OFFSPRING 2 
        cont = 0 
        cont_p = 0 
        while cont < cp_1:
            if(p2_copy[cont_p] != -1):
                primitive_offspring_2_chromosome[cont]=p2_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1
        cont = cp_2 
        while cont < n:
            if(p2_copy[cont_p] != -1):
                primitive_offspring_2_chromosome[cont]=p2_copy[cont_p]
                cont +=1
                cont_p+=1
            else:
                cont_p+=1


        print("LEGALIZED CHROMOSOME 1", primitive_offspring_1_chromosome)
        print("LEGALIZED CHROMOSOME 2", primitive_offspring_2_chromosome)
        
        return np.array(primitive_offspring_1_chromosome), np.array(primitive_offspring_2_chromosome)
    
class PMX(CrossoverOp): 
    

    '''
        Operador de cruza Partially Mapped Crossover 
    '''

    def cross(self,parent1, parent2):
        
        n = len(parent1.chromosome)
        
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]
        #cp_1, cp_2 = 6,10
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
    
    def timed_cross(self,parent1, parent2):
        time_start = time.time()
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

        time_end = time.time()  

        return time_end-time_start
    
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
        
        '''
        Funcion que genera dos relaciones de mapeo a partir de dos subcadenas. Las relaciones de mapeo 
        son dos diccionarios. Un cuyas llaves son los elementos de substr_1 y valores los elementos de substr_2
        y el otro el caso analogo 

        Params: 
            substr_1 : list[int]
                Subcadena de un cromosoma
            substr_2 : list[int]
                Subcadena de un cromosoma 
        
        Returns : 
            map_rel_R : dic[int:int] 
                Relacion de mapeo de direccion substr_2 : substr_1

            map_rel_L : dic[int:int] 
                Relacion de mapeo de direccion substr_1 : substr_2
        '''
        map_rel_R={}
        map_rel_L={}

        for i in range(len(substr_1)):
            map_rel_R[substr_2[i]] = substr_1[i]
            map_rel_L[substr_1[i]] = substr_2[i]

        return map_rel_R,map_rel_L


#Recuperado de https://stackoverflow.com/questions/53254449/how-to-perform-partial-mapped-crossover-in-python3
class PMXSTACK(CrossoverOp): 
    
    def timed_cross(self,parent1, parent2):

        time_start = time.time()
        parent1 = parent1.chromosome
        parent2 = parent2.chromosome

        cutoff_1, cutoff_2 = np.sort(np.random.choice(np.arange(len(parent1)+1), size=2, replace=False))

        def PMX_one_offspring(p1, p2):
            offspring = np.zeros(len(p1), dtype=p1.dtype)

            # Copy the mapping section (middle) from parent1
            offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

            # copy the rest from parent2 (provided it's not already there
            for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
                candidate = p2[i]
                while candidate in p1[cutoff_1:cutoff_2]: # allows for several successive mappings
                    #print(f"Candidate {candidate} not valid in position {i}") # DEBUGONLY
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                offspring[i] = candidate
            return offspring

        offspring1 = PMX_one_offspring(parent1, parent2)
        offspring2 = PMX_one_offspring(parent2, parent1)
        
        time_end = time.time() 
        
        return time_end-time_start
    
    def cross(self,parent1, parent2):
        
        parent1 = parent1.chromosome
        parent2 = parent2.chromosome

        cutoff_1, cutoff_2 = np.sort(np.random.choice(np.arange(len(parent1)+1), size=2, replace=False))

        def PMX_one_offspring(p1, p2):
            offspring = np.zeros(len(p1), dtype=p1.dtype)

            # Copy the mapping section (middle) from parent1
            offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

            # copy the rest from parent2 (provided it's not already there
            for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
                candidate = p2[i]
                while candidate in p1[cutoff_1:cutoff_2]: # allows for several successive mappings
                    # print(f"Candidate {candidate} not valid in position {i}") # DEBUGONLY
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                offspring[i] = candidate
            return offspring

        offspring1 = PMX_one_offspring(parent1, parent2)
        offspring2 = PMX_one_offspring(parent2, parent1)
        
        
        return offspring1, offspring2
        
    def debugged_cross(self,parent1, parent2):
        pass 

       
class IPMX(CrossoverOp):
    
    def timed_cross(self,parent1, parent2):
        time_start = time.time()
        n = len(parent1.chromosome)
        #Paso 1
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]

        #Copias de los padres 
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()
		
        subs_1 = parent2.chromosome[cp_1:cp_2]
        subs_2 = parent1.chromosome[cp_1:cp_2]

        
        #Paso 2
        #Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte
        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        #Paso 3 y 4: Generamos e inicializamos la exchange list  (O(m))
        m = len(parent2.chromosome[cp_1:cp_2])
        exchange_list = np.zeros((m,3), dtype=int)

        for i in range(m):
            exchange_list[i,0]= subs_1[i]
            exchange_list[i,1]= subs_2[i]
            exchange_list[i,2]= 1

        #Paso 5,6 y 7 
        #Generacion de la guide list
        #Nuestra implementacion tiene que usar -1 en vez de 0 por que el 0 se ocupa en nuestra representacion 

        guide_list = np.full(n,-1,dtype=int)
        guide_list_prev = np.full(n,-1,dtype=int)
        for i in range(m):
            guide_list[exchange_list[i,0]]=exchange_list[i,1]
            #Es la de regreso 
            guide_list_prev[exchange_list[i,1]]=exchange_list[i,0]

        #guide_list_preve -> Es un diccionario en el otro sentido

        l1 = np.zeros(n,dtype=int)
        l2 = np.zeros(n,dtype=int)

        for i in range(m):
            l1[exchange_list[i,0]] = 1
            l2[exchange_list[i,1]] = 1
        
        l1_l2 = l1+l2

        #Actualizar la exchange list haciendo los nodos intermedios 0 
        for i in range(m):
            # l1[exchange_list[i,0]]+l1[exchange_list[i,0]]
            index = exchange_list[i,0]
            if l1_l2[index] == 2: 
                exchange_list[i,2] = 0
                #index es nodo intermedio 
                next_node = guide_list[index] 
                prev_node = guide_list_prev[index]
                
                guide_list[prev_node] = next_node
                guide_list_prev[next_node] = prev_node

                guide_list[index] = -1 
                guide_list_prev[index] = -1

        ###PASO  8
        for i in range(cp_1):
            #Primero queremos el gen en el padre (excluyendo los genes entre los puntos de corte)
			#Para ese gen buscamos si su valor en la guide list es distinto de -1
			#Si lo es, entonces lo sustituimos por el nuevo valor actualizado de la exchange list
            value = guide_list[parent1.chromosome[i]]
            if value != -1:
                primitive_offspring_1_chromosome[i] = value

        for i in range(cp_2,len(primitive_offspring_1_chromosome)):
            value = guide_list[parent1.chromosome[i]]
            if value != -1:
                primitive_offspring_1_chromosome[i] = value

        f = np.full(n,-1,dtype=int)
        for i in range(n):
            f[primitive_offspring_1_chromosome[i]]=parent1.chromosome[i]

        for i in range(n):
            primitive_offspring_2_chromosome[i]=f[parent2.chromosome[i]]

        time_end = time.time()  

        return time_end-time_start 

    def cross(self,parent1, parent2):
        
        n = len(parent1.chromosome)
        #Paso 1
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]

        #Copias de los padres 
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()
		
        subs_1 = parent2.chromosome[cp_1:cp_2]
        subs_2 = parent1.chromosome[cp_1:cp_2]

        
        #Paso 2
        #Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte
        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        #Paso 3 y 4: Generamos e inicializamos la exchange list  (O(m))
        m = len(parent2.chromosome[cp_1:cp_2])
        exchange_list = np.zeros((m,3), dtype=int)

        for i in range(m):
            exchange_list[i,0]= subs_1[i]
            exchange_list[i,1]= subs_2[i]
            exchange_list[i,2]= 1

        #Paso 5,6 y 7 
        #Generacion de la guide list
        #Nuestra implementacion tiene que usar -1 en vez de 0 por que el 0 se ocupa en nuestra representacion 

        guide_list = np.full(n,-1,dtype=int)
        guide_list_prev = np.full(n,-1,dtype=int)
        for i in range(m):
            guide_list[exchange_list[i,0]]=exchange_list[i,1]
            #Es la de regreso 
            guide_list_prev[exchange_list[i,1]]=exchange_list[i,0]

        #guide_list_preve -> Es un diccionario en el otro sentido

        l1 = np.zeros(n,dtype=int)
        l2 = np.zeros(n,dtype=int)

        for i in range(m):
            l1[exchange_list[i,0]] = 1
            l2[exchange_list[i,1]] = 1
        

        l1_l2 = l1+l2

                


        #Actualizar la exchange list haciendo los nodos intermedios 0 
        for i in range(m):
            # l1[exchange_list[i,0]]+l1[exchange_list[i,0]]
            index = exchange_list[i,0]
            if l1_l2[index] == 2: 
                exchange_list[i,2] = 0
                #index es nodo intermedio 
                next_node = guide_list[index] 
                prev_node = guide_list_prev[index]
                
                guide_list[prev_node] = next_node
                guide_list_prev[next_node] = prev_node

                guide_list[index] = -1 
                guide_list_prev[index] = -1

                
        #Actualizar exchange list eliminando nodos intermedios y conectado los nodos con camino directo 
        # for i in range(m):
        #     if exchange_list[i,2] == 1: 
        #         #El reemplazo actual 
        #         current_replacement = exchange_list[i,1]
        #         #El reemplazo final
        #         final_replacement = -1
        #         while current_replacement != -1:#Al menos se cumple la primera vez 
        #             final_replacement = current_replacement
        #             current_replacement = guide_list[final_replacement]

        #         exchange_list[i,1] = final_replacement

        #Ahora actualizamos la guide list   
        # for i in range(m):
        #     if(exchange_list[i,2]==1):
        #         guide_list[exchange_list[i,0]] =  exchange_list[i,1]
        #     else:
        #         guide_list[exchange_list[i,0]] = -1

        ###PASO  8
        for i in range(cp_1):
            #Primero queremos el gen en el padre (excluyendo los genes entre los puntos de corte)
			#Para ese gen buscamos si su valor en la guide list es distinto de -1
			#Si lo es, entonces lo sustituimos por el nuevo valor actualizado de la exchange list
            value = guide_list[parent1.chromosome[i]]
            if value != -1:
                primitive_offspring_1_chromosome[i] = value

        for i in range(cp_2,len(primitive_offspring_1_chromosome)):
            value = guide_list[parent1.chromosome[i]]
            if value != -1:
                primitive_offspring_1_chromosome[i] = value

        f = np.full(n,-1,dtype=int)
        for i in range(n):
            f[primitive_offspring_1_chromosome[i]]=parent1.chromosome[i]

        for i in range(n):
            primitive_offspring_2_chromosome[i]=f[parent2.chromosome[i]]

          
        return primitive_offspring_1_chromosome,primitive_offspring_2_chromosome

    def debugged_cross(self,parent1, parent2):
        
        n = len(parent1.chromosome)
        #Paso 1
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]
        print("Cutting points :"+str(cp_1)+" ,"+str(cp_2))

        #Copias de los padres 
        primitive_offspring_1_chromosome = parent1.chromosome.copy()
        primitive_offspring_2_chromosome = parent2.chromosome.copy()

        print("Parent 1 :"+str(primitive_offspring_1_chromosome))
        print("Parent 2 :"+str(primitive_offspring_1_chromosome))
		
        subs_1 = parent2.chromosome[cp_1:cp_2]
        subs_2 = parent1.chromosome[cp_1:cp_2]

        print("Substring 1:"+str(subs_1))
        print("Substring 1:"+str(subs_2))
        #Generamos relacion de mapeo
        
        #Paso 2
        #Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte
        primitive_offspring_1_chromosome[cp_1:cp_2] = parent2.chromosome[cp_1:cp_2]
        primitive_offspring_2_chromosome[cp_1:cp_2] = parent1.chromosome[cp_1:cp_2]

        print("Primitive Offspring 1 :"+ str(primitive_offspring_1_chromosome))
        print("Primitive Offspring 2 :"+ str(primitive_offspring_2_chromosome))

        #Paso 3 y 4: Generamos e inicializamos la exchange list  (O(m))
        m = len(parent2.chromosome[cp_1:cp_2])
        exchange_list = np.zeros((m,3), dtype=int)

        for i in range(m):
            exchange_list[i,0]= subs_1[i]
            exchange_list[i,1]= subs_2[i]
            exchange_list[i,2]= 1

        print("Initial Exchange List :")
        print(exchange_list)

        #Paso 5,6 y 7 
        #Generacion de la guide list
        #Nuestra implementacion tiene que usar -1 en vez de 0 por que el 0 se ocupa en nuestra representacion 

        guide_list = np.full(n,-1,dtype=int)
        for i in range(m):
            guide_list[exchange_list[i,0]]=exchange_list[i,1]

        print("Initial Guide List")
        print(guide_list)

        l1 = np.zeros(n,dtype=int)
        l2 = np.zeros(n,dtype=int)

        for i in range(m):
            l1[exchange_list[i,0]] = 1
            l2[exchange_list[i,1]] = 1
        
        print("l1 :"+str(l1))
        print("l2 :"+str(l2))

        l1_l2 = l1+l2
        print("L1 + L2 :"+str(l1_l2))

        #Actualizar la exchange list haciendo los nodos intermedios 0 
        for i in range(m):
            index = exchange_list[i,0]
            if l1_l2[index] == 2: 
                exchange_list[i,2] = 0
        
        print("Updated Exchange List (Removing mid nodes) :")
        print(exchange_list)

        #Actualizar exchange list eliminando nodos intermedios y conectado los nodos con camino directo 
        for i in range(m):
            if exchange_list[i,2] == 1: 
                #El reemplazo actual 
                print("QUE PASHO FOR")
                current_replacement = exchange_list[i,1]
                #El reemplazo final
                final_replacement = -1
                while current_replacement != -1:#Al menos se cumple la primera vez 
                    final_replacement = current_replacement
                    current_replacement = guide_list[final_replacement]

                exchange_list[i,1] = final_replacement

        print("Updated Exchange List (New Paths)")
        print(exchange_list)

        #Ahora actualizamos la guide list   
        for i in range(m):
            if(exchange_list[i,2]==1):
                guide_list[exchange_list[i,0]] =  exchange_list[i,1]
            else:
                guide_list[exchange_list[i,0]] = -1

        print("Updated Guide list")
        print(guide_list)

        ###PASO  8
        for i in range(cp_1):
            #Primero queremos el gen en el padre (excluyendo los genes entre los puntos de corte)
			#Para ese gen buscamos si su valor en la guide list es distinto de -1
			#Si lo es, entonces lo sustituimos por el nuevo valor actualizado de la exchange list
            value = guide_list[parent1.chromosome[i]]
            if value != -1:
                primitive_offspring_1_chromosome[i] = value

        for i in range(cp_2,len(primitive_offspring_1_chromosome)):
            value = guide_list[parent1.chromosome[i]]
            if value != -1:
                primitive_offspring_1_chromosome[i] = value

        print("Offspring 1 : Legalized :")
        print(str(primitive_offspring_1_chromosome))

        f = np.full(n,-1,dtype=int)
        for i in range(n):
            f[primitive_offspring_1_chromosome[i]]=parent1.chromosome[i]

        for i in range(n):
            primitive_offspring_2_chromosome[i]=f[parent2.chromosome[i]]

        print("Offspring 2 : Legalized :")
        print(str(primitive_offspring_2_chromosome))

        return primitive_offspring_1_chromosome,primitive_offspring_2_chromosome


#Basado en https://github.com/castudil/pmx/blob/master/src/pmx/PMX.java
class PMXCastudil(CrossoverOp): 
    def timed_cross(self,parent1, parent2):
        time_start = time.time()
        n = len(parent1.chromosome)
        #Paso 1
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]

        #visited = np.full(n+1,False,dtype=bool)
        visited = np.full(n+1,False,dtype=bool)
        visited_2 = np.full(n+1,False,dtype=bool)

        z = np.full(n,-1,dtype=int)
        z_2 = np.full(n,-1,dtype=int)

        top= cp_2

        if(cp_2==n):
            top=n-1
        
        for i in range(cp_1,top+1):
            z[i] = parent1.chromosome[i]
            visited[z[i]] = True

        for i in range(cp_1,top+1):
            z_2[i] = parent2.chromosome[i]
            visited_2[z_2[i]] = True

        #PRIMER HIJO  
        for i in range(cp_1,top+1):
            if not (visited[parent2.chromosome[i]]):
                k_2 = i
                elementToBeCopied = parent2.chromosome[i]
                #Simulando el do - while 
                while True:
                    V = parent1.chromosome[k_2]
                    for j in range(n):
                        if(parent2.chromosome[j] == V):
                            k_2=j
    
                    if z[k_2] == -1: 
                        break
                z[k_2] = elementToBeCopied
                visited[z[k_2]]=True      
        
        for i in range(n):
            if(z[i]==-1):
                z[i]=parent2.chromosome[i]

        #SEGUNDO HIJO
        for i in range(cp_1,top+1):
            if not (visited_2[parent1.chromosome[i]]):
                k_2 = i
                elementToBeCopied = parent1.chromosome[i]
                #Simulando el do - while 
                while True:
                    V = parent2.chromosome[k_2]
                    for j in range(n):
                        if(parent1.chromosome[j] == V):
                            k_2=j
    
                    if z_2[k_2] == -1: 
                        break
                z_2[k_2] = elementToBeCopied
                visited_2[z_2[k_2]]=True      
        
        for i in range(n):
            if(z_2[i]==-1):
                z_2[i]=parent1.chromosome[i]
        
        time_end = time.time()
        return time_end-time_start
    
    def cross(self,parent1, parent2):


        
        n = len(parent1.chromosome)
        #Paso 1
        #Puntos de corte
        cut_points= sorted(np.random.choice(np.arange(n),2,replace=False))
        cp_1, cp_2 = cut_points[0],cut_points[1]


        #visited = np.full(n+1,False,dtype=bool)
        visited = np.full(n+1,False,dtype=bool)
        visited_2 = np.full(n+1,False,dtype=bool)

        z = np.full(n,-1,dtype=int)
        z_2 = np.full(n,-1,dtype=int)

        top= cp_2

        if(cp_2==n):
            top=n-1
        
        for i in range(cp_1,top+1):
            z[i] = parent1.chromosome[i]
            visited[z[i]] = True

        for i in range(cp_1,top+1):
            z_2[i] = parent2.chromosome[i]
            visited_2[z_2[i]] = True

        #PRIMER HIJO  
        for i in range(cp_1,top+1):
            if not (visited[parent2.chromosome[i]]):
                k_2 = i
                elementToBeCopied = parent2.chromosome[i]
                #Simulando el do - while 
                while True:
                    V = parent1.chromosome[k_2]
                    for j in range(n):
                        if(parent2.chromosome[j] == V):
                            k_2=j
    
                    if z[k_2] == -1: 
                        break
                z[k_2] = elementToBeCopied
                visited[z[k_2]]=True      
        

        for i in range(n):
            if(z[i]==-1):
                z[i]=parent2.chromosome[i]

        #SEGUNDO HIJO
        for i in range(cp_1,top+1):
            if not (visited_2[parent1.chromosome[i]]):
                k_2 = i
                elementToBeCopied = parent1.chromosome[i]
                #Simulando el do - while 
                while True:
                    V = parent2.chromosome[k_2]
                    for j in range(n):
                        if(parent1.chromosome[j] == V):
                            k_2=j
    
                    if z_2[k_2] == -1: 
                        break
                z_2[k_2] = elementToBeCopied
                visited_2[z_2[k_2]]=True      
        
        for i in range(n):
            if(z_2[i]==-1):
                z_2[i]=parent1.chromosome[i]
        
        return np.array(z), np.array(z_2) 


    def debugged_cross(self,parent1, parent2):
        pass 

##>>>>>>>>EXPERIMENTAL NO FUNCIONA 
#Recuperado de https://github.com/Myrrthud/Implementation-of-IPMX-Genetic-Algorithm/blob/main/gaipmx.ipynb
class IMPXMYR(CrossoverOp):
    def timed_cross(self,parent1, parent2):
        pass
    def cross(self,parent1, parent2):

        parent1 = parent1.chromosome
        parent2 = parent2.chromosome

        print(parent1)
        print(parent2)
        n = len(parent2)

        # Step 1: Randomly select two cut points, cp1 and cp2, on parent chromosomes, parent1 and parent2.
        cp1, cp2 = sorted(rnd.sample(range(n), 2))
        

        primitive_offspring1 = parent1[cp1:cp2]
        primitive_offspring2 = parent2[cp1:cp2]

        # Step 2: Produce primitive-offspring1 and primitive-offspring2 by swapping the substrings between cp1 and cp2.
        # primitive_offspring1 = parent1.copy()
        # primitive_offspring2 = parent2.copy()

        # #Intercambiamos información de los padres con la generacion primitiva usando los puntos de corte
        # primitive_offspring1[cp1:cp2] = parent2[cp1:cp2]
        # primitive_offspring2[cp1:cp2] = parent1[cp1:cp2]

        # Step 3: Define an exchange list with respect to the chosen substrings.
        exchange_list = [-1] * n  # Initialize exchange_list with -1
        for i in range(cp1, cp2):
            exchange_list[parent2[i]] = parent1[i]

        # Step 4: Generate a directed graph of the exchange list.
        directed_graph = {i: [] for i in range(n)}
        for i in range(cp1, cp2):
            if i != cp1 and exchange_list[parent1[i]] != parent1[i]:
                directed_graph[parent1[i]].append(exchange_list[parent1[i]])
                
        
        # Step 5: Find all distinct paths between nodes in the graph.
        def find_paths(node, path=[]):
            path = path + [node]
            if node == primitive_offspring2[0]:
                return [path]
            paths = []
            for neighbor in directed_graph[node]:
                if neighbor not in path:
                    new_paths = find_paths(neighbor, path)
                    for new_path in new_paths:
                        paths.append(new_path)
            return paths
        
        # Step 6: For each path that contains more than two nodes, add an edge between two endpoints and then remove all mid nodes.
        def refine_paths(paths):
            refined_paths = []
            for path in paths:
                if len(path) > 2:
                    start, end = path[0], path[-1]
                    refined_paths.append((start, end))
            return refined_paths
        
        paths = find_paths(primitive_offspring1[0])
        refined_paths = refine_paths(paths)
        
        # Step 7: Update the exchange_list based on the refined paths
        for start, end in refined_paths:
            exchange_list[start] = end
        
        # Step 8: Apply the updated exchange_list to primitive_offspring1 to produce offspring1
        offspring1 = [exchange_list[elem] if exchange_list[elem] != -1 else elem for elem in primitive_offspring1]
        
        # Step 9: Generate F, a list of the same length as the parent chromosomes, initialized to zero
        F = [0] * n
        
        # Step 10: Produce offspring2 by performing the following operations:
        # (a) F[offspring1[i]] = parent1[i]
        # (b) offspring2[i] = F[parent2[i]]
        for i, elem in enumerate(offspring1):
            F[elem] = parent1[cp1 + i]
        
        offspring2 = [F[elem] for elem in parent2]
        
        return np.array(offspring1), np.array(offspring2)
    def debugged_cross(self,parent1, parent2):
        pass 