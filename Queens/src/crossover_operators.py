from abc import ABC, abstractmethod
import math 
import numpy as np 
import random as rnd 
import copy

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
    
    def debugged_cross(self,parent1, parent2):
        pass

class PMX(CrossoverOp): 
    

    '''
        Operador de cruza Partially Mapped Crossover 
    '''

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

       
class IMPX(CrossoverOp):
    
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
        for i in range(m):
            guide_list[exchange_list[i,0]]=exchange_list[i,1]

        l1 = np.zeros(n,dtype=int)
        l2 = np.zeros(n,dtype=int)

        for i in range(m):
            l1[exchange_list[i,0]] = 1
            l2[exchange_list[i,1]] = 1
        

        l1_l2 = l1+l2

        #Actualizar la exchange list haciendo los nodos intermedios 0 
        for i in range(m):
            index = exchange_list[i,0]
            if l1_l2[index] == 2: 
                exchange_list[i,2] = 0
        
        #Actualizar exchange list eliminando nodos intermedios y conectado los nodos con camino directo 
        for i in range(m):
            if exchange_list[i,2] == 1: 
                #El reemplazo actual 
                current_replacement = exchange_list[i,1]
                #El reemplazo final
                final_replacement = -1
                while current_replacement != -1:#Al menos se cumple la primera vez 
                    final_replacement = current_replacement
                    current_replacement = guide_list[final_replacement]

                exchange_list[i,1] = final_replacement

        #Ahora actualizamos la guide list   
        for i in range(m):
            if(exchange_list[i,2]==1):
                guide_list[exchange_list[i,0]] =  exchange_list[i,1]
            else:
                guide_list[exchange_list[i,0]] = -1

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

