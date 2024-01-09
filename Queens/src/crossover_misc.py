    # def find_replacement(self,elemnt,mapp_rel, tabu):
    # 		#La lista tabu es una lista de 0/1 del tamanio de n 
    # 		# Si 
    # 		if tabu[mapp_rel[elemnt]] == 1 :
    # 			print("Este es el reemplazo "+str(elemnt))
    # 			return elemnt
    # 		else:
    # 			tabu[mapp_rel[elemnt]] = 1
    # 			tabu[elemnt] = 1
    # 			elemnt = mapp_rel[elemnt]
    # 			return self.find_replacement(elemnt,mapp_rel,tabu)
            
    # 	def pmx_generate_mapping_relationship(self, subs_str_1,subs_str_2 ):
    # 		#Sabemos que ambas cadenas tienen la misma longitud
    # 		if len(subs_str_1) != len(subs_str_2): 
    # 			raise ValueError("La longitud de las subcadenas es distinta, no se puede realizar la relacion de mapeo")
            
    # 		#La relacion de mapeo va a ser una lista de tuplas (x,y) donde tambien existe (y,x)
    # 		mapping_r = []
    # 		mapping_ocurrences = [0 for x in range(self.n_queens)]
            
    # 		#Aqui hay que tener cuidado, puede ser que la tupla (x,y) y (y,x) ya se encuentre y pueda agregarse de nuevo, por lo que hay que revisar que no esté ya en la relacion de intercambio 

    # 		for i in range(len(subs_str_1)): 
                
    # 			if(subs_str_1[i]!=subs_str_2[i]):#EXPERIMENTAL -> Aqui estamos discriminando las tuplas [x,x], eso funciona ? 
    # 				mapping_r.append([subs_str_1[i], subs_str_2[i]])
    # 				mapping_r.append([subs_str_2[i], subs_str_1[i]])
    # 				mapping_ocurrences[subs_str_1[i]] = 1
    # 				mapping_ocurrences[subs_str_2[i]] = 1
                
    # 		#Tambien buscamos una forma directa para saber si un elemento esta en la relacion de mapeo

    # 		return np.array(mapping_r), np.array(mapping_ocurrences)
            
    # 	def pmx_replacement(self,elemt, mapp_rel, end):
    # 		#Esta funcion recursiva es lineal ya que no necesita resolver las llamadas anteriores para regresar 
    # 		#el valor 

    # 		#Se tiene que pasar una copia de la relacion de mapeo
    # 		if len(mapp_rel) == 0 : 
    # 			#print(end)
    # 			return end
            
    # 		if mapp_rel[0][0] == elemt:
    # 			end = mapp_rel[0][1] #El reemplazo es el elemento de la tupla 
    # 			#Borramos tanto el indice 0 y 1 por que sabemos que el elemento buscado esta en esas tuplas
    # 			mapp_rel = np.delete(mapp_rel,0,axis=0)
    # 			mapp_rel = np.delete(mapp_rel,0,axis=0)
    # 			self.pmx_replacement(end,mapp_rel,end)
    # 		elif mapp_rel[0][1] == elemt :
    # 			end = mapp_rel[1][0] #El reemplazo es el elemento de la tupla 
    # 			#Borramos tanto el indice 0 y 1 por que sabemos que el elemento buscado esta en esas tuplas
    # 			mapp_rel = np.delete(mapp_rel,0,axis=0)
    # 			mapp_rel = np.delete(mapp_rel,0,axis=0)
    # 			self.pmx_replacement(end,mapp_rel,end)

    # 		else: #Son elementos que no nos interesan
    # 			mapp_rel = np.delete(mapp_rel,0,axis=0)
    # 			mapp_rel = np.delete(mapp_rel,0,axis=0)
    # 			self.pmx_replacement(elemt,mapp_rel,end)
            
    # 		return end