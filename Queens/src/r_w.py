


def write_knapsack_file(relative_path, data):
    '''
    Args: 
    relative_path : string 
        la ruta del archivo a escribir
    data : list : string 
        la lista de datos a escribir 
    
    '''
    #Generamos la ruta 
    #Ejemplo de ruta relativa : '/data/ejeL14n45.txt'
    path = os.getcwd()+relative_path

    with open(path,'w') as f:
        for da in data: 
            f.writelines(da)
