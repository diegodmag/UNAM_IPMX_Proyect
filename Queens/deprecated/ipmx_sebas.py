import random
import time

def pmx(parent1, parent2, start_cut, end_cut):
    # Paso 1
    # Asegurarse de que start_cut y end_cut son 0-indexados
    start_cut -= 1
    end_cut -= 1


    # Paso 2: Intercambiar substrings entre puntos de corte
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    offspring1[start_cut:end_cut+1], offspring2[start_cut:end_cut+1] = offspring2[start_cut:end_cut+1], offspring1[start_cut:end_cut+1]

    # Paso 3: Determinar la relación de mapeo con respecto a las substrings elegidas
    mapping1 = {}
    mapping2 = {}
    for i in range(start_cut, end_cut+1): # Para cada valor en nuestra area de corte ligalo con su reemplazo
        mapping1[parent2[i]] = parent1[i]
        mapping2[parent1[i]] = parent2[i]

    # Paso 4: Legalizar offsprings primitivos usando la relación de mapeo
    for i in range(0, start_cut):
        while offspring1[i] in mapping1:
            offspring1[i] = mapping1[offspring1[i]]

        while offspring2[i] in mapping2:
            offspring2[i] = mapping2[offspring2[i]]

    for i in range(end_cut+1, len(offspring1)):
        while offspring1[i] in mapping1:
            offspring1[i] = mapping1[offspring1[i]]

        while offspring2[i] in mapping2:
            offspring2[i] = mapping2[offspring2[i]]


    return offspring1, offspring2

def restaUno(array):
    return list(map(lambda x: x - 1, array))

def ipmx(parent1, parent2, start_cut, end_cut):
    # Asegurarse de que start_cut y end_cut son 0-indexados
    start_cut -= 1
    end_cut -= 1

    # Inicialización
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()
    print("P1"+str(restaUno(offspring1)))
    print("P1"+str(restaUno(offspring2)))
    # Paso 2
    for i in range(start_cut, end_cut+1):
        offspring1[i] = parent2[i]
        offspring2[i] = parent1[i]

    print("POffspring 1"+str(restaUno(offspring1)))
    print("POffspring 2"+str(restaUno(offspring2)))
    # Paso 3 & 4
    exchange_list = [offspring1[start_cut:end_cut+1], offspring2[start_cut:end_cut+1], [1 for _ in range(end_cut-start_cut+1)]]
    #exchange_list = [offspring2[start_cut:end_cut+1], offspring1[start_cut:end_cut+1], [1 for _ in range(end_cut-start_cut+1)]]
    print("Initial exchange list ")
    print(restaUno(exchange_list[0]))
    print(restaUno(exchange_list[1]))
    print(exchange_list[2])

    
    #print("Initial exchange list "+str(restaUno(exchange_list)))

    # Paso 5
    guide_list = [0]*len(parent1)
    for i in range(len(exchange_list[0])):
        guide_list[exchange_list[0][i]-1] = exchange_list[1][i]
    print("Initial guide List "+str(restaUno(guide_list)))

    l1 = [0]*len(parent1)
    l2 = [0]*len(parent1)
    sum_1_2 = [0]*len(parent1)

    for value in exchange_list[0]:
        l1[value-1] = 1

    for value in exchange_list[1]:
        l2[value-1] = 1

    for i in range(len(sum_1_2)):
        sum_1_2[i] = l1[i] + l2[i]

    print("L1 :"+str(l1))
    print("L1 :"+str(l2))
    print("L1 + L2:"+str(sum_1_2))
    # Paso 6 & 7
    # Actualizar exchange_list con 0's
    for row in range(len(exchange_list[0])):
        index = exchange_list[0][row]-1
        if sum_1_2[index] == 2:
            exchange_list[2][row] = 0
    #print("Updated exchange list with 0's"+str(restaUno(exchange_list)))
    print("Updated 1 (pone ceros) exchange list ")
    print(restaUno(exchange_list[0]))
    print(restaUno(exchange_list[1]))
    print(exchange_list[2])
    # Actualizar exchange_list con nuevos caminos
    for row in range(len(exchange_list[0])):
        if exchange_list[2][row] == 1:
            last_checked = exchange_list[1][row]
            last_known = 0
            while last_checked != 0: # Siempre verdadero al menos una vez
                last_known = last_checked
                last_checked = guide_list[last_checked-1]
            exchange_list[1][row] = last_known

    #print("Updated exchange list with new rows"+str(restaUno(exchange_list)))
    print("Updated 2 (nuevos caminos) exchange list ")
    print(restaUno(exchange_list[0]))
    print(restaUno(exchange_list[1]))
    print(exchange_list[2])

    # Actualizar guide_list
    for i in range(len(exchange_list[0])):
        guide_list[exchange_list[0][i]-1] = exchange_list[1][i]
    print("Updated guide list with new rows"+str(restaUno(guide_list)))

    # Paso 8
    for i in range(start_cut):
        value = guide_list[parent1[i]-1]
        if value != 0:
            offspring1[i] = value

    for i in range(end_cut+1, len(parent1)):
        value = guide_list[parent1[i]-1]
        if value != 0:
            offspring1[i] = value

    print("Offspring 1 Legalized"+str(restaUno(offspring1)))

    # Paso 9
    F = [0]*len(parent1)
    for i in range(len(F)):
        F[offspring1[i]-1] = parent1[i]

    # Paso 10
    for i in range(len(F)):
        offspring2[i] = F[parent2[i]-1]

    return offspring1, offspring2

def testExamples():
    parent1 = [11,5,12,6,9,1,4,2,13,10,8,3,7]
    parent2 = [1,2,8,7,4,3,6,13,10,12,5,9,11]
    start_cut = 7
    end_cut = 11
    offs1 = [11, 8, 2, 4, 9, 1, 6, 13, 10, 12, 5, 3, 7]
    offs2 = [1, 12, 5, 7, 6, 3, 4, 2, 13, 10, 8, 9, 11]
    # test(parent1, parent2, start_cut, end_cut, offs1, offs2)

    test1 = [5,10,2,3,1,6,7,9, 8, 4]
    offs1 = [7,10,1,2,3,5,6,8, 9, 4]

    F     = [2, 3,1,4,6,7,5,9, 8,10]
    test2 = [7, 4,1,2,3,5,6,8,10, 9]
    offs2 = [5, 4,2,3,1,6,7,9,10, 8]
    start_cut = 3
    end_cut = 8
    test(test1, test2, start_cut, end_cut,offs1,offs2)

def test(parent1, parent2, start_cut, end_cut, expected_offspring1, expected_offspring2):

    offsA, offsB = ipmx(parent1, parent2, start_cut, end_cut)
    print("IPMX")
    print("offs1 correct = ", offsA == expected_offspring1)
    print("Given: ", offsA)
    print("Expected: ", expected_offspring1)
    print("offs2 correct = ", offsB == expected_offspring2)
    print("Given: ", offsB)
    print("Expected: ", expected_offspring2)

    offsC, offsD = pmx(parent1, parent2, start_cut, end_cut)
    print("PMX")
    print("offs1 correct = ", offsC == expected_offspring1)
    print("Given: ", offsC)
    print("Expected: ", expected_offspring1)
    print("offs2 correct = ", offsD == expected_offspring2)
    print("Given: ", offsD)
    print("Expected: ", expected_offspring2)

    print("PMX = IPMX")
    print((offsA == offsC) & (offsB == offsD))


def makeRandomCases(size):
    parent1 = [i for i in range(1,size+1)]
    parent2 = parent1.copy()

    random.shuffle(parent1)
    random.shuffle(parent2)
    return parent1, parent2

if __name__ == '__main__':
    print("Generating random parent cases: ")

    size = 10
    #parent1, parent2 = makeRandomCases(size)
    parent1,parent2 = [6, 2, 1 ,3 ,7 ,5, 4, 8], [5, 1, 8, 2, 4, 7, 6, 3]

    print("Using:")
    print(parent1)
    print(parent2)

    # start_cut = random.randint(1,size / 2-1)
    # end_cut = random.randint(size/2+1,size)

    start_cut = 1
    end_cut = 3

    print("Cutting (M): ", end_cut - start_cut+1)
    print("From: ",start_cut)
    print("To: ", end_cut)

    print("Comparing:")
    print("ipmx")

    i_start_time = time.time()
    i_cpu_start = time.process_time()
    offspring1, offspring2 = ipmx(parent1, parent2, start_cut, end_cut)
    i_cpu_end = time.process_time()
    i_end_time = time.time()
    # print("Offspring1: ", offspring1)
    # print("Offspring2: ", offspring2)

    # i_ellapsed = i_end_time - i_start_time
    # i_cpu_used = i_cpu_end - i_cpu_start
    # print("Took: ", i_ellapsed, " seconds. Used: ", i_cpu_used, " seconds of CPU execution")

    # print("pmx")

    # start_time = time.time()
    # cpu_start = time.process_time()
    # offspring1, offspring2 = pmx(parent1, parent2, start_cut, end_cut)
    # cpu_end = time.process_time()
    # end_time = time.time()
    # print("Offspring1: ", offspring1)
    # print("Offspring2: ", offspring2)
    # ellapsed = end_time - start_time
    # cpu_used = cpu_end - cpu_start
    # print("Took: ", ellapsed, " seconds. Used: ", cpu_used, " seconds of CPU execution")

    # print("ipmx took: ", i_ellapsed - ellapsed, " more than pmx")
