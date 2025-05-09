from abc import ABC, abstractmethod
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
class PermutationBasedProblem(ABC):
    
    '''
    Super clase que modela los metodos necesarios para los problemas 
    con representaciones basadas en permutaciones 
    
    '''
    @abstractmethod
    def __init__(self,permutation):
        self.chromosome = permutation
        self.optimal = 0
        pass 

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def get_instance(self,permutation):
        pass 

    @abstractmethod
    def get_population(self, chromosomes):
        pass 

    @abstractmethod
    def output_plot(self):
        pass 
    

    #For plot the visualizacion
    def get_path_for_output(self,figure_name):

        current_address = os.path.dirname(os.path.abspath(__file__))
        output_address = os.path.join(current_address,'..','output')
        
        if not os.path.exists(output_address):
            os.makedirs(output_address)
        
        return os.path.join(output_address,figure_name)


class NQueens(PermutationBasedProblem):
    '''
	Class modeling a Solution for the n-queens problems. 

	Attributes : 
	chromosome : lits [int]
		The chromosome is a permutation of n
	fitness : int 
		The fitness value for the solution, we aim to minimize the value in order to obtain the optimal solution (which has fitness value 0)
	board_rep : string 
		A visual representation of the queen's positions 
	'''
    def __init__(self, permutation):
        super().__init__(permutation)
        self.fitness=0
        self.board_rep = ""
        self.optimal = 0 

    def __str__(self):
        return "Chromosome : {}".format(str(self.chromosome))+"\n"+"Fitness Value : {}".format(self.fitness)+"\n"+"Board Representation :"+"\n"+str(self.set_board())

    # def evaluate(self):
    #     '''
	# 	El valor objetivo de una solución es la cantidad de conflictos, por lo que mientras menos conflictos mejor solución es 
	# 	'''
    #     conflicts=0
    #     for i in range(len(self.chromosome)):
    #         for j in range(i+1,len(self.chromosome)):
    #             if(abs(i-j) == abs(self.chromosome[i]-self.chromosome[j])):
    #                 conflicts=conflicts+1
        
    #     self.fitness = conflicts
    
    def evaluate(self):
        diagonal1 = [0] * (2*len(self.chromosome) - 1)
        diagonal2 = [0] * (2*len(self.chromosome) - 1)

            # Contar las ocurrencias en cada diagonal
        for i in range(len(self.chromosome)):
            diagonal1[i+self.chromosome[i]] += 1
            diagonal2[len(self.chromosome)-i+self.chromosome[i]-1] += 1

            # Calcular los conflictos
        conflicts = 0
        for i in range(2*len(self.chromosome) - 1):
            if diagonal1[i] > 1:
                conflicts += diagonal1[i] - 1
            if diagonal2[i] > 1:
                conflicts += diagonal2[i] - 1

        self.fitness = conflicts


    def get_instance(self,chromosome):
        return NQueens(chromosome)
         
    def get_population(self, chromosomes):
        nqueens_pop = []
        for chrom in chromosomes:
            nqueens_pop.append(NQueens(chrom))
        return np.array(nqueens_pop)

    def set_board(self):

        self.board_rep= [ ["--" for i in self.chromosome] for j in self.chromosome]
        '''
        Generate the visual representation od the individual 
        ''' 
        for i in range(len(self.chromosome)):
            self.board_rep[i][self.chromosome[i]] = "R{}".format(i)
        board = ""
        for row in self.board_rep:
            board = board+str(row)+"\n"

        return board

    def output_plot(self):
        self.plot_queens()
        
    def plot_queens(self):
        n = len(self.chromosome)
        chessboard = np.zeros((n, n), dtype=int)

        # Place queens on the chessboard
        for row, col in enumerate(self.chromosome):
            chessboard[row][col] = 1

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw the chessboard and place queens
        for i in range(n):
            for j in range(n):
                is_white = (i + j) % 2 == 0
                color = 'white' if is_white else 'black'
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
                queen_color = 'black' if is_white else 'white'  # Opposite color for the queen
                if chessboard[i][j] == 1:
                    queen_symbol = u'\u265B'  # Unicode symbol for the queen (♛)
                    ax.text(j + 0.5, i + 0.5, queen_symbol, fontsize=24, ha='center', va='center', color=queen_color)

        # Check for diagonal conflicts and draw lines between conflicting queens
        for i in range(n):
            row_i = i
            col_i = self.chromosome[i]
            for j in range(i + 1, n):
                row_j = j
                col_j = self.chromosome[j]
                if abs(row_i - row_j) == abs(col_i - col_j):
                    # Queens are on the same diagonal; draw a red line between them
                    ax.plot([col_i + 0.5, col_j + 0.5], [row_i + 0.5, row_j + 0.5], color='red', linewidth=2)

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

        plt.savefig(self.get_path_for_output(f'Board{n}'))
        plt.close()
        #plt.show()  # Uncomment if you want to display the plot
