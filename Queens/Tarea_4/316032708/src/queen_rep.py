import numpy as np 
import math 

class Queen_Solution: 
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


	def __init__(self, c):
		self.chromosome = c
		self.max_conflics = (len(c)*(len(c)-1))/2 
		self.fitness = 0 
		self.board_rep = [ ["--" for i in self.chromosome] for j in self.chromosome]
		# 

	def __str__(self):
		return "Chromosome : {}".format(str(self.chromosome))+"\n"+"Fitness Value : {}".format(self.fitness)+"\n"+"Best possible fitness: "+str(self.max_conflics)+"\n"+"Board Representation :"+"\n"+str(self.set_board())

	def set_board(self): 
		'''
		Generate the visual representation of the individual (chromosome)
		'''
		for i in range(len(self.chromosome)):
			self.board_rep[i][self.chromosome[i]] = "R{}".format(i)
		board = ""
		for row in self.board_rep:
			board = board+str(row)+"\n"

		return board

	def evaluate(self): 

		'''
		Funcion que evalua los conflictos que tiene la solucion 
		'''

		#Conflicts 
		conflicts=0

		for i in range(len(self.chromosome)):
			for j in range(len(self.chromosome)):
				if(i!=j):
					if(abs(i-j) == abs(self.chromosome[i]-self.chromosome[j])):
						conflicts=conflicts+1

		self.fitness = self.max_conflics-conflicts/2

