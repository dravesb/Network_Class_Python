#-----------------------------------------------
#
# 	Set up Class for Networks
# 		
#-----------------------------------------------

#----------------------------
#	Import libraries
#----------------------------

import numpy as np
import matplotlib.pyplot as plt
import math

class Graph:

	#constructor 
	def __init__(self, A): 
		#set adjacency matrix
		self.A = A

	#---------------------
	#	summary methods
	#---------------------
	
	def vertex_count(self):
		Nv = self.A.shape[0]
		return(Nv)

	def edge_count(self):
		
		def f(x):
			if x==0:
				return(0)
			else:
				return(1)


		vectorized_f = np.vectorize(f)
		mat = vectorized_f(self.A)

		Ne = mat.sum()/2
		return(Ne)

	def density(self):
		Nv = self.vertex_count()
		return(round(self.edge_count() / (Nv *(Nv -1)/2), 2))

	#------------------------
	#	Embedding of A
	#------------------------

	#set up ASE function
	def __ase(self, d = 2):
		#get eigenvalues and eigenvectors
		U, s, V = np.linalg.svd(self.A)

		#identify top d eigenvalues
		ind = np.argsort(-s)[list(range(0, d))]

		#normalize columns function
		def norm_x(x): 
			return x/np.linalg.norm(x)
		
		#store in U and S^{1/2}
		S = np.diag(list(map(math.sqrt, s[ind])))
		U = np.apply_along_axis(norm_x, 0, U[:,ind])
		
		#return ASE
		return np.matmul(U, S)

	#set up LSE function
	def __lse(self, d = 2):

		#set up Laplacian
		D = np.matmul(self.A, np.ones(shape = (self.A.shape[1], 1)))
		L = D - self.A

		#get eigenvalues and eigenvectors
		U, s, V = np.linalg.svd(L)

		#identify top d eigenvalues
		ind = np.argsort(-s)[list(range(0, d))]

		#normalize columns function
		def norm_x(x): 
			return x/np.linalg.norm(x)
		
		#store in U and S^{1/2}
		S = np.diag(list(map(math.sqrt, s[ind])))
		U = np.apply_along_axis(norm_x, 0, U[:,ind])
		
		#return ASE
		return np.matmul(U, S)

	#------------------------
	#	Viusalize Embeddings
	#------------------------

	def plot_ase(self, col = 'red'):

		#get embedding
		Xhat = self.__ase()

		#get top two coordinates
		x = Xhat[:,0]
		y = Xhat[:,1]

		#plot vectors
		plt.scatter(x, y, color = col, alpha=0.5)
		plt.title('ASE')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()

	def plot_lse(self, col = 'red'):

		#get embedding
		Xhat = self.__lse()

		#get top two coordinates
		x = Xhat[:,0]
		y = Xhat[:,1]

		#plot vectors
		plt.scatter(x, y, color = col, alpha=0.5)
		plt.title('LSE')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()	




