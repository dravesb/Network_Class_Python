#-----------------------------------------------
#
# 				Examples
# 		
#-----------------------------------------------

#source file
exec(open('./graph_class.py').read())
exec(open('../../SBM_GMM_Simulation/scripts/basic_functions.py').read())

#----------------------
# Set model parameters
#----------------------
n = 100
p = .5 

#----------------------
# Set up adjacency
#----------------------

#set up block matrix 
B = np.array([[.4, .1], 
			  [.1, .2]]) # core-periphery structure

#set group sizes
n = 100 
P = np.kron(B, np.ones(shape = (n, n)))

#sample P 
A = sampP(P)

#----------------------
# initialize object
#----------------------
graph = Graph(A)
print(graph.density())

#----------------------
# plot embeddings
#----------------------
c = np.concatenate([np.repeat('red', n), np.repeat('blue', n)])

graph.plot_ase(col = c)
graph.plot_lse(col = c)
