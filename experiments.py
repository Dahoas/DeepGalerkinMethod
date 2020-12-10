import h1d_parameterized
import h2d_parameterized 

for num_layers in range(1,11):
	for nodes_per_layer in range(2,6):
		for nSim_interior in range(1,6):
			for nSim_bound in range(1,6):
				for nSim_initial in range(1,6):
					name = str(num_layers) + "-" + str(nodes_per_layer) + "-" +  str(nSim_interior) +"-" +  str(nSim_bound) +"-" +  str(nSim_initial)
					print("Starting " + name)
					h1d_parameterized.run_heat_1d(10*nodes_per_layer,num_layers,500*nSim_interior,50*nSim_bound,50*nSim_initial,name)
					h2d_parameterized.run_heat_2d(10*nodes_per_layer,num_layers,500*nSim_interior,50*nSim_bound,50*nSim_initial,name)

#chanegs in dim, boundary, and initial