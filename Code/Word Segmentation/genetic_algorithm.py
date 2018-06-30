import numpy as np
import random

pop_size = 100
p_cross = 0.8
p_mut = 0.1
num_iter = 100
population = []
global_max = []
max_fitness = 100000
rand_par1 = 1
rand_par2 = 10

string = "Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and other animals. In computer science AI research is defined as the study of intelligent agents any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term artificial intelligence is applied when a machine mimics cognitive functions that humans associate with other human minds, such as learning and problem solving. The scope of AI is disputed as machines become increasingly capable, tasks considered as requiring intelligence are often removed from the definition, a phenomenon known as the AI effect, leading to the quip, AI is whatever hasn't been done yet. For instance, optical character recognition is frequently excluded from artificial intelligence, having become a routine technology. Capabilities generally classified as AI as of 2017 include successfully understanding human speech, competing at the highest level in strategic game systems, autonomous cars, intelligent routing in content delivery network and military simulations. Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding followed by new approaches, success and renewed funding. For most of its history, AI research has been divided into subfields that often fail to communicate with each other. These sub-fields are based on technical considerations, such as particular goals the use of particular tools, or deep philosophical differences. Subfields have also been based on social factors.x"

def generate_individual(string):
	n = len(string)
	i = 0
	j = random.randint(rand_par1,rand_par2)
	segs = []
	while j+i < n:
		segs.append(j+i)
		i = j+i
		j = random.randint(rand_par1,rand_par2)
	
	segs = np.array(segs)
	return segs

def get_segments(string,indices):
	segments = []
	segments.append(string[:indices[0]])
	for i in range(0,len(indices)-1):
		segments.append(string[indices[i]:indices[i+1]])
	segments.append(string[indices[len(indices)-1]:])
	segments = np.array(segments)
	return segments
	
def fitness_function(individual, string, flag):
	
	segmentation = get_segments(string,individual)
	
	tok_counts = {}
	for tok in segmentation:
		tok_counts[tok] = 1 + tok_counts.get(tok, 0)
		
	codebook = ''.join(tok_counts.keys())
	codebook_counts = {}
	for tok in codebook:
		codebook_counts[tok] = 1 + codebook_counts.get(tok, 0)
		
	codebook_res = 0
	for tok, count in codebook_counts.iteritems():
		codebook_res += count * np.log(float(count) / len(codebook))
		
	res = 0
	for tok, count in tok_counts.iteritems():
		res += count * np.log( float(count) / len(segmentation) )
	
	if flag:
		print -res
		print -codebook_res

	return -res + -2*codebook_res 
	
	
	

if __name__ == '__main__':
	for i in range(0,pop_size):
		population.append(generate_individual(string))
	#population.append(np.array([10, 22, 24, 36, 48, 50, 59, 61, 69, 71, 74, 81, 93, 102, 104, 110, 113, 118, 126, 128, 136, 143, 145, 153, 155, 162, 164, 167, 172, 174, 185, 191, 194, 200, 204, 213, 216, 227, 230, 235, 242, 246, 254, 257, 263, 265, 277, 286, 289, 295, 308, 311, 315, 325, 337, 339, 346, 350, 351, 358, 364, 373, 382, 386, 392, 401, 405, 410, 415, 421, 425, 427, 435, 438, 445, 453, 456, 461, 463, 465, 467, 475, 477, 485, 491, 503, 511, 516, 526, 528, 537, 549, 552, 557, 564, 568, 571, 582, 583, 593, 598, 600, 603, 605, 612, 619, 621, 624, 629, 631, 633, 641, 647, 651, 655, 659, 662, 671, 678, 687, 698, 700, 710, 718, 722, 732, 745, 751, 757, 758, 765, 776, 788, 797, 807, 809, 811, 813, 815, 819, 826, 838, 851, 856, 863, 872, 874, 877, 884, 889, 891, 900, 904, 912, 922, 927, 938, 945, 947, 954, 962, 969, 972, 980, 992, 1002, 1014, 1017, 1024, 1026, 1028, 1036, 1046, 1048, 1053, 1056, 1058, 1061, 1066, 1071, 1074, 1085, 1092, 1097, 1099, 1108, 1116, 1118, 1132, 1135, 1138, 1142, 1144, 1151, 1159, 1161, 1164, 1175, 1182, 1185, 1192, 1200, 1203, 1207, 1209, 1212, 1220, 1222, 1230, 1233, 1237, 1244, 1248, 1257, 1261, 1266, 1270, 1272, 1283, 1287, 1291, 1297, 1302, 1312, 1315, 1320, 1322, 1331, 1346, 1350, 1352, 1362, 1367, 1370, 1373, 1375, 1385, 1391, 1393, 1397, 1410, 1422, 1431, 1435, 1439, 1443, 1448, 1450, 1456]))
	
	max_fit = population[0]
	
	for i in range(0,num_iter):
		fitness = []
		for j in range(0,pop_size):
			fitness.append(fitness_function(population[j],string,False))
		
		fitness = np.array(fitness)
		avg_fitness = fitness.sum()/pop_size
		
		print i,avg_fitness
		
		print sum([item.shape[0] for item in population])/100.0
		print "###"
		max_fit = population[np.argmin(fitness)]
		
		worst = population[np.argmax(fitness)]
		#print worst,fitness_function(worst,string,True)
		if fitness_function(max_fit,string,False) < max_fitness:
			max_fitness = fitness_function(max_fit,string,True)
			max_fit = population[np.argmin(fitness)]
			
			
			#print get_segments(string,max_fit)
			#print max_fit	
			#print max_fitness
		
		#print fitness
		fitness = fitness/1000
		fitness = 1/fitness
		#print fitness
		fitness = fitness/(fitness.sum())
		#print fitness

		for j in range(0,fitness.shape[0]):
			if j != fitness.shape[0]-1:
				fitness[j+1] += fitness[j]
		#print fitness
		
		########################## Crossover
		
		crossed = []
		for j in range(0,pop_size,2):
			curr = random.uniform(0,1)
			for k in range(0,len(fitness)):
				if fitness[k] > curr:
					break
			p1 = population[k]
			curr = random.uniform(0,1)
			for k in range(0,len(fitness)):
				if fitness[k] > curr:
					break
			p2 = population[k]
			
			try:

				index1 = np.random.randint(0,p1.shape[0]-1)
				index2 = np.random.randint(0,p2.shape[0]-1)
				p = random.uniform(0,1)
				if p < p_cross:
					c1 = np.append(p1[:index1],p2[index2:])
					c1 = np.unique(c1)
					c1 = np.sort(c1)
				
					c2 = np.append(p2[:index2],p1[index1:])
					c2 = np.unique(c2)
					c2 = np.sort(c2)
				
				else:
					c1 = p1
					c2 = p2
			
			except:
				c1 = generate_individual(string)
				c2 = generate_individual(string)		


			crossed.append(c1)
			crossed.append(c2)	
		
		#######################
		
		#### Mutation operation in the genetic algorithm #########
		"""
		for j in range(0,len(crossed)):
			curr = random.uniform(0,1)
			if curr < p_mut:
				#print "Mutated"
				p1 = crossed[j]
				#print crossed[j]
				index = np.random.randint(0,p1.shape[0]-1)
				val = random.uniform(0,1)
				#if val < 0.5:
					# Remove an index
					#p1 = np.delete(p1,[index])
				#else:
				#	p1 = np.append(p1,np.array([index]))
				#	p1 = np.unique(p1)
				#	p1 = np.sort(p1)
				crossed[j] = p1
				#print crossed[j]
				#print "End of mutation"
		"""
		population = crossed
########################################################
print max_fit
print fitness_function(max_fit,string,True)
print get_segments(string,worst)
print get_segments(string,max_fit)		
