"""
	Cluster proteins based on MDL based similarity metric and assign 'Brown' code

	Rahul Kejriwal
	CS14BO23
"""

# Python Imports
from tqdm import tqdm
import sys
from multiprocessing import Pool
import numpy as np
from scipy.cluster.hierarchy import fclusterdata

# Custom Imports
from read_data import read_proteins, prepare_data
from dlg import DLGainSegmentor


"""
	MDL based similarity metric
"""
def similarity(protein_1, protein_2):
	p1_dl = DLGainSegmentor(protein_1).opt_dl()
	p2_dl = DLGainSegmentor(protein_2).opt_dl()
	segmentor = DLGainSegmentor(protein_1+protein_2)
	# print segmentor.segment()
	combined_dl = segmentor.opt_dl()
	return max(p1_dl+p2_dl-combined_dl, 0) / min(len(protein_1), len(protein_2))


def dist(p1, p2):
	return 1 - dist.similarity[p1[0],p2[0]]


def process_queue_batch(args):
	batch,k = args
	sim = {}
	for i,j in tqdm(batch, position=k):
		sim[(i,j)] = similarity(train_data[i][0], train_data[j][0])
		sim[(j,i)] = sim[(i,j)]
	return sim


if __name__ == '__main__':

	num_threads = int(sys.argv[1])

	# Read and prep data
	data_2_05 = read_proteins("../Data/Data/astral-scope-95-2.05.fa")
	train_data = prepare_data(data_2_05)

	# Select out few samples
	np.random.shuffle(train_data)
	train_data = train_data[:500]

	# Build similarity score matrix
	sim = np.zeros( (len(train_data), len(train_data)) )
	work_queue = [(i,j) for i in range(len(train_data)) for j in range(i, len(train_data))]
	work_per_thread = int(np.ceil(float(len(work_queue)) / num_threads))

	# Parallely execute load
	p = Pool(num_threads)
	res = p.map(process_queue_batch, [(work_queue[i*work_per_thread: (i+1)*work_per_thread],i) for i in range(num_threads)])

	# Merge results
	for d in res:
		for i,j in d:
			sim[i,j] = d[(i,j)]

	# Save matrix and samples
	with open('tmp/sim_matrix.npy','w') as fp, open('tmp/samples.npy','w') as fs:
		np.save(fs, train_data)
		np.save(fp, sim)

	# # Cluster the samples
	# X = np.array(list(range(len(train_data)))).reshape( (-1,1) )
	# dist.similarity = sim
	# c = fclusterdata(X, 4, metric=dist)
