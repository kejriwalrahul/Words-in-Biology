"""
	Cluster proteins based on MDL based similarity metric and assign 'Brown' code

	Rahul Kejriwal
	CS14BO23
"""

# Python Imports
from tqdm import tqdm
import numpy as np

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
	return max(p1_dl+p2_dl-combined_dl, 0)


if __name__ == '__main__':

	# Read and prep data
	data_2_05 = read_proteins("../Data/Data/astral-scope-95-2.05.fa")
	train_data = prepare_data(data_2_05)

	# Select out few samples
	np.random.shuffle(train_data)
	train_data = train_data[:10]

	# Build similarity score matrix
	sim = np.zeros( (len(train_data), len(train_data)) )
	for i in tqdm(range(len(train_data))):
		for j in tqdm(range(i, len(train_data))):
			sim[i,j] = similarity(train_data[i][0], train_data[j][0])
			sim[j,i] = sim[i,j]

	# Save matrix and samples
	with open('sim_matrix.npy','w') as fp, open('samples.npy','w') as fs:
		np.save(fs, train_data)
		np.save(fp, sim)