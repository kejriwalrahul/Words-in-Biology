"""
	Cluster proteins based on MDL based similarity metric and assign 'Brown' code

	Rahul Kejriwal
	CS14BO23
"""

# Python Imports
from tqdm import tqdm

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
	return (p1_dl+p2_dl-combined_dl) 


if __name__ == '__main__':
	data_2_05 = read_proteins("../Data/Data/astral-scope-95-2.05.fa")
	data_2_06 = read_proteins("../Data/Data/astral-scope-95-2.06.fa")
	data_2_06 = {
		'A': data_2_06['A'].difference(data_2_05['A']),
		'B': data_2_06['B'].difference(data_2_05['B']),
		'C': data_2_06['C'].difference(data_2_05['C']),
		'D': data_2_06['D'].difference(data_2_05['D']),
	}

	train_data = prepare_data(data_2_05)
	test_data  = prepare_data(data_2_06)

	for i in range(100):
		print train_data[0][1], train_data[i][1], similarity(train_data[0][0], train_data[i][0])