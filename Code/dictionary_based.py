from read_data import *
import operator
import cPickle
import numpy as np

max_len = 4
thres = 100

def get_features(data):

	big_dict = {}

	for i in range(1,max_len+1):
		curr_dict = {}
		for key in data.keys():
			for item in data[key]:
				string = item[1]
				n = len(string)
				for j in range(0,n+1-i):
					if string[j:j+i] in curr_dict:
						curr_dict[string[j:j+i]] += 1
					else:
						curr_dict[string[j:j+i]] = 1
	
		#print curr_dict
		if len(curr_dict.keys()) < thres:
			big_dict[i] = curr_dict
		else:
			sorted_dict = sorted(curr_dict.items(), key=operator.itemgetter(1))
			sorted_dict.reverse()
			sorted_dict = sorted_dict[:thres]
			sorted_dict = dict(sorted_dict)
			big_dict[i] = sorted_dict
	return big_dict
	
def segment(features,sequence,position,segNum,wordLen,prod,N):
	if position >= N:
		return -1
	if position == N-1:
		segNum[position] = 1
		wordLen[position] = 1
		prod[position] = np.log(features[1][sequence[N-1]])
		return segNum[position]
	if segNum[position] != 0:
		return segNum[position]
	
	len_a = np.zeros(max_len)
	num_a = np.zeros(max_len)
	
	for i in range(0,max_len):
		if sequence[position:position+i+1] in features[i+1]:
			len_a[i] = i+1
			num_a[i] = 1 + segment(features,sequence,position+i+1,segNum,wordLen,prod,N)
	min_val = min([k for k in num_a if k > 0])
	
	max_prod = 0
	max_arg = -1
	for i in range(0,max_len):
		if num_a[i] == min_val:
			product = prod[position+i+1] + np.log(features[i+1][sequence[position:position+i+1]])
			if product > max_prod:
				max_prod = product
				max_arg = i
	
	wordLen[position] = len_a[max_arg]
	segNum[position] = num_a[max_arg]
	prod[position] = max_prod
	return segNum[position]
	
def get_segmented(protein,wordLen,feature_list):
	segmented = ''
	N = wordLen.shape[0]
	i = 0
	while i < N:
		next = int(i+wordLen[i])
		segmented = segmented + str(feature_list.index(protein[i:next])) + ' '
		#segmented = segmented + protein[i:next] + ' '
		i = next
	return segmented.strip()
	
def segment_protein(features,protein,family,fpw,feature_list):
	N = len(protein)
	if N > 1000:
		return
	wordLen = np.zeros(N)
	segNum = np.zeros(N)
	prod = np.zeros(N)
	
	segment(features,protein,0,segNum,wordLen,prod,N)
	segmented = get_segmented(protein,wordLen,feature_list)
	fpw.write(family+":"+segmented+'\n')
	
def get_features(features):
	
	feature_list = []
	for key in features.keys():
		feature_list += features[key].keys()
	return feature_list	
	
	
if __name__ == '__main__':
	
	FRESH = False
	data = read_proteins("../Data/Data/astral-scope-95-2.05.fa")
	data1 = read_proteins("../Data/Data/astral-scope-95-2.06.fa")
	if FRESH:
		features = get_features(data)
		with open('features.pkl', 'wb') as fp:
			cPickle.dump(features, fp)
		
	with open('features.pkl', 'rb') as fp:
		features = cPickle.load(fp)
	
	feature_list = get_features(features)
	
	fpw = open("train.txt","w")
	for key in data.keys():
		print key
		for item in data[key]:
			segment_protein(features,item[1],key,fpw,feature_list)
	fpw.close()	
	
	
	"""
	data2 = {}
	data2['A'] = data1['A'].difference(data['A'])
	data2['B'] = data1['B'].difference(data['B'])
	data2['C'] = data1['C'].difference(data['C'])
	data2['D'] = data1['D'].difference(data['D'])
	
	fpw = open("test.txt","w")
	for key in data2.keys():
		print key
		for item in data2[key]:
			segment_protein(features,item[1],key,fpw,feature_list)
	fpw.close()	
	"""
