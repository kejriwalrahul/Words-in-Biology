"""
	Program to compute optimal segmentations

	Rahul Kejriwal
	CS14B023
"""

# Python Imports
import numpy as np
from tqdm import tqdm

# Custom Imports
from SegBase import BaseSegmentor


class BranchEntropyAndMDLSegmentor(BaseSegmentor):

	"""
		Build Counter etc.

		Complexity: O(NlogN)
	"""
	def __init__(self, corpus, n):
		self.HX = None
		self.n = n
		self.init_seg = None
		super(BranchEntropyAndMDLSegmentor, self).__init__(corpus)


	"""
		No smoothing for ngram probabilities 
		
		Complexity: O(logN)
	"""
	def __ngram_prob_unsmoothed(self, ngram):
		denom_str,num_str = ngram
		count_denom = self.ctr.count(denom_str)
		if count_denom != 0:
			return float(self.ctr.count(num_str)) / count_denom
		else:
			return 0


	"""
		Returns H(X_k;X_{k-1}|x_{k-1:k-n};x_{k:k+n-1})
		
		Complexity: O(VlogN)
	"""
	def __BranchingEntropy(self, k, n):
		res = 0
		for x in self.vocab:
			fwd_ngram = (self.corpus[k-n:k],self.corpus[k-n:k]+x)
			bwd_ngram = (self.corpus[k:k+n],x+self.corpus[k:k+n])
			fwd_prob, bwd_prob = self.__ngram_prob_unsmoothed(fwd_ngram), self.__ngram_prob_unsmoothed(bwd_ngram)
			res += (fwd_prob * np.log(fwd_prob)) if fwd_prob > 0 else 0 
			res += (bwd_prob * np.log(bwd_prob)) if bwd_prob > 0 else 0
		return -res


	"""
		Return initial segmentation based on threshold of branching entropy
		
		Complexity: O(NVlogN + NlogN) 
	"""
	def __init_seg(self):

		# PreProcessing
		n = self.n
		if self.init_seg != None:
			return self.init_seg

		# Get sorted branching entropies for finding thresholds
		if self.HX == None:
			self.HX = [self.__BranchingEntropy(i,n) for i in range(1, len(self.corpus))]
		HX_vals = sorted(self.HX)

		curr_threshold_index = len(HX_vals) / 2
		step_size = len(HX_vals) / 4

		# 1 = Forward, 0 = Backward
		direction = 1
		minimum = float('inf')

		tqdm_bar = tqdm(total=int(np.log(len(self.corpus))))
		while step_size > 0:
			print "Step - %.2f %.2f" % (curr_threshold_index, HX_vals[curr_threshold_index])

			tqdm_bar.update(1)

			"""	
				Try moving in last direction
			"""
			deviation = step_size * (1 if direction == 1 else -1)
			next_threshold_index = curr_threshold_index+deviation
			curr_DL = self.__calc_DL(self.__segmentation_at_threshold(HX_vals[next_threshold_index]))

			if curr_DL < minimum:
				minimum = curr_DL
				curr_threshold_index = next_threshold_index 
				step_size /= 2
				continue

			"""
				If fail, move in opp direction
			"""
			direction = 0 if direction == 1 else 1
			deviation = step_size * (1 if direction == 1 else -1)
			next_threshold_index = curr_threshold_index+deviation
			curr_DL = self.__calc_DL(self.__segmentation_at_threshold(HX_vals[next_threshold_index]))

			if curr_DL < minimum:
				minimum = curr_DL
				curr_threshold_index = next_threshold_index 
				step_size /= 2
				continue

			"""
				If fail again, revert direction and halve move size
			"""
			direction = 0 if direction == 1 else 1
			step_size /= 2
		tqdm_bar.close()

		print "Optimal threshold = ", HX_vals[curr_threshold_index]
		print "Optimal threshold index = ", curr_threshold_index
		self.init_seg = self.__segmentation_at_threshold(HX_vals[curr_threshold_index])
		return self.init_seg


	"""
		Calculate DL of corpus given segmentation

		Complexity: O(N)
	"""
	def __calc_DL(self, segmentation):
		
		"""
			Data Counts
		"""
		tok_counts = {}
		for tok in segmentation:
			tok_counts[tok] = 1 + tok_counts.get(tok, 0)

		"""
			CodeBook Counts
		"""
		codebook = ''.join(tok_counts.keys())
		codebook_counts = {}
		for tok in codebook:
			codebook_counts[tok] = 1 + codebook_counts.get(tok, 0)

		"""
			Compute L(M)
		"""
		codebook_res = 0
		for tok, count in codebook_counts.iteritems():
			codebook_res += count * np.log(float(count) / len(codebook))

		"""
			Compute L(D|M)
		"""
		res = 0
		for tok, count in tok_counts.iteritems():
			res += count * np.log( float(count) / len(segmentation) )

		print "DL   - %.2f %.2f %.2f" % (-res, -codebook_res, -res+-codebook_res)
		return -res + -codebook_res 


	"""
		Segments corpus using a given threshold

		Complexity: O(N)
	"""
	def __segmentation_at_threshold(self, threshold):
		segmented_list = []
		last_index = 0
		for i in range(len(self.HX)):
			if self.HX[i] >= threshold:
				segmented_list.append(self.corpus[last_index:i+1])
				last_index = i+1
		segmented_list.append(self.corpus[last_index:])
		return segmented_list


	"""
		Return DL of optimal segmentation of corpus
		
		Complexity: O(NVlogN + NlogN) 	for uncached
					O(N) 				for cached
	"""
	def opt_dl(self):
		return self.__calc_DL(self.__init_seg())


	"""
		Return optimal segmentation of corpus

		Complexity: O(NVlogN + NlogN) 
	"""
	def opt_segmentation(self):
		segmentation = self.__init_seg()
		print "Seg Length = ", len(segmentation)
		return segmentation


if __name__ == '__main__':

	corpus = raw_input()

	segmentor = BranchEntropyAndMDLSegmentor(corpus, 4)

	print segmentor.segment()
	print "Orig DL = ", segmentor.dl
	print "Net DL = ", segmentor.opt_dl()