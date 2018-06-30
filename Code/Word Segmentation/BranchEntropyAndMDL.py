"""
	Program to compute optimal segmentations

	Rahul Kejriwal
	CS14B023
"""

# Python Imports
import numpy as np
from tqdm import tqdm
import sys

# Custom Imports
from SegBase import BaseSegmentor


def outward_iterator(sz):
	# For even sizes
	if sz%2 == 0:
		middle = int(sz/2)
		for dist in range(middle):
			yield middle + dist
			yield middle - dist - 1
	else:
		middle = int(sz/2)
		yield middle
		for dist in range(1, middle+1):
			yield middle + dist
			yield middle - dist


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
			# print "Step - %.2f %.2f" % (curr_threshold_index, HX_vals[curr_threshold_index])

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

		# print "Optimal threshold = ", HX_vals[curr_threshold_index]
		# print "Optimal threshold index = ", curr_threshold_index
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

		# print "DL   - %.2f %.2f %.2f" % (-res, -codebook_res, -res+-codebook_res)
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
		Return delta change in contribution for a given token to DL

		Complexity: O(1)
	"""
	def __changed_dl_contribution(self, old_count, new_count):
		return old_count * (np.log(old_count) if old_count != 0 else 0) - new_count * (np.log(new_count) if new_count != 0 else 0)


	"""
		Compute DL cost of Codebook given segment counts

		Complexity: O(N)
	"""
	def __codebook_cost(self, seg_counts):
		codebook = ''.join([key for key in seg_counts if seg_counts[key]>0])
		codebook_counts = {}
		for tok in codebook:
			codebook_counts[tok] = 1 + codebook_counts.get(tok, 0)

		codebook_res = 0
		for tok, count in codebook_counts.iteritems():
			codebook_res += count * np.log(float(count) / len(codebook))
		return -codebook_res


	"""
		Do repeated splits and merges till convergence
		
		Complexity: ?
	"""
	def __split_and_merge_till_convergence_local(self):

		# Sorted Positions by BranchEntropy Values
		HX_vals = sorted(enumerate(self.HX), key= lambda x:-x[1])
		positions = [pos for pos, val in HX_vals]

		# State Vars
		change = True
		segmented_at = set(np.cumsum([len(el) for el in self.init_seg]))
		N = len(self.init_seg)
		w = {}
		for segment in self.init_seg:
			w[segment] = 1 + w.get(segment, 0)
		T1 = -sum( (w[el]*np.log(w[el]) for el in w) )
		T2 = N * np.log(N)
		C  = self.__codebook_cost(w) 

		tqdm_bar = tqdm()
		while change:
			change = False
			tqdm_bar.update(1)

			"""
				Split
			"""
			for position in positions:
				if position not in segmented_at:
					
					# Get left token
					lftpos = position-1
					while lftpos not in segmented_at and lftpos>=0:
						lftpos -= 1
					if lftpos<0: lftpos = 0
					leftToken = self.corpus[lftpos:position]

					# Get right token
					rtpos = position
					while rtpos not in segmented_at and rtpos<len(self.corpus):
						rtpos += 1
					rightToken = self.corpus[position:rtpos]

					longToken = leftToken + rightToken

					assert(w[longToken] -1 >= 0)
					w_new = dict(w)
					w_new.update({
							leftToken:  1 + w.get(leftToken, 0), 
							rightToken: 1 + w.get(rightToken, 0), 
							longToken: -1 + w.get(longToken, 0), 
						})
					T1_new = T1 + self.__changed_dl_contribution(w.get(leftToken,0), w.get(leftToken,0)+1) + self.__changed_dl_contribution(w.get(rightToken,0), w.get(rightToken,0)+1) + self.__changed_dl_contribution(w[longToken], w[longToken]-1)
					T2_new = (N+1) * np.log(N+1)
					C_new  = self.__codebook_cost(w_new)

					if T1_new + T2_new + C_new < T1 + T2 + C:
						change = True
						segmented_at.add(position) 
						N = N+1
						w[leftToken]  = 1 + w.get(leftToken, 0)
						w[rightToken] = 1 + w.get(rightToken, 0)
						w[longToken] = w.get(longToken, 0) - 1
						T1 = T1_new 
						T2 = T2_new
						C  = C_new

			"""
				Merge
			"""
			# FIX BUG PLS
			for position in reversed(positions):
				if position in segmented_at:			

					# Get left token
					lftpos = position-1
					while lftpos not in segmented_at and lftpos>=0:
						lftpos -= 1
					if lftpos<0: lftpos = 0
					leftToken = self.corpus[lftpos:position]

					# Get right token
					rtpos = position+1
					while rtpos not in segmented_at and rtpos<len(self.corpus):
						rtpos += 1
					rightToken = self.corpus[position:rtpos]

					longToken = leftToken + rightToken

					assert(w[leftToken]  -1 >= 0)
					assert(w[rightToken] -1 >= 0)
					w_new  = dict(w)
					w_new.update({
							leftToken:  -1 + w.get(leftToken, 0), 
							rightToken: -1 + w.get(rightToken, 0), 
							longToken:   1 + w.get(longToken, 0), 
						})
					T1_new = T1 + self.__changed_dl_contribution(w.get(leftToken,0), w.get(leftToken,0)-1) + self.__changed_dl_contribution(w.get(rightToken,0), w.get(rightToken,0)-1) + self.__changed_dl_contribution(w.get(longToken,0), w.get(longToken,0)+1)
					T2_new = (N-1) * np.log(N-1)
					C_new  = self.__codebook_cost(w_new)

					if T1_new + T2_new + C_new < T1 + T2 + C:
						change = True
						segmented_at.remove(position) 
						N = N-1
						w[leftToken]  = w.get(leftToken, 0)  - 1 
						w[rightToken] = w.get(rightToken, 0) - 1
						w[longToken]  = w.get(longToken, 0)  + 1
						T1 = T1_new
						T2 = T2_new
						C  = C_new
		
		tqdm_bar.close()

		# Final segmentation in segmented_at
		segment_boundaries = sorted(list(segmented_at))
		self.init_seg = []
		prev_idx = 0
		for seg_bound in segment_boundaries:
			self.init_seg.append(self.corpus[prev_idx:seg_bound])
			prev_idx = seg_bound
		self.init_seg.append(self.corpus[prev_idx:])

		return self.init_seg


	"""
		Do repeated splits and merges till convergence
		
		Complexity: ?
	"""
	def __split_and_merge_till_convergence_global(self):

		tokens = {}
		for token in self.init_seg:
			tokens[token] = 1 + tokens.get(token, 0)
		tokens = [token for token, count in sorted(list(tokens.iteritems()), key= lambda x:x[1])]

		token_segments = {}
		curr_pos = 0
		for token in self.init_seg:
			token_segments[token] = token_segments.get(token, set([]))
			token_segments[token].add(curr_pos)
			curr_pos += len(token)

		# State Vars
		change = True
		N = len(self.init_seg)
		w = {}
		for segment in self.init_seg:
			w[segment] = 1 + w.get(segment, 0)
		T1 = -sum( (w[el]*np.log(w[el]) for el in w) )
		T2 = N * np.log(N)
		C  = self.__codebook_cost(w) 

		# print "OLD: ", T1+T2+C

		tqdm_bar = tqdm()
		while change:
			change = False
			tqdm_bar.update(1)

			"""
				Split
			"""
			for segment in tokens:
				for pos in outward_iterator(len(segment)):
										
					longToken = segment				
					leftToken = segment[:pos]
					rightToken = segment[pos:]

					w_new = dict(w)
					w_new.update({
							leftToken:  w.get(longToken, 0) + w.get(leftToken, 0), 
							rightToken: w.get(longToken, 0) + w.get(rightToken, 0), 
							longToken:  0, 
						})
					T1_new = T1 + self.__changed_dl_contribution(w.get(leftToken,0), w.get(longToken, 0) + w.get(leftToken, 0)) + self.__changed_dl_contribution(w.get(rightToken,0), w.get(longToken, 0) + w.get(rightToken, 0)) + self.__changed_dl_contribution(w[longToken], 0)
					T2_new = (N+w.get(longToken, 0)) * np.log(N+w.get(longToken, 0))
					C_new  = self.__codebook_cost(w_new)

					# print segment, pos, T1_new+T2_new+C_new

					if T1_new + T2_new + C_new < T1 + T2 + C:
						change = True
						N = N+w.get(longToken, 0)
						w  = w_new
						T1 = T1_new 
						T2 = T2_new
						C  = C_new
						token_segments[leftToken] = token_segments.get(leftToken, set([])).union(token_segments[longToken])	
						token_segments[rightToken] = token_segments.get(rightToken, set([])).union([pos+len(leftToken) for pos in token_segments[longToken]])
						token_segments[longToken] = set([])

			"""
				Merge
			"""
			# FIX BUG PLS
			for segment in reversed(tokens):
				for pos in outward_iterator(len(segment)):
					
					longToken = segment				
					leftToken = segment[:pos]
					rightToken = segment[pos:]

					mergable_count = 0
					for pos in token_segments.get(leftToken, set([])):
						if pos+len(leftToken) in token_segments.get(rightToken, set([])):
							mergable_count += 1

					w_new = dict(w)
					w_new.update({
							leftToken:  w.get(leftToken, 0) - mergable_count, 
							rightToken: w.get(rightToken, 0) - mergable_count, 
							longToken:  w.get(longToken, 0) + mergable_count, 
						})
					T1_new = T1 + self.__changed_dl_contribution(w.get(leftToken,0), w.get(leftToken, 0)-mergable_count) + self.__changed_dl_contribution(w.get(rightToken,0), w.get(rightToken, 0)-mergable_count) + self.__changed_dl_contribution(w[longToken], w[longToken]+mergable_count)
					T2_new = (N-mergable_count) * np.log(N-mergable_count)
					C_new  = self.__codebook_cost(w_new)

					if T1_new + T2_new + C_new < T1 + T2 + C:
						change = True
						N = N-mergable_count
						w  = w_new
						T1 = T1_new 
						T2 = T2_new
						C  = C_new
						for pos in list(token_segments.get(leftToken, set([]))):
							if pos+len(leftToken) in token_segments.get(rightToken, set([])):
								token_segments[leftToken].remove(pos)
								token_segments[rightToken].remove(pos+len(leftToken))
								token_segments[longToken].add(pos)
		
		tqdm_bar.close()

		# Final segmentation in segmented_at
		segmented_at = []
		for token in token_segments:
			segmented_at += list(token_segments[token])

		segment_boundaries = sorted(list(segmented_at))
		self.init_seg = []
		prev_idx = 0
		for seg_bound in segment_boundaries:
			self.init_seg.append(self.corpus[prev_idx:seg_bound])
			prev_idx = seg_bound
		self.init_seg.append(self.corpus[prev_idx:])

		return self.init_seg


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
		# print "Seg Length = ", len(segmentation)
		# print "Orig DL = ", self.dl
		# print "Net DL = ", self.opt_dl()
		# print segmentation
		# print ""

		segmentation = self.__split_and_merge_till_convergence_local()
		# print "Seg Length = ", len(segmentation)
		# print "Orig DL = ", self.dl
		# print "Net DL = ", self.opt_dl()

		segmentation = self.__split_and_merge_till_convergence_global()
		# print "Seg Length = ", len(segmentation)
		# print "Orig DL = ", self.dl
		# print "Net DL = ", self.opt_dl()

		return segmentation


if __name__ == '__main__':

	with open(sys.argv[1]) as fp:
		corpus = ''.join(fp.readlines())

	segmentor = BranchEntropyAndMDLSegmentor(corpus, 4)

	print segmentor.segment()
	print "Orig DL = ", segmentor.dl
	print "Net DL = ", segmentor.opt_dl()