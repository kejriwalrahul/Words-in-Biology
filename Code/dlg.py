"""
	Unsupervised Word Segmentation using MDL gain paradigm
		(as described in "Unsupervised Learning of Word Boundary with Description Length Gain")

	Rahul Kejriwal
	CS14B023
"""

# Python Lib imports
import numpy as np
from tqdm import tqdm

# Custom Imports
from SegBase import BaseSegmentor


"""
	Creates Segmentor based on DL Gain paradigm
"""
class DLGainSegmentor(BaseSegmentor):

	"""
		DL of corpus after adding one segment to codebook
	"""
	def __dl_segment(self, segment):
		c_segment = self.ctr.count(segment)
		new_corpus_length = self.corpus_length - c_segment*len(segment) + c_segment + len(segment) + 1

		total = 0
		for token in self.vocab:
			c_token = self.ctr.count(token) - (c_segment-1)*segment.count(token)
			if c_token > 0:
				total -= c_token * np.log( float(c_token) / new_corpus_length )

		# For new index
		c_token = c_segment
		total -= c_token * np.log( float(c_token) / new_corpus_length )

		return total


	"""
		Find DLG of a given segmentation
	"""
	def __dlg(self, segmentation):
		return sum((self.dl - self.__dl_segment(segment) for segment in segmentation))


	"""
		Viterbi Algo applied to find optimal segmentation
	"""
	def opt_segmentation(self):

		# If already done
		if self.segmented != None:	return self.segmented

		string = self.corpus

		OS = []
		for k in range(len(string)+1):
			OS.append([])
			curr_dlg = -float("inf")

			for j in range(k-1,-1,-1):
				if self.ctr.count(string[j+1:k+1]) < 2:	break
				try_dlg = self.__dlg(OS[j]+[string[j+1:k+1]])
				if try_dlg > curr_dlg:
					OS[k] = OS[j]+[string[j+1:k+1]]
					curr_dlg = try_dlg

			if OS[k] == []:
				OS[k] = OS[k-1]+[string[k]]

		# Save Segmentation and return
		self.segmented = OS[-1]
		return OS[-1]


	"""
		Return optimal DL
	"""
	def opt_dl(self):
		segmentation = self.opt_segmentation()
		
		seg_counts = {}
		for tok in segmentation:
			seg_counts[tok] = 1 + seg_counts.get(tok, 0)

		total = 0
		for token, count in seg_counts.iteritems():
			total -= count * np.log( float(count) / len(segmentation) )
		return total


if __name__ == '__main__':

	corpus = raw_input()

	segmentor = DLGainSegmentor(corpus)
	print segmentor.segment()
	print "Orig DL = ", segmentor.dl
	print "Net DL = ", segmentor.opt_dl()