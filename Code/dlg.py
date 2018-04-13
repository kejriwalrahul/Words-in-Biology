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
from ngram_count import NgramCounter


class Segmentor:


	def __init__(self, corpus):
		self.corpus = self.__prepare_corpus(corpus)
		self.corpus_length = len(self.corpus)
		self.ctr = NgramCounter(self.corpus)
		self.vocab = list(set(self.corpus))
		self.dl = self.__dl()


	"""
		Follow corpus pre-processing as given in paper
	"""
	def __prepare_corpus(self, string):
		return string.lower().replace('.', '@').replace('\n',' ')


	"""
		Viterbi Algo applied to find optimal segmentation
	"""
	def __viterbi_opt_segmentation(self):
		string = self.corpus

		OS = []
		for k in tqdm(range(len(string)+1)):
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

		return OS[-1]


	"""
		Return segmented string
	"""
	def segment(self):
		return ('[' + ']['.join(self.__viterbi_opt_segmentation()) + ']').replace(' ','_')


	"""
		Returns DL of corpus
	"""
	def __dl(self):
		total = 0
		for token in self.vocab:
			c_token = self.ctr.count(token)
			total -= c_token * np.log( float(c_token) / self.corpus_length )
		return total


	def __dl_segment(self, segment):
		c_segment = self.ctr.count(segment)
		new_corpus_length = self.corpus_length - c_segment*len(segment) + c_segment + len(segment) + 1

		total = 0
		for token in self.vocab:
			c_token = self.ctr.count(token) - (c_segment-1)*segment.count(token)
			total -= c_token * np.log( float(c_token) / new_corpus_length )

		# For new index
		c_token = c_segment
		total -= c_token * np.log( float(c_token) / new_corpus_length )

		return total


	"""
		Returns find DLG by using a given segment
	"""
	def __segment_dlg(self, segment):	
		return self.dl - self.__dl_segment(segment)


	"""
		Find DLG of a given segmentation
	"""
	def __dlg(self, segmentation):
		return sum((self.__segment_dlg(segment) for segment in segmentation))


if __name__ == '__main__':

	corpus = "alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought alice 'without pictures or conversation?' so she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a white rabbit with pink eyes ran close by her. there was nothing so very remarkable in that; nor did alice think it so very much out of the way to hear the rabbit say to itself, 'oh dear! oh dear! i shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge." 

	segmentor = Segmentor(corpus)
	print segmentor.segment()
	print ""