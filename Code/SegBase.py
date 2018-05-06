"""
	Base Class for Segmentation Algorithms 

	Rahul Kejriwal
	CS14B023
"""

# Python Imports
import numpy as np
import abc

# Custom Imports
from ngram_count import NgramCounter


class BaseSegmentor:
	__metaclass__ = abc.ABCMeta


	"""
		Takes stram and prepares Segmentor
	"""
	def __init__(self, corpus):
		self.corpus = self.__prepare_corpus(corpus)
		self.corpus_length = len(self.corpus)
		self.ctr = NgramCounter(self.corpus)
		self.vocab = list(set(self.corpus))
		self.dl = self.__dl()
		self.segmented = None

	"""
		Corpus pre-processing
	"""
	def __prepare_corpus(self, string):
		return string.lower().replace('.', '@').replace('\n',' ')


	"""
		Returns DL of corpus
	"""
	def __dl(self):
		total = 0
		for token in self.vocab:
			c_token = self.ctr.count(token)
			total -= c_token * np.log( float(c_token) / self.corpus_length )
		return total


	"""
		Returns the optimal description length after making codebook
	"""
	@abc.abstractmethod
	def opt_dl(self):
		return


	"""
		Returns optimal segmentation of corpus into words
	"""
	@abc.abstractmethod
	def opt_segmentation(self):
		return


	"""
		Return segmented string
	"""
	def segment(self):
		return ('[[' + ']][['.join(self.opt_segmentation()) + ']]').replace(' ','_')