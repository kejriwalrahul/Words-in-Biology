"""
	Estimating Ngram Counts of any length using virual corpus approach
		(as described in "The Virtual Corpus approach to deriving n-gram statistics from large scale corpora")

	Rahul Kejriwal
	CS14B023
"""

from nltk.corpus import gutenberg
import sys

class NgramCounter:

	def __init__(self, string):
		self.corpus = string
		self.corpus_length = len(self.corpus)

		self.vc = list(range(len(string)))
		self.vc = sorted(self.vc, cmp= self.__ptr_comparator)

		self.cached_counts = {}


	def __ptr_comparator(self, p1, p2):
		k = 0
		while(self.corpus[p1+k]==self.corpus[p2+k]):
			k += 1
			if p1+k >= self.corpus_length and p2+k >= self.corpus_length:
				return 0
			if p1+k >= self.corpus_length:
				return -1
			if p2+k >= self.corpus_length:
				return 1
		return ord(self.corpus[p1+k]) - ord(self.corpus[p2+k])


	def __bsearch(self, ngram, checker):
		l, r = 0, len(self.vc)-1
		m = (l+r)/2
		while r >= l:
			c = checker(m, ngram)
			if   c ==  0:	return m
			elif c ==  1:	r = m-1
			elif c == -1:   l = m+1
			else:	raise TypeError("Bad Checker!")
			m = (l+r)/2
		return -1


	def __left_checker(self, i, ng):
		if self.corpus[self.vc[i]:self.vc[i]+len(ng)]<ng:
			return -1
		elif self.corpus[self.vc[i]:self.vc[i]+len(ng)] >= ng and (i-1<0 or self.corpus[self.vc[i-1]:self.vc[i-1]+len(ng)] < ng):
			return 0
		else:
			return 1


	def __right_checker(self, i, ng):
		if self.corpus[self.vc[i]:self.vc[i]+len(ng)]>ng:
			return 1
		elif self.corpus[self.vc[i]:self.vc[i]+len(ng)] <= ng and (i+1>=len(self.vc) or self.corpus[self.vc[i+1]:self.vc[i+1]+len(ng)] > ng):
			return 0
		else:
			return -1


	def count(self, ngram):
		if ngram in self.cached_counts:
			return self.cached_counts[ngram]

		left_bound = self.__bsearch(ngram, self.__left_checker)
		right_bound = self.__bsearch(ngram, self.__right_checker)
		if left_bound == -1 or right_bound == -1:	return 0
		
		self.cached_counts[ngram] = right_bound - left_bound + 1
		return right_bound - left_bound + 1


if __name__ == '__main__':

	corpus = gutenberg.raw('carroll-alice.txt').lower()
	counter = NgramCounter(corpus)

	print counter.count('alice')