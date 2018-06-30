"""
	Read Protein Data from file

	Rahul Kejriwal
	CS14BO23
"""

# Python Imports
import re

# Custom Imports


def read_proteins(file):
	proteins = {
		'A': set([]),
		'B': set([]),
		'C': set([]),
		'D': set([]),
	}
	fline = re.compile('>(.*?) (.*)?.*')

	with open(file) as fp:
		pcount = 0
		line = fp.readline().strip()
		while line:	
			try:
				match_obj = fline.match(line)
				family, pid = match_obj.group(2), match_obj.group(1)
				family = family[0].upper()
			except:
				print line
				raise NotImplementedError("error")

			curr_protein = ""
			currline = fp.readline().strip()
			while not currline.startswith('>') and currline:
				curr_protein += currline
				currline = fp.readline().strip()
			curr_protein = (pid, curr_protein)

			if family in ['A', 'B', 'C', 'D']:
				proteins[family].add(curr_protein)
			pcount += 1
			line = currline

	return proteins


def prepare_data(data):
	dataset = []
	for family in data:
		for pid, protein in data[family]:
			dataset.append([protein, family])
	return dataset


if __name__ == '__main__':
	data_2_05 = read_proteins("../Data/Protein Data/astral-scope-95-2.05.fa")
	data_2_06 = read_proteins("../Data/Protein Data/astral-scope-95-2.06.fa")

	a,b,c,d = len(data_2_05['A']), len(data_2_05['B']), len(data_2_05['C']), len(data_2_05['D'])
	print "A:%d,\tB:%d,\tC:%d,\tD:%d,\tTot:%d" % (a,b,c,d,a+b+c+d)
	a,b,c,d = len(data_2_06['A'].difference(data_2_05['A'])), len(data_2_06['B'].difference(data_2_05['B'])), len(data_2_06['C'].difference(data_2_05['C'])), len(data_2_06['D'].difference(data_2_05['D']))
	print "A:%d,\tB:%d,\tC:%d,\tD:%d,\tTot:%d" % (a,b,c,d,a+b+c+d)

	data_2_06 = {
		'A': data_2_06['A'].difference(data_2_05['A']),
		'B': data_2_06['B'].difference(data_2_05['B']),
		'C': data_2_06['C'].difference(data_2_05['C']),
		'D': data_2_06['D'].difference(data_2_05['D']),
	}
	print len(prepare_data(data_2_05))
	print len(prepare_data(data_2_06))