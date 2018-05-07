"""
	Measure Precision and Recall of given segmentation for English 

	Rahul Kejriwal
"""
import re
import sys
import collections

out_file = sys.argv[1]

with open(out_file) as fp:
	segmentation = fp.readline().strip()

seg_re = re.compile('\[\[(.*?)\]\]')
matches = re.findall(seg_re, segmentation)
matches = list(matches)

count = 0
for i in range(len(matches)-1):
	if matches[i].endswith('_') or (matches[i+1].startswith('_') and matches[i+1]!='_'):
		count += 1

c1 = collections.Counter(matches)

print count, len(matches), (segmentation.count('_'))

p, r = float(count) / (len(matches)-1), float(count) / (segmentation.count('_'))
print "Precision: ", p
print "Recall: ", r
print "F1: ", (2.0 * p * r) / (p+r)