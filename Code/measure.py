"""
	Measure Precision and Recall of given segmentation for English 

	Rahul Kejriwal
"""
import re
import sys

out_file = sys.argv[1]

with open(out_file) as fp:
	segmentation = fp.readline().strip()

seg_re = re.compile('\[\[(.*?)\]\]')
matches = re.findall(seg_re, segmentation)

count = 0
for i in range(len(matches)-1):
	if matches[i].endswith('_') or matches[i+1].startswith('_'):
		count += 1

p, r = float(count) / (len(matches)-1), float(count) / (segmentation.count('_'))
print "Precision: ", p
print "Recall: ", r
print "F1: ", (2.0 * p * r) / (p+r)