import cPickle
import numpy as np

from sklearn import svm
from sklearn.metrics import classification_report

def get_features(features):
	
	feature_list = []
	for key in features.keys():
		feature_list += features[key].keys()
	return feature_list	
	
def get_accuracy(vec1,vec2):
	
	num_correct = (vec1==vec2).sum()
	total = vec1.shape[0]
	
	return float(num_correct)/total

def load_data(filename,num_features):
	
	X = []
	Y = []
	
	fp = open(filename,'r')
	for line in fp:
		line = line.strip().split(':')
		Y.append(ord(line[0])-ord('A'))
		line = line[1].strip().split()
		featurized = np.zeros(num_features)
		for item in line:
			featurized[int(item)] += 1
		X.append(featurized)
	
	X = np.array(X)
	Y = np.array(Y)
	return (X,Y)

if __name__ == '__main__':	
	with open('features.pkl', 'rb') as fp:
		features = cPickle.load(fp)
	
	feature_list = get_features(features)
	num_features = len(feature_list)
	
	train_X, train_Y = load_data("train.txt",num_features)
	
	test_X, test_Y = load_data("test.txt",num_features)
	
	print "Loaded"
	model = svm.SVC(verbose=True,max_iter=10000)
	
	model.fit(train_X,train_Y)
	predicted = model.predict(test_X)
	print "Accuracy: "+str(get_accuracy(predicted,test_Y))
	
	print(classification_report(test_Y, predicted))
	
		
