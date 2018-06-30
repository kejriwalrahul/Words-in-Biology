"""
	Trying RNN Models for protein classification

	Adapted (heavily) from:
		https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

	Rahul Kejriwal
	CS14B023
"""

# Custom Imports
from read_data import read_proteins, prepare_data

# LSTM with dropout for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import classification_report


def data_formatting(data, vocab=None):
	if vocab == None:
		vocab = set([])
		for sample, label in data:
			vocab = vocab.union(set(list(sample)))
		# O for OOV
		vocab = {el:i+1 for i, el in enumerate(vocab)}
		vocab['<S>']  = len(vocab.keys())+1
		vocab['</S>'] = len(vocab.keys())+1

	X, Y = [], []
	for sample, label in data:
		Y.append([1 if ord(label)-65==i else 0 for i in range(4)])
		X.append([vocab['<S>']]+[vocab.get(el,0) for el in sample]+[vocab['</S>']])
	return np.array(X), np.array(Y), vocab


if __name__ == '__main__':

	# fix random seed for reproducibility
	np.random.seed(7)

	# Load Data
	data_2_05 = read_proteins("../Data/Data/astral-scope-95-2.05.fa")
	data_2_06 = read_proteins("../Data/Data/astral-scope-95-2.06.fa")
	data_2_06 = {
		'A': data_2_06['A'].difference(data_2_05['A']),
		'B': data_2_06['B'].difference(data_2_05['B']),
		'C': data_2_06['C'].difference(data_2_05['C']),
		'D': data_2_06['D'].difference(data_2_05['D']),
	}
	X_train, y_train, vocab = data_formatting(prepare_data(data_2_05))
	X_test, y_test, vocab  = data_formatting(prepare_data(data_2_06), vocab)

	# truncate and pad input sequences
	max_seq_length = 200
	X_train = sequence.pad_sequences(X_train, maxlen=max_seq_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_seq_length)

	# Find number of protein lexemes
	print "Protein Lexemes Count =", len(vocab.keys()) + 1

	# create the model
	embedding_vecor_length = 6
	model = Sequential()
	model.add(Embedding(len(vocab)+1, embedding_vecor_length, input_length=max_seq_length))
	model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))	
	model.add(LSTM(80, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(4, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, epochs=50, batch_size=64)

	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	# Precision and accuracy
	pred = model.predict(X_test, batch_size=64, verbose=1)
	predicted = np.argmax(pred, axis=1)
	print classification_report(np.argmax(y_test, axis=1), predicted)
