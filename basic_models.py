import sys, os, re, csv, codecs
import numpy as np, pandas as pd

from keras.layers import Dense, Input, LSTM, GRU, Embedding, SpatialDropout1D, Dropout, Activation, Flatten
from keras.layers import Bidirectional, Conv1D, MaxPooling1D, Concatenate
from keras.models import Sequential, Model
from keras import initializers, regularizers, constraints, optimizers, layers, backend

import tensorflow as tf

from cleverhans.utils_keras import KerasModelWrapper


def word_rnn(alg, capacity, word_size, sentence_size):
	model = Sequential()
	RNN = globals()[alg]
	layers = [Bidirectional(RNN(capacity, return_sequences=True,
			  				dropout=0.1, recurrent_dropout=0.1,
			  				input_shape=(sentence_size, word_size))),
			  Bidirectional(RNN(capacity, return_sequences=False, 
			  					dropout=0.1, recurrent_dropout=0.1)),
			  Dense(capacity, activation="relu"),
			  Dropout(0.1),
			  Dense((capacity//2), activation="relu"),
			  Dropout(0.1),
			  Dense(2, activation="softmax")]
	for layer in layers:
		model.add(layer)
	return model


def basic_lstm(capacity=128, word_size=300, sentence_size=200):
	"""
	Returns a basic untrained LSTM architecture.

	Arguments:

	- capacity: An int representing the model capacity (default=128)
	- word_size: Number of dimensions for a word in the embedding (default=300)
	- sentence_size: Number of words in a sentence (default=200)
	"""
	return word_rnn('LSTM', capacity, word_size, sentence_size)


def basic_gru(capacity=128, word_size=300, sentence_size=200):
	"""
	Returns a basic untrained GRU architecture.

	Arguments:

	- capacity: An int representing the model capacity (default=128)
	- word_size: Number of dimensions for a word in the embedding (default=300)
	- sentence_size: Number of words in a sentence (default=200)
	"""
	return word_rnn('GRU', capacity, word_size, sentence_size)

def kim_cnn(word_size=300, sentence_size=200, num_filters=100, 
			  filter_sizes=[3,4,5]):
	"""
	Return a basic untrained word-CNN architecure.

	Arguments:

	-num_filters: Number of filters to apply for each filter size (default=100)
	-filter_sizes: List of filter_sizes to convolve (default=[3,4,5])
	- word_size: Number of dimensions for a word in the embedding (default=300)
	- sentence_size: Number of words in a sentence (default=200)
	"""
	model = Sequential()
	graph_in = Input(shape=(sentence_size, word_size))
	conv_blocks = []
	for fs in filter_sizes:
		conv = Conv1D(num_filters, kernel_size=fs, 
					  padding='valid', activation='relu', 
					  kernel_initializer='he_uniform')(graph_in)
		pool = MaxPooling1D(pool_size=sentence_size-fs+1)(conv)
		flattened = Flatten()(pool)
		conv_blocks.append(flattened)
	graph_out = Concatenate()(conv_blocks)
	cnn_layer = Model(inputs=graph_in, outputs=graph_out)
	layers = [cnn_layer, 
			  Dense(128), Dropout(0.5), Activation('relu'),
			  Dense(2), Activation('softmax')]
	for layer in layers: 
		model.add(layer)
	return model

def basic_cnn(word_size=300, sentence_size=100, num_filters=100, 
			  filter_sizes=[2,4,6]):
	model = Sequential()
	layers = []
	for f in range(len(filter_sizes)):
		if f==0:
			layers.append(Conv1D(num_filters, kernel_size=filter_sizes[f], 
						 padding='valid', activation='relu',
						 kernel_initializer='he_uniform',
						 input_shape=(sentence_size, word_size)))
		else:
			layers.append(Conv1D(num_filters, kernel_size=filter_sizes[f], 
						 padding='valid', activation='relu',
						 kernel_initializer='he_uniform'))
		layers.append(MaxPooling1D(pool_size=2))
		layers.append(Dropout(0.5))
	layers.append(Flatten())
	layers.append(Dense(100, activation='relu'))
	layers.append(Dropout(0.5))
	# layers.append(Dense(50, activation='relu'))
	# layers.append(Dropout(0.5))
	layers.append(Dense(2, activation='softmax'))
	for layer in layers:
		model.add(layer)
	return model

	

# cnn = basic_cnn()
# print(cnn.summary())

# lstm = basic_gru()
# X = np.random.rand(1000, 200, 300)
# Y = np.stack((np.zeros(1000), np.ones(1000)), axis=1)
# lstm.compile(loss='binary_crossentropy', optimizer='adam',
# 			 metrics=['accuracy'])
# lstm.fit(X, Y, batch_size=64, epochs=3, validation_split=0.1)
# print(lstm.summary())
