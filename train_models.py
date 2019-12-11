import sys, os, re, csv, codecs
import pickle
import numpy as np, pandas as pd

from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

import basic_models
from filenames import model_file, data_file
from handle_text import handle_text


# Load pre-trained word vectors
EMBEDDING ='data/glove.840B.300d.txt' 

# EMBEDDING='data/crawl-300d-2M.vec'

# Save training data
# TRAIN_DATA ='train_500_075.pkl'
# TEST_DATA = 'test_500_075.pkl'
# FB_DATA = 'data/rated_fb_comments.pkl'

TRAIN_DATA ='train_500_075.pkl'
TEST_DATA = 'test_500_075.pkl'
FB_DATA = 'data/fb_rated.pkl'


def train_model(model, X, Y, filename, batch_size=64, epochs=20):
	"""
	Trains and returns a Keras model on some input data and labels.

	Arguments:
	- model: A Keras sequential model
	- X: An input Numpy array that matches the input shape of the model
	- Y: A Numpy array with labels for the samples of X
	- batch_size: Number of samples to process together (default=64)
	- epochs: Maximum number of times to pass data through model (default=20)
	"""

	model.compile(loss='categorical_crossentropy', optimizer='adam', 
				  metrics=['accuracy'])
	es = EarlyStopping(monitor='val_loss',
	                   min_delta=0,
	                   patience=3,
	                   verbose=0, mode='auto')
	checkpoint = ModelCheckpoint(filename, monitor='val_loss', 
								 verbose=0, save_best_only=True, mode='auto')
	model.fit(X, Y, batch_size=batch_size, epochs=epochs,
			  callbacks=[es,checkpoint], validation_split=0.1)
	final_model = load_model(filename)
	return final_model


def split(comments, num_train, num_test, threshold):
	"""
	Returns two dataframes of comments, for training and testing respectively.
	Testing data only has comments with toxicity above a certain threshold.

	Arguments:
	- comments: Dataframe with comments and their corresponding toxicities
	- num_train: Number of comments for training set
	- num_test: Number of comments for testing set
	- threshold: Value between 0 and 1 for minimum toxicity in testing data
	"""

	num_comments = (comments.shape)[0]
	assert(num_train + num_test < num_comments)

	train_indices = np.random.choice(num_comments, num_train, replace=False)
	training_data = comments.iloc[train_indices]

	rest_data = comments.drop(train_indices)
	toxic_comments = rest_data[rest_data['toxicity'] >= threshold]
	num_toxic = (toxic_comments.shape)[0]
	test_indices = np.random.choice(num_toxic, num_test, replace=False)
	testing_data = toxic_comments.iloc[test_indices]

	return training_data, testing_data 

def save_train_test(comments, num_test, threshold):
	toxic = comments[comments['toxicity'] >= threshold]
	mask = (toxic['comment'].str.split().apply(len) >= 10)
	long_toxic = toxic.loc[mask]
	num_toxic = (long_toxic.shape)[0]

	np.random.seed(726)
	test_indices = np.random.choice(num_toxic, num_test, replace=False)
	testing_data = long_toxic.iloc[test_indices]
	
	training_data = comments.drop(testing_data.index.values)

	file_ext = ('_' + str(num_test) + '_' 
				+ str(threshold)[:1] + str(threshold)[2:]
				+ '.pkl')

	training_data.to_pickle('train' + file_ext)
	testing_data.to_pickle('test' + file_ext)


def bulk_train_and_vectorize(model_type, train_sizes):
	model = getattr(basic_models, model_type)()
	print("Opening pickle files")
	full_train = pickle.load(open(TRAIN_DATA, 'rb'))
	num_train = (full_train.shape)[0]
	test = pickle.load(open(TEST_DATA, 'rb'))
	np.random.seed(726)

	for train_size in train_sizes:
		train_indices = np.random.choice(num_train, train_size, replace=False)
		train = full_train.iloc[train_indices]
		print("Handling text for training size " + str(train_size))
		x_train, y_train, x_test, y_test = handle_text(train, test)

		print("Saving test data as a .npy file")
		test_file = data_file(train_size)
		np.save(test_file, x_test)
		np.save('fb_test_y', y_test)

		print("Training model now")
		model_filename = model_file(model_type, train_size)
		train_model(model, x_train, y_train, model_filename)

# Saves test and training pickle files
# comments = pickle.load(open(FB_DATA, 'rb'))
# save_train_test(comments, 500, 0.75)

# # Tests that the dataframe was saved correctly
# df = pickle.load(open('test_500_075.pkl', 'rb'))
# comments = list(df['comment'].values)
# lengths = [len(c.split()) for c in comments]
# print(lengths)

# Tests that models can be bulk-trained
bulk_train_and_vectorize('basic_cnn', [300000])
