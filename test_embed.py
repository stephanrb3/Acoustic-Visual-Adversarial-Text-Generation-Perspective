import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, SpatialDropout1D, Dropout, Activation
from keras.layers import Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Model, load_model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers, backend
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

from basic_models import basic_cnn
from train_models import train_model

# Load pre-trained word vectors
EMBEDDING ='data/glove.840B.300d.txt' 
# EMBEDDING='data/crawl-300d-2M.vec'

# Save training and testing data
TRAIN_DATA ='train.csv'
TEST_DATA = 'test.csv'
SAMPLE_SUB = 'sample_submission.csv' 

# Size of word vector, given by our pre-trained vectors
embed_size = 300 
# Number of unique words to use (i.e num rows in embedding matrix)
max_features = 150000 
# Max number of words in a comment to use
maxlen = 50 

# Load data into pandas
train = pd.read_csv(TRAIN_DATA, usecols=["id", "comment_text", "toxic"])
test = pd.read_csv(TEST_DATA)
submission = pd.read_csv(SAMPLE_SUB, usecols=["id", "toxic"])

# Replace missing values in training and test set
list_train = train["comment_text"].fillna("_na_").values
classes = ["toxic"]
y = train[classes].values
list_test = test["comment_text"].fillna("_na_").values

def handle_text():
	""" 
# 	This function simply assumes that list_train and list_test are Pandas
# 	columns with training/testing comments, and that EMBEDDING is a trained
# 	embedding. Then, it outputs those comments in vector form (one for train,
# 	 one for test) along with an embedding matrix.
# 	""" 
	
	# Use Keras preprocessing tools
	tok = Tokenizer(num_words=max_features)
	tok.fit_on_texts(list(list_train))
	tokenized_train = tok.texts_to_sequences(list_train)
	tokenized_test = tok.texts_to_sequences(list_test)

	# Pad vectors with 0s for sentences shorter than maxlen
	X_t = pad_sequences(tokenized_train, maxlen=maxlen)
	X_te = pad_sequences(tokenized_test, maxlen=maxlen)

	# Read word vectors into a dictionary
	def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
	embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING))
	close(EMBEDDING)
# 	print("Example Key")
# 	print((list(embeddings_index))[0])
# 	print("Example Value")
# 	print((list(embeddings_index.values()))[0])
# 	# Create the embedding matrix
	word_index = tok.word_index
	nb_words = min(max_features, len(word_index))
	embedding_matrix = np.zeros((nb_words, embed_size))
	for word, i in word_index.items():
	    if i >= max_features: continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

	return X_t, X_te, embedding_matrix

X_t, X_te, embedding_matrix = handle_text()
y_t = train['toxic'].values
y = np.stack((y_t, np.ones((y_t.shape)[0]) - y_t), axis=1)
# print(embedding_matrix.shape)

# print(X_te.shape)


vocab_size = 150000
word_size = 300
sentence_size = 50
# examples = 1000

model = Sequential()
# embedding_matrix = np.random.rand(vocab_size, word_size)
# X = np.random.randint(1000, size=(examples, sentence_size))
# Y = np.stack((np.zeros(examples), np.ones(examples)), axis=1)
model.add(Embedding(vocab_size, word_size, weights=[embedding_matrix],
						input_length=sentence_size, trainable=False))
model.compile('rmsprop', 'mse')
vectors_X = model.predict(X_t)

new_model = basic_cnn()
train_model(new_model, vectors_X, y, 'test_cnn.h5')


# new_model = Sequential()
# layers = [Bidirectional(LSTM(50, return_sequences=False, 
# 			  					dropout=0.1, recurrent_dropout=0.1)),
# 			  Dense(50, activation="relu"),
# 			  Dropout(0.1),
# 			  Dense(2, activation="softmax")]
# for layer in layers: 
# 	new_model.add(layer)
# new_model.compile(loss='binary_crossentropy', optimizer='adam', 
# 				  metrics=['accuracy'])
# new_model.fit(output, Y, batch_size=64, epochs=3, validation_split=0.1)
# print(new_model.summary())




# Load best model and predict
# model = load_model(best_model)
# print ('**Predicting on test set**')
# pred = model.predict(X_te, batch_size=1024, verbose=1)
# submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = pred
# submission.to_csv('preds/submission15.csv', index=False)
