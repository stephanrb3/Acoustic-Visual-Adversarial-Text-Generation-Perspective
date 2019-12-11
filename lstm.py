import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Embedding, SpatialDropout1D, Dropout, Activation
from keras.layers import Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint


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
maxlen = 200 

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
	This function simply assumes that list_train and list_test are Pandas
	columns with training/testing comments, and that EMBEDDING is a trained
	embedding. Then, it outputs those comments in vector form (one for train,
	 one for test) along with an embedding matrix.
	""" 
	
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

	# Create the embedding matrix
	word_index = tok.word_index
	nb_words = min(max_features, len(word_index))
	embedding_matrix = np.zeros((nb_words, embed_size))
	for word, i in word_index.items():
	    if i >= max_features: continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

	return X_t, X_te, embedding_matrix

X_t, X_te, embedding_matrix = handle_text()

def get_model(filename):
	"""
	This function returns a model and callbacks to be used during a call
	to "fit." The callbacks ensure that the best model is saved to the 
	filename provided. 
	"""

	# Bidirectional LSTM-CNN with max-pooling and 2 FC layers
	inp = Input(shape=(maxlen,))
	x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
	x = SpatialDropout1D(0.2)(x)
	x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
	x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform", activation="relu")(x)
	avg_pool = GlobalAveragePooling1D()(x)
	max_pool = GlobalMaxPooling1D()(x)
	x = concatenate([avg_pool, max_pool])
	x = Dense(128, activation="relu")(x)
	x = Dropout(0.1)(x)
	x = Dense(64, activation="relu")(x)
	x = Dropout(0.1)(x)
	x = Dense(1, activation="sigmoid")(x)
	model = Model(inputs=inp, outputs=x)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	es = EarlyStopping(monitor='val_loss',
	                   min_delta=0,
	                   patience=3,
	                   verbose=0, mode='auto')
	checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
	return model, [es, checkpoint]

def next_filename():
	file_nums = [int(file.replace('model', '').replace('.h5', '')) 
				for file in os.listdir("models")]
	return 'models/model' + str(max(file_nums) + 1) + '.h5'

best_model = next_filename()
model, callbacks = get_model(best_model)

# Fit the model
model.fit(X_t, y, batch_size=1024, epochs=30, callbacks=callbacks, validation_split=0.1)

# Load best model and predict
# model = load_model(best_model)
# print ('**Predicting on test set**')
# pred = model.predict(X_te, batch_size=1024, verbose=1)
# submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = pred
# submission.to_csv('preds/submission15.csv', index=False)
