import sys, os, re, csv, codecs, numpy as np, pandas as pd


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential


# Load pre-trained word vectors
EMBEDDING ='data/glove.840B.300d.txt' 
# EMBEDDING='data/crawl-300d-2M.vec'

# Save training and testing data
TRAIN_DATA ='train.csv'

# Load data into pandas
train = pd.read_csv(TRAIN_DATA, usecols=["id", "comment_text", "toxic"])

# Replace missing values in training and test set
list_train = train["comment_text"].fillna("_na_").values


def tokenize(training_data, testing_data, sentence_size, vocab_size):
	# Return training and testing data tokenized, along with the tokenizer.
	# Every word is replaced by an integer representing a word index.

	tok = Tokenizer(num_words=vocab_size, filters='')
	tok.fit_on_texts(list(training_data))
	tok_train = tok.texts_to_sequences(training_data)
	tok_test = tok.texts_to_sequences(testing_data)

	# Pad each comment with blank words if shorter than maxlen
	x_train = pad_sequences(tok_train, maxlen=sentence_size)
	x_test = pad_sequences(tok_test, maxlen=sentence_size)
	return x_train, x_test, tok


def create_embedding(tok, word_size, vocab_size): 
	# From a tokenizer, construct an embedding matrix with dimensions
	# (nb_words, word_size). The minimum of vocab_size and the number of
	# words in the tokenizer is chosen as nb_words. 

	# The embedding matrix is return along with nb_words.

	def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
	with open(EMBEDDING, 'r') as embed_file: 
		embeddings_index = dict(get_coefs(*o.strip().split(" ")) 
								for o in embed_file)

	word_index = tok.word_index
	nb_words = min(vocab_size, len(word_index)+1)
	embedding_matrix = np.zeros((nb_words, word_size))
	for word, i in word_index.items():
	    if i >= vocab_size: continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

	return embedding_matrix, nb_words


def get_embedded_text(tok_text, embedding_matrix,
					  word_size, sentence_size, vocab_size):
	# Use an embedding matrix to turn tokenized text data into 
	# embedded vectors with dimensions (samples, sentence_size, word_size).

	model = Sequential()
	model.add(Embedding(vocab_size, word_size, weights=[embedding_matrix],
						input_length=sentence_size, trainable=False))
	model.compile('rmsprop', 'mse')
	embedded_vectors = model.predict(tok_text, batch_size=1)
	return embedded_vectors


def text_to_vector(train, test, sentence_size, word_size, vocab_size):
	"""
	Returns train and test text data in a vectorized form (Numpy array).

	Arguments:
	- train: List of text comments to train on
	- test: List of text comments (words not in train are treated as unknown)
	- sentence_size: Number of words per comment (default=200)
	- word_size: Number of dimensions per word vector (default=300)
	- vocab_size: Maximum number of words in the vocabulary (default=150000)s
	"""

	tok_train, tok_test, tok = tokenize(train, test, 
									sentence_size=sentence_size, 
									vocab_size=vocab_size)
	embedding_matrix, vocab_size = create_embedding(tok, 
												    word_size=word_size,
												    vocab_size=vocab_size)
	x_train = get_embedded_text(tok_train, embedding_matrix, 
								word_size=word_size, vocab_size=vocab_size,
					  			sentence_size=sentence_size)
	x_test = get_embedded_text(tok_test, embedding_matrix, 
								word_size=word_size, vocab_size=vocab_size,
					  			sentence_size=sentence_size)
	return x_train, x_test


def handle_text(training_data, testing_data, sentence_size=100, 
				word_size=300, vocab_size=150000):
	"""
	Creates Numpy arrays for training and testing features and labels
	from two Pandas dataframes with comments and their toxicity scores.
	Returns x_train, y_train, x_test, y_test (scikit-learn convention).

	Arguments:
	- training_data: Pandas dataframe
	- testing_data: Pandas dataframe
	""" 
	train_comments = training_data.iloc[:,0]
	test_comments = testing_data.iloc[:,0]
	x_train, x_test = text_to_vector(train_comments, test_comments, 
									 sentence_size, word_size, vocab_size)
	y_t = training_data.iloc[:,1].values
	y_train = np.stack((y_t, np.ones((y_t.shape)[0]) - y_t), axis=1)
	y_te = testing_data.iloc[:,1].values
	y_test = np.stack((y_te, np.ones((y_te.shape)[0]) - y_te), axis=1)
	return x_train, y_train, x_test, y_test


# train_text = list_train[0:500]
# test_text = list_train[500:600]
# train, test = text_to_vector(train_text, test_text)
# print(train.shape)
# print(test.shape)