from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_eval

import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.platform import flags
import os
import pandas as pd
import numpy as np


from basic_models import basic_cnn
from handle_text import text_to_vector

# Eventually parameters to a function instead of globals
word_size = 300
sentence_size = 200
source_samples = 480
nb_classes = 2
batch_size = 32

TRAIN_DATA ='train.csv'

# Load data into pandas
train = pd.read_csv(TRAIN_DATA, usecols=["id", "comment_text", "toxic"])

# Replace missing values in training and test set
list_train = train['comment_text'].fillna('_na_').values
list_y = train['toxic'].values
text_x_test = list_train[0:source_samples]
y_test = list_y[0:source_samples]
y_test = np.stack((y_test, np.ones(source_samples)-y_test), axis=1)
# y_test[y_test == 0] = 0.1
# y_test[y_test == 1] = 0.9 
print("Converting text to vectors")
x_train, x_test = text_to_vector(text_x_test, text_x_test)
# x_test = np.random.rand(source_samples, sentence_size, word_size)
print("Shape of x_test")
print(x_test.shape)
print("Shape of y_test")
print(y_test.shape)


# Set up random seeds for reproducibility and initialize session
keras.layers.core.K.set_learning_phase(0)
tf.set_random_seed(1234)

sess = tf.Session()
keras.backend.set_session(sess)

# Load test data, also eventually parameters
# x_test = np.random.rand(source_samples, sentence_size, word_size)
# y_test = np.stack((np.zeros(source_samples), np.ones(source_samples)),
# 				   axis=1)



# Placeholder tensors
x = tf.placeholder(tf.float32, shape=(None, sentence_size, word_size))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

# Instantiate model, probably also a parameter

print("Defined Tensorflow model graph")

saved_model_path = 'models/model7.h5'
# saver = tf.train.Saver()
# saver.restore(sess)
trained_model = load_model(saved_model_path)
preds = trained_model(x)
print("Preparing trained model for attack")

# fgsm = FastGradientMethod(wrap, sess=sess)
# fgsm_params = {'eps': 0.3,
# 			   'clip_min': 0.,
# 			   'clip_max': 1.}
wrap = KerasModelWrapper(trained_model)
cw = CarliniWagnerL2(wrap, back='tf', sess=sess)
cw_params = {'binary_search_steps': 1,
             'y': y_test,
             'max_iterations': 5,
             'learning_rate': 0.1,
             'batch_size': batch_size,
             'initial_const': 10}


print("Performing attack")
adv_x = cw.generate_np(x_test, **cw_params)

print("Making sure adversarial samples were extracted correctly")
print(type(adv_x))
print(adv_x.shape)

print("Finished attack, now for evaluation")
# adv_x = tf.stop_gradient(adv_x)
adv_preds = trained_model.predict(adv_x)
final_preds = np.around(adv_preds[:,0])
final_test = np.around(y_test[:,0])
correct = np.sum(final_preds == final_test)
print("Adversarial accuracy = " + str(correct/source_samples))
print(x_test[0])
print(adv_x[0])
lead_count=0
total_count=0
unk = np.zeros(300)
# for word in x_test[0]:
# 	if not np.array_equal(word, unk):
# 		break
# 	else:
# 		lead_count += 1
indices = []
for word_index in range((x_test[0].shape)[0]):
	if np.array_equal(x_test[0][word_index], unk):
		indices.append(word_index)
# print("Number of leading 0s is: " + str(lead_count))	
print("Indices where the word is 0s: ")
print(indices)
# adv_x = tf.stop_gradient(adv_x)
# adv_preds = trained_model(adv_x)
# print("Finished attack, now for evaluation")

# eval_par = {'batch_size': batch_size}
# acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
# print('Test accuracy on regular examples: %0.4f\n' % acc)


# print(adv_preds)
# adv_acc = model_eval(sess, x, y, adv_preds, x_test, y_test, args=eval_par)
# print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

sess.close()