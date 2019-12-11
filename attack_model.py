from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cleverhans.attacks import CarliniWagnerL2, SaliencyMapMethod
from cleverhans.utils_keras import KerasModelWrapper

import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.platform import flags

import pandas as pd
import numpy as np


from handle_text import text_to_vector


def attack_model(alg, params, model_file, x_test, y_test):

	source_samples = (y_test.shape)[0]
	nb_classes = 2
	
	# Set up random seeds for reproducibility and initialize session
	keras.layers.core.K.set_learning_phase(0)
	tf.set_random_seed(1234)
	sess = tf.Session()
	keras.backend.set_session(sess)

	model = load_model(model_file)
	wrap = KerasModelWrapper(model)
	
	Attack_Method = globals()[alg]
	attack = Attack_Method(wrap, back='tf', sess=sess)
	adv_features = attack.generate_np(x_test, **params)
	adv_preds = model.predict(adv_features)
	final_preds = np.around(adv_preds[:,0])
	final_test = np.around(y_test[:,0])
	correct = np.sum(final_preds == final_test)
	print("Adversarial accuracy = " + str(correct/source_samples))

	return adv_features


def carlini_attack(model_file, x_test, y_test,  batch_size=32, 
				   search_steps=1, max_iters=10, lr=0.1, init_constant=10):
	"""
	This function applies the Carlini attack to a certain model given
	some input data and labels, and returns a Numpy array of perturbed
	inputs. The final 4 arguments use default values from the Cleverhans
	documentation.

	Arguments:
	- model_file: An .h5 file storing a trained Keras model
	- x_test: A Numpy array of input data
	- y_test: A Numpy array of corresponding labels
	- batch_size: Number of inputs to attack simultaneously (default=32)
	- search_steps: Number of times to perform a binary search to find
	the best tradeoff between distortion and confidence
	- max_iters: Number of iterations (higher value means less distortion)
	- lr: Learning rate (higher value means faster but worse results)
	- init_constant: Initial tradeoff constant for distortion-confidence
	(higher value means more distortion)
	"""

	params = {'binary_search_steps': search_steps,
              'y': y_test,
              'max_iterations': max_iters,
              'learning_rate': lr,
              'batch_size': batch_size,
              'initial_const': init_constant}
	return attack_model('CarliniWagnerL2', params, model_file, 
						x_test, y_test)


def jsma_attack(model_file, x_test, y_test, theta=1., g=0.1, cmin=0., cmax=1.):
	"""
	NOT FUNCTIONAL: This function does not yet work. Do not use.
	"""
	params = {'theta': theta, 
			  'gamma': g, 
			  'clip_min': cmin, 
			  'clip_max': cmax, 
			  'y_target': y_test[:,[1, 0]]}
	return attack_model('SaliencyMapMethod', params, model_file, 
						x_test, y_test)



# train = pd.read_csv('train.csv', usecols=['id', 'comment_text', 'toxic'])
# text = (train['comment_text'].fillna('_na_').values)[0:480]
# _, x_test = text_to_vector(text, text)
# toxicity = (train['toxic'].values)[0:480]
# y_test = np.stack((toxicity, np.ones(480)-toxicity), axis=1)
# adv_features = jsma_attack('models/model7.h5', x_test, y_test)
# print(adv_features)