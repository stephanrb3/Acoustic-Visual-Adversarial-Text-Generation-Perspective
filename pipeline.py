import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from filenames import model_file, data_file
from attack_model import carlini_attack
from select_features import select_features
from candidate import generate_candidates
from keras.models import load_model
from choose_input import choose_input
from evaluate import evaluate
import numpy as np
import pickle
import math
from functools import reduce
import sys 
warnings.filterwarnings("ignore")

def nCr(n, r):
	f = math.factorial
	return f(n)//(f(r) * f(n-r))


def num_candidates(e, n): 
	return n**e
	# return reduce((lambda x, y: x + (nCr(e, y) * n**y)), range(0, e+1))


def pipeline(model_type, attack_type, edit_distance, num_queries, num_neighbors=2):
	''' The pipeline uses the Carlini attack to generate adversarial features.
		The most perturbed words are then used to get a list of candidate sentences.
		Those candidate sentences are used to create adversairal examples which are 
		queryed by Perspective to calculate the change in accuracy and then returns it.'''
	#Load model and input/labels
	model_filename = model_file(model_type, num_queries)
	print("Model is: ",model_filename)
	x_test = np.load(data_file(num_queries)) ##inputs  
	y_test = np.load('embed_text/fb_test_y.npy') ## labels

	print("Finished loading model, now conducting attack")
	# Use attack module to generate adversarial features
	adv_features = carlini_attack(model_filename, x_test, y_test, 
								  batch_size=20)

	print("Finished perturbing features with Carlini attack")

	# Select words that have been perturbed most between
	# orig_features and adv_features and returns the indices
	# of the selected words. 
	selected_rows = select_features(edit_distance, x_test, adv_features)
	# print("Selected Rows Shape", selected_rows.shape)
	print("Found the most perturbed words from the attack")

	# Generate possible text candidates for each comment 
	print("num neighbors: ", num_neighbors)
	n_candidates = num_candidates(edit_distance, num_neighbors)
	candidate_text, candidate_features = (generate_candidates(testing_data, selected_rows, adv_features,  x_test, num_neighbors, attack_type, edit_distance))
	num_text = len(candidate_text)
	print("Got a list of " + str(num_text) + " candidate sentences," +  
		  " now to choose among them")

	# Maximizes class flip on the surrogate model while making sure that
	# the candidate text is close to the original text syntactically and
	# semantically.
	adv_inputs = choose_input(model_filename, test_comments, candidate_features, 
							  candidate_text, n_candidates, attack_type)
	print("Final adversarial example has been chosen for each toxic comment")
	# print("Final inputs: ", adv_inputs)
	# print("Original Comments", testing_data)

	# Queries our adversarial inputs to Perspective and reports back a success
	# rate of classification flip.
	print("Now Evaluating the adversarial examples by querying Perspective now")
	adv_accuracy = evaluate(testing_data, testing_scores, adv_inputs, edit_distance, num_queries, n_candidates)
	# adv_accuracy = "NULL"
	print("Change in accuracy on adversarial samples: " + str(adv_accuracy))  #add str
	return adv_accuracy
	

TEST_DATA = pickle.load(open('test_500_075.pkl', 'rb'))
testing_data = TEST_DATA['comment'].values
test_comments = list(testing_data)
testing_scores = TEST_DATA['toxicity'].values

if __name__ == "__main__":
	# default_params = [('model_type': 'basic_cnn'),
	# 				  ('edit_distance': 3),
	# 				  ('num_queries': 50000),
	# 				  ('num_neighbors': 2)]
	# index = int(input("Please enter the number of the parameter to vary: \n\n"
	# 				  + "1. Model Type - basic_cnn, basic_lstm, basic_gru\n"
	# 				  + "2. Edit Distance - Any integers\n"
	# 				  + "3. Number of Queries - 5k, 10k, 20, 50k, 100k\n"
	# 				  + "4. Number of Candidate Words Tried - Any integer\n"))
	# if index == 2 or index == 4:
	# 	user_list = str(input("Please enter some values to test: "))

	## Attack type is type of attack, word, acoustic, or visual 
	attack_type = sys.argv[1]
	#edit_distance = int(sys.argv[2])
	#print(type(edit_distance))
	#num_neighbors = sys.argv[3]
	#num_queries = sys.argv[4] 
	#num_queries = num_queries*1000
	TEST_DATA = pickle.load(open('test_500_075.pkl', 'rb'))
	# edit_distances = [5]
	query_totals = [10000, 20000, 50000,100000]
	neighbor_counts = [2, 3, 4, 5, 10]
	accuracies = []
	# for neighbor_count in neighbor_counts:
	acc = pipeline('basic_cnn', attack_type=attack_type, edit_distance=3, num_queries=300000, num_neighbors=3)
	accuracies.append(acc)
	pickle.dump(accuracies, open('results/NewResults_NB5.pkl', 'wb'))

	# pickle.dump(accuracies, open('results/neighbors_edit2_query10000.pkl', 'wb'))