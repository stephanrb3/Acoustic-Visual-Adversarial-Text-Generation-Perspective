import spacy
import numpy as np
import pandas as pd
from keras.models import load_model
from query import get_toxicity
import operator
import itertools
import collections


def split_list(data, sublist_len):
	return [data[x:x+sublist_len] for x in range(0, len(data), sublist_len)]


def spacy_similarity(original_text, candidate_text, num_candidates):
	"""
	This function calculates syntactic similarity between original
	comment and some candidate variants. It returns Numpy array with
	dims (samples, num_candidates).

	Arguments:
	- original_text: A list of comments
	- candidate_text: A list of of comments (larger than original_text
	by a factor of num_candidates)
	- num_candidates: The number of candidates per comment
	"""
	
	nlp = spacy.load('en_core_web_lg')
	originals = list(map(nlp, original_text))
	candidates = split_list(list(map(nlp, candidate_text)), num_candidates)

	def spacy_score(orig, candidates):
		return [orig.similarity(candidate) for candidate in candidates]

	return np.array(list(map(lambda z: spacy_score(*z), list(zip(originals, candidates)))))


def class_flip(model_file, candidate_features, num_candidates):
	"""
	Returns a Numpy array of a surrogate model's ratings for adversarial 
	candidates, with dims (samples, num_candidates).

	Arguments:
	- model_file: A filename for a trained Keras model ending in .h5
	- candidate_features: A Numpy array with adversarial text candidates
	as embedded vectors
	- num_candidates: Number of candidates generated per comment
	"""
	# print(candidate_features)
	model = load_model(model_file)	
	results = model.predict(candidate_features)[:,1]
	# print(len(results))
	# print(results.shape)
	# print(results)	
	# print(len(candidate_features))
	# print(np.reshape(results, (-1, num_candidates)).shape)
	# print(np.reshape(results, (-1, num_candidates))

	return np.reshape(results, (-1, num_candidates))

def merge(list1, list2): 
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list 
      

def class_flip_for_Char(candidate_text, num_candidates):
	"""
	Returns a Numpy array of perspectives ratings for adversarial 
	candidates, with dims (samples, num_candidates).

	Arguments:
	- candidate_text: A list of of comments (larger than original_text
	by a factor of num_candidates
	- num_candidates: Number of candidates generated per comment
	"""
	# print(num_candidates)
	# print(candidate_text)
	new_candidates = []
	advscores = get_toxicity(candidate_text)
	# print(advscores)
	# zipped = zip(candidate_text, advscores)
	# adv_score_dict = dict(zipped)
	c = [candidate_text,advscores]
	adv = merge(candidate_text,advscores)
	print(adv[80])
	for i in range (500):
		if(bool(adv)):
			first_n_pairs = adv[:8]
			# print("\nfirst n pairs from dict: ", first_n_pairs)
			# for item in first_n_pairs:
			a = max(first_n_pairs, key=lambda item:item[1])
			# print("max: ", a)
			new_candidate = a[0]
			new_candidates.append(new_candidate)
			del adv[:8] 
		else:
			break

			# print(new_candidate)
		# print(adv_score_dict)

	return new_candidates


# 	# print(len(candidate_text))
# 	n_samples = len(candidate_text)/num_candidates
# 	# print("n_samples", n_samples)
# 	# print("adv score dict ", adv_score_dict)
# 	# print("adv score dict length", len(adv_score_dict))
# 	for i in range (500):
# 		if(not adv_score_dict):
# 			break
# 		else:
# 			first_n_pairs = {k: adv_score_dict[k] for k in list(adv_score_dict)[:2]}
# 			print("\nfirst n pairs from dict: ", first_n_pairs)
# 			new_candidate = max(first_n_pairs, key=lambda k: first_n_pairs[k])
# 			print(new_candidate)
# 			new_candidates.append(new_candidate)
# 			for key in first_n_pairs:
# 				# print(len(new_candidates))
# 				adv_score_dict.pop(key)
# 		# print(adv_score_dict)

# 	return new_candidates



def choose_input(model_file, test_data, candidate_features, 
				 candidate_text, n_candidates, attack_type):
	# print(len(candidate_text),len(candidate_features))
	# print("Log class flip ", np.log(cf_score))
	if attack_type == "word":
		cf_score = class_flip(model_file, candidate_features, n_candidates)
		print("Class flip shape: ", cf_score.shape) #(2000,2)
		syntactic_score = spacy_similarity(test_data, candidate_text, n_candidates)
		print("Syntactic score shape: ", syntactic_score.shape)
		agg_score =  np.log(cf_score) + np.log(syntactic_score)
		max_indices = list(np.argmax(agg_score, axis=1))
		flat_indices = np.array([val + i*n_candidates 
								for i, val in enumerate(max_indices)])
		res =  (np.array(candidate_text))[flat_indices]
		return (np.array(candidate_text))[flat_indices]


	if attack_type == "acoustic" or attack_type == "visual":
		cf_score = class_flip_for_Char(candidate_text, n_candidates)
		print("Class flip candidates: ", len(cf_score))
		print(cf_score[10])
		return cf_score


if __name__ == "__main__":
    pass

# sentences = ["This is a sentence.", "So is this one."]
# candidates = ["This is a bad sentence.", "This is a good sentence",
# 			  "So is this two", "So are this one"]
# num_candidates = len(candidates)//len(sentences)
# scores = spacy_similarity(sentences, candidates, num_candidates)
# print(scores)