from query import get_toxicity
import re
import numpy as np
import pickle
from tqdm import tqdm


"""
given scores of original sentences and the corresponding adversarial sentences as numpy arrays,
returns the ratio of successful sentences to total sentences. Also prints the average word count, 
average original score, and average change in score for successes and failures separately
"""
def evaluate(originalsentences, scores, newsentences, edit, query, neighbors, save = True):
	success, failure = partition(originalsentences, scores, newsentences)

	if success.size:
		print("average word count of successes is " + str(average_length(success[:, 1])))
		print("average original score of successes is " + str(np.mean(success[:, 2].astype(np.float))))
		print("average score change of successes is " + str(average_difference(success[:, 2], success[:, 3])))

	if failure.size:
		print("average word count of failures is " + str(average_length(failure[:, 1])))
		print("average original score of failures is " + str(np.mean(failure[:, 2].astype(np.float))))
		print("average score change of failures is " + str(average_difference(failure[:, 2], failure[:, 3])))

	if save:
		pickle.dump(success, open("results/edit" + str(edit) + "_query" + str(query) + "_neighbors" + str(neighbors) + "_successes.pkl", "wb"))
		pickle.dump(failure, open("results/edit" + str(edit) + "_query" + str(query) + "_neighbors" + str(neighbors) + "_failures.pkl", "wb"))
		print("successes saved at results/edit" + str(edit) + "_query" + str(query) + "_neighbors" + str(neighbors) + "_successes.pkl, failures saved at edit" + str(edit) + "_query" + str(query) + "_neighbors" + str(neighbors) + "_failures.pkl")

	return success.shape[0] / (success.shape[0] + failure.shape[0])


"""
given scores of original sentences and corresponding adversarial sentences, returns 
2-dimensional numpy arrays of tuples for succesful sentences and failed sentences
column index 0 is original sentence, 1 is adversarial sentence, index 2 is original score, index 3 is adversarial score
"""
def partition(originalsentences, scores, newsentences):
	print(originalsentences.size)
	print(len(newsentences))
	print(originalsentences[10])
	print(newsentences[10])
	advscores = get_toxicity(newsentences)
	success = []
	failure = []

	for i in range(len(newsentences)):  ##### change to .size for word level
		if advscores[i] < 0.5:
			success.append([originalsentences[i].encode("utf-8").decode('utf-8'), newsentences[i], scores[i], advscores[i]])
			# print("Successful Sentence appended")
			# print("Original sentence: ", originalsentences[i].encode("utf-8"),"\n")
			# print("adv sentence: ", newsentences[i].encode("utf-8"),"\n") 
			# print("orig score: ", scores[i])
			# print("adv score: ", advscores[i],"\n","\n") 
		else:
			failure.append([originalsentences[i].encode("utf-8").decode('utf-8'), newsentences[i], scores[i], advscores[i]])
			# print("Failure appended: ")
			# print("Original Senctence: ", originalsentences[i].encode("utf-8") ,"\n")
			# print("adv sentence(No Flip): ", newsentences[i].encode("utf-8"),"\n") 
			# print("orig score: ", scores[i])
			# print("adv score: ", advscores[i] ,"\n","\n") 
	return np.array(success), np.array(failure)

def average_length(sentences):
	sum = 0
	for sentence in sentences:
		sum += len(re.sub("[^\w]", " ", sentence).split())
	return sum / len(sentences)

def average_difference(scores, advscores):
	return np.mean(np.abs(scores.astype(np.float) - advscores.astype(np.float)))

"""
unit testing
"""
if __name__ == "__main__":
	sentence = "the big kitty"
	score = 0.1
	sentences = np.array([sentence for i in range(5)])
	scores = np.array([score for i in range(5)])
	evaluate(sentence, scores, sentences, 3, 10, 2, save = False)

