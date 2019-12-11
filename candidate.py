import numpy as np
import nearestneighbors
import re
from tqdm import tqdm
import time
from choose_input import choose_input
import pickle 
import json 
import homoglyphAttack
import phonemeAttack

def sentence_to_array(sentence):
    return np.array(re.sub("[^\w]", " ", sentence).split())


def create_word_vectors(speech=True):
    word_vectors = {}
    if speech == True:
        file = "data/s2v_300.txt"
    else:
        file = "data/glove.840B.300d.txt"

    num_lines = sum(1 for line in open(file,"r",encoding="utf-8"))
    with open(file, encoding="utf-8") as f:
        with tqdm(total=num_lines) as pbar:
            pbar.set_description("Copying index from embedding")
            for line in f:
                if speech == True:
                    word = line.split()[0]
                    values = np.array(line.split()[1:]).astype(float)
                else:
                    splitLine = line.split(' ')
                    word = splitLine[0]
                    values = np.asarray(splitLine[1:]).astype(float)
                #print(word)
                word_vectors[word] = values
                pbar.update(1)

    pbar.close() 
    # print(word_vectors)
    return word_vectors

def process_sentences(sentence, indices, replacement, original, factor, attack_type, edit_distance):
    """
    Takes in a list of words and a numpy array of indices to replace, and returns
    a list of sentences with replaced words and the corresponding matrices.
    Each matrix is zero-padded so that it is 100x300 dimensions.

    @param sentence: Numpy 1d array of strings. A sentence. Each string is a word.
    @param indices: Numpy 1d array of integers. Indices of the word(s) to replace. 
            The last word in the sentence always has index 99.
    @param replacement: Adversarial tensor (2D numpy array) shape (100, 300). Each row in the tensor
            corresponds to a word in sentence.
    @param original: Original tensor (2D numpy array) shape (100, 300). See replacements.
    @param factor: Integer. How many different words to substitute per index.
    @param wordlist: list of words in index 
    """    
    #print("object align?")
    #print(indices.size)
    if not indices.size:
        return [], []
    outputw = []
    outputv = []
    num_words = replacement.shape[0]
    #print("n_words: ", num_words)
    #print(word_list)
    index = indices[0]

    if attack_type == "word":
        neighborwords, neighborvectors = nearestneighbors.query(replacement[index], factor, speech=False)

    sentencecopy = np.copy(sentence)
    replaceable_word = sentencecopy[index + sentence.shape[0] - num_words]
    
    if attack_type == "acoustic":
        neighborwords = phonemeAttack.Find_Phonemes(replaceable_word, factor, edit_distance) 
        
    if attack_type == "visual":
        neighborwords = homoglyphAttack.Find_Homoglyphs(replaceable_word, factor, edit_distance) 

    # print(neighborwords)
    originalcopy = np.copy(original)
    #print("Querying word: ", sentencecopy[])
    #print("index: ", index, "index.shape: ", index.shape ,"sentence copy: ", sentencecopy, "num words: ", num_words, "neighborwords: ", neighborwords, "sentence.shape: ", sentence.shape)
    for i in range(factor):
        if attack_type == "acoustic" or attack_type == "visual":
            sentencecopy[index + sentence.shape[0] - num_words] = neighborwords

        if attack_type == "word":
            sentencecopy[index + sentence.shape[0] - num_words] = neighborwords[i]
            originalcopy[index] = neighborvectors[i]

        sentencestring = ""
        for word in sentencecopy:
            sentencestring += word + " "
        sentencestring = sentencestring.rstrip()
        
        if indices.size == 1:
            outputw.append(sentencestring)
            if attack_type == "word":
                outputv.append(original)
        

        subw, subv = process_sentences(sentencecopy, indices[1:], replacement, originalcopy, factor, attack_type, edit_distance)
        outputw += subw
        if attack_type == "word":
            outputw += subw
            outputv += subv
    return outputw, outputv


def generate_candidates(sentences, indices, replacements, originals, factor, attack_type, edit_distance):
    """
    Same thing as process_sentences but for a lot of sentences all together. Also, the sentence inputs
    are full strings instead of lists of words.

    @param sentences: Numpy 1d array of strings. Each string is a sentence.
    @param indices: Numpy 2D array of integers. Each list of integers corresponds to indices to replace for a given sentence.
    @param replacement: Numpy array of adversarial matrices (Numpy 2d arrays). Each row in the matrix
            corresponds to a word in a sentence.
    @param original: Numpy array of original matrices (Numpy 2d arrays). See replacements.
    @param factor: Integer. How many different words to substitute per index.
    """

    outputws = []
    outputvs = []

    ### Create Index of the Embedding for Annoy --- Only Run once to create index file
    # wordv = create_word_vectors(speech=False)
    # word_list = nearestneighbors.annoy_build(wordv, speech=False)


    with tqdm(total=sentences.shape[0]) as pbar:
        pbar.set_description("Querying Nearest Neighbors of Words to be Replaced in each sentence")
        for i in range(sentences.shape[0]):
            sentencearray = sentence_to_array(sentences[i])
            subws, subvs = process_sentences(sentencearray, indices[i], replacements[i], originals[i], factor, attack_type, edit_distance)
            outputws += subws
            outputvs += subvs
            pbar.update(1)
    pbar.close()
    
    # print("Output words: ", outputws)
    # print("Output Vectors: ", outputvs)
    return outputws, np.array(outputvs)

"""
attempt at unit test
"""
if __name__ == '__main__':
    s = np.array(["the big brown the the lifestyle", "and the the they all died horribly and", "they should the the have settled for the the the hills they", "read it and weep", "please let my suffering end"])
    i = np.array([[98, 99], [98, 99], [98, 99], [98, 99], [98, 99]])
    r = np.ones((5, 100, 300))
    l = np.zeros((5, 100, 300))

    factor = 2
    edit_distance = 3
    w, v = generate_candidates(s, i, r, l, factor, "visual", edit_distance)

    # print(v.shape)
    # print(w)
    print("num neighbors: ", factor)
    print("Got a list of " + str(len(w)) + " candidate sentences," +  
		  " now to choose among them")

	# # Maximizes class flip on the surrogate model while making sure that
	# # the candidate text is close to the original text syntactically and
	# # semantically.
    TEST_DATA = pickle.load(open('test_500_075.pkl', 'rb'))
    testing_data = TEST_DATA['comment'].values
    test_comments = list(testing_data)

    adv_inputs = choose_input("models/basic_cnn_10k.h5", test_comments, v, w, factor, "visual")
    
    print(adv_inputs)
    print("Final adversarial examsple has been chosen for each toxic comment")