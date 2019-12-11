import numpy as np
from annoy import AnnoyIndex
import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import ngtpy ##Pybind11
#from ngt import base as ngt ##Undefined Symbols/Ctypes
from tqdm import tqdm
import time
import progressbar

# Usage: 
# in terminal: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
# Manually run this file once to create the graph index.
# Call the static method nearestneighbors to search for neighbors
# to a vector.


# vector: 1D array-like object of length 300. 
# number: integer.
# output: list object containing <number> words corresponding to
# the <number> closest vectors in the embedding to <vector>, 
# according to angular distance/cosine similarity.

def print_nearest(word):
    for idx in index.get_nns_by_vector(test_vectors[word],10):
        print(words[idx])

def print_neighbors(vector, number):
    for idx in index.get_nns_by_vector(test_vectors[word],10):
        print(words[idx])

# Query Usage: 
# Uses annoy index to find nearestneighbors of a vector.
# vector: 1D array-like object of length 300 corresponding to the word to be changed 
# factor: How many different words to substitute per index.
# words: a list of words from the index
# Outputs neighbor words and their corresponding vectors
def query(vector, factor, speech = True):
    # filepath = "data/glove.840B.300d.txt" #hard code
    idx = 0
    if speech == True:
        index = AnnoyIndex(300, 'euclidean')
        index.load('data/Speech.ann')
    else:
        index = AnnoyIndex(300, 'euclidean')
        index.load('data/840b.ann')

    outputw = []
    outputv = []
    if speech == True:
        f = open('data/speechwords.txt', 'r')
    else:
        f = open('data/words.txt', 'r')
    # f = open('speech.txt', 'r')

    try:
        words = f.readlines()
    except:
        # print("oh no unicode")
        if speech == True:
            f = open('data/speechwords.txt', 'r', encoding="utf-8")
        else:
            f = open('data/words.txt', 'r', encoding="utf-8")
            words = f.readlines()
    f.close()

    for idx in index.get_nns_by_vector(vector, factor, search_k=100):
        #print("Nearest Neighbor word ", idx, ":", words[idx].encode("utf-8"))
        outputw.append(words[idx])
        outputv.append(index.get_item_vector(idx))

    #print("nn query complete")
    #print("vectors: ", outputv, "words: ", outputw)
    return outputw, outputv

def main():
    pass

def annoy_build(word_vectors, speech=True):
    idx = 0
    if speech == True:
        index = AnnoyIndex(300, 'euclidean')
    else:
        index = AnnoyIndex(300, 'euclidean')
    ###########################################################################################
    ### In this case we take word_vectors which looks like                                  ###
    ###    {("dog", 1.00 4.22 3.95...), ("cat", 4.40 2.22 7.33...), ("kind", 6.32 3.45...)} ###
    ### and outputs words= [dog, cat, kind] and vectors[1.00 4.22 3.95..., 4.40 2.22 7.33..]###                   
    ###########################################################################################
    words, vectors = zip(*word_vectors.items())

    if speech == True:
        num_lines = sum(1 for line in open('data/speechwords.txt', 'r', encoding='utf-8'))
        with open('data/speechwords.txt', 'w', encoding='utf-8') as out:
            with tqdm(total=num_lines) as pbar:
                pbar.set_description("Writing index")
                for item in words:
                    out.write("%s\n" % item)
                    pbar.update(1)
        pbar.close() 
    else:
        num_lines = sum(1 for line in open('data/words.txt', 'r', encoding='utf-8'))
        with open('data/words.txt', 'w', encoding='utf-8') as out:
            with tqdm(total=num_lines) as pbar:
                pbar.set_description("Writing index")
                for item in words:
                    out.write("%s\n" % item)
                    pbar.update(1)
        pbar.close() 

    #### Then we add the vectors to annoy index for building
    for idx, vector in enumerate(vectors):
        index.add_item(idx, vector)

    if speech == True:
        index.build(100) # trees
        index.save('data/Speech.ann')
    else:
        index.build(100) # trees
        index.save('data/840b.ann')

    print("index saved")
    return words

if __name__ == "__main__":
    pass

