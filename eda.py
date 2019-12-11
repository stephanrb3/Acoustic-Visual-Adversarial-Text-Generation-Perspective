import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt 
import pickle 

import sys, os, re, csv, codecs
import pandas as pd 
import numpy as np
import pickle

import scipy.stats as ss

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
#import matplotlib_venn as venn 

#fb_data = pickle.load(open('data/fb_data.pkl', 'rb'))
#fb_comments = fb_data['comment'].values

#print(fb_comments)

lengths = pickle.load(open("data/tmp_commentlengths.pkl", "rb"))#[len(re.sub("[^\w]", " ", sentence).split()) for sentence in fb_comments]

#pickle.dump(lengths, open('data/tmp_commentlengths.pkl', 'wb'))

lengthsbins = np.append(np.arange(0, 161, 1), [np.inf])

plt.hist(lengths, bins = lengthsbins)
plt.xlabel("Number of words in comment")
plt.ylabel("Number of comments")
plt.title("Frequency Distribution of Comments by Word Count")
plt.savefig("wordcount_frq.png")
plt.show()

plt.hist(lengths, bins = lengthsbins, normed = True, cumulative = True)
plt.xlabel("Number of words in comment")
plt.ylabel("Total fraction of comments")
plt.title("Cumulative Distribution of Comments by Word Count")
plt.savefig("wordcount_cml.png")
plt.show()

sys.exit(0)

# Save training and testing data
TRAIN_DATA ='train.csv' 
TEST_DATA ='test.csv'
SAMPLE_SUB ='sample_submission.csv'

print('data saved')


# Load data into pandas

rating_df = pd.read_pickle("./rating_list.pkl")

rating_df = np.array(rating_df)
rating_df = rating_df[rating_df != -1]
print(rating_df)   

print(np.count_nonzero(rating_df == -1))

# the histogram of the data
#n, bins, patches = plt.hist(rating_df, density=True, facecolor='g')
#weights = np.ones_like(rating_df)/float(len(rating_df))
#plt.hist(rating_df, weights=weights)


# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(rating_df, 50, facecolor='g', alpha=0.75)

weights = np.ones_like(rating_df)/float(len(rating_df))
plt.hist(rating_df, weights=weights)

plt.xlabel('Toxicity')
plt.ylabel('Density')
plt.title('Histogram of Toxic Comment Density')

plt.savefig('Toxicity.png')   # save the figure to file
plt.show()

#print(np.sum(hist * np.diff(bins)))
#print('pandas loaded')

