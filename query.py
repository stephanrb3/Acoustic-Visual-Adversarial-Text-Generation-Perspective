from googleapiclient import discovery, errors
import pandas as pd

import json
import time
import pickle

# Note that the actual quota is 1,000 API calls every 100 seconds.
QUOTA = 980
WAIT_TIME = 100
API_KEY='AIzaSyBBRvTXGdGc1im1KxMEJQxqqGuQCMmfKfY'
WIKI_DATA='train.csv'
FB_DATA='data/fb_news_comments_1000K.csv'


# Generates API client object dynamically based on service name and version.
service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)

# Handle Wikipedia comments data
wiki = pd.read_csv(WIKI_DATA)
wiki_train = wiki["comment_text"].dropna()

fb = pd.read_csv(FB_DATA)
# Original length: 1038319
fb_train = (fb['message'].str.replace('http\S+', '', case=False)).dropna()
# Postprocessing: 1011597


def rate_comments(comments, ratings):
	# Rates one fold of comments and adds their ratings to a passed-in list
	
	for comment in comments:
		analyze_request = {
	  		'comment': { 'text': comment },
	  		'requestedAttributes': {'TOXICITY': {}}
	  	}
		try:
			response = service.comments().analyze(body=analyze_request).execute()
			ratings.append(response['attributeScores']['TOXICITY']['summaryScore']['value'])
		except errors.HttpError:
			ratings.append(-1)
	return ratings


def get_toxicity (comments):
	""" 
	Takes in a list or a Pandas dataframe of string comments
	in English and returns a list of corresponding toxicity scores. If 
	a comment is not recognized as English, then its rating will be -1.
	This function also periodically saves the ratings in a pickled list
	in case the query process gets interrupted. 
	"""
	
	ratings = []
	comments = list(comments) # Convert dataframe or string into list	
	num_comments = len(comments)
	print("Number of total comments: " + str(num_comments+1))
	start_time = time.time()
	ratings = rate_comments(comments[0:QUOTA], ratings)
	folds = ((num_comments-1)//QUOTA)
	print("Number of folds: " + str(folds))
	

	for f in range(folds):
		print("Number of comments rated: " + str((f+1)*QUOTA))
		comment_fold = comments[(f+1)*QUOTA:(f+2)*QUOTA]
		end_time = time.time()
		delay = max(WAIT_TIME-(end_time-start_time), 0)
		print("Waiting " + str(delay)
			  + " more seconds for next fold")
		time.sleep(delay)
		start_time = time.time()
		rate_comments(comment_fold, ratings)
		if (f+1)%8 == 0:
			print("Saved " + str((f+1)*QUOTA) + 
				  " rated comments in a pickle file")
			with open('rating_list.pkl', 'wb') as file:
				pickle.dump(ratings, file)
	
	return ratings


def get_dataframe(comments, filename):
	"""
	Produces a pickled dataframe that contains comments along
	with their ratings as given by Google's Perspective API.


	Arguments:


	comments - A list or Pandas dataframe containing comments
	filename - A string filename to save to (no .pkl extension needed)
	"""
	toxicities = get_toxicity(comments)
	print("Finished rating comments")
	df = pd.DataFrame(data={'comment': comments, 
							'toxicity': toxicities})
	df = df[df.toxicity != -1]
	file = filename + '.pkl'
	df.to_pickle(file)
	print("Saved the comments and ratings in " + file)


def read_ratings(comments, ratings_file, new_file):
	"""
	In case the query process ended prematurely, this function
	reads in the incomplete ratings file and saves it with the
	corresponding comments in a pickled Pandas dataframe.


	Arguments:


	comments - Pandas dataframe of comments
	ratings_file - String filename of the incomplete ratings list
	new_file - String filename to save to (no .pkl extension needed)
	"""
	print("Opening up ratings file")
	ratings_list = pickle.load(open(ratings_file, 'rb'))
	num_ratings = len(ratings_list)
	# print("Num Ratings: ",num_ratings)
	comments_list = list(comments.iloc[0:num_ratings])
	# print(len("Num Comments: ", comments_list))
	print("Creating dataframe of comments and toxicities")
	df = pd.DataFrame(data={'comment': comments_list,'toxicity': ratings_list})
	df = df[df.toxicity != -1]
	file = new_file + '.pkl'
	print("Saving dataframe in " + file)
	df.to_pickle(file)
	print("Finished!")


# get_dataframe(fb_train, 'fb_data')
# read_ratings(fb_train, 'rating_list.pkl', 'fb_rated')

# print(get_toxicity(wiki_train.iloc[0:100]))

# print(get_dataframe(['friendly greetings from python', 'you are an idiot good sir']))