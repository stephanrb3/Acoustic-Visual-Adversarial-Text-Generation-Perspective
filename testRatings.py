import pickle

pickle_in = open("rating_list.pkl","rb")
ratings = pickle.load(pickle_in)

print(len(ratings))