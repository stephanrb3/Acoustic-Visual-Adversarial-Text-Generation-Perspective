import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import pickle
from itertools import cycle, islice


# neighbors = range(2,6)
# neighbor_acc = pickle.load(open('results/neighbors_edit3_query100000.pkl', 'rb'))

# edits = range(1,5)
# edit_acc = pickle.load(open('results/edits_query100000_neighbors3.pkl', 'rb'))

# queries = [10000, 20000, 50000, 100000]
# query_acc = pickle.load(open('results/queries_edit3_neighbors3.pkl', 'rb'))


# Successes = pickle.load(open('results/edit2_query100000_neighbors100_successes.pkl', 'rb'))
# Failures = pickle.load(open('results/edit2_query100000_neighbors100_failures.pkl', 'rb'))

# # query_acc = pickle.load(open('results/queries_edit3_neighbors3.pkl', 'rb'))
# df1 = pd.DataFrame(Successes)
# print(df1)
# df1.to_csv('gsuccesses.csv')
# df1 = pd.DataFrame(Failures)
# print(df1)
# df1.to_csv('gFails.csv')

# Successes = pickle.load(open('results/edit3_query100000_neighbors125_successes.pkl', 'rb'))
# Failures = pickle.load(open('results/edit3_query100000_neighbors125_failures.pkl', 'rb'))

# # query_acc = pickle.load(open('results/queries_edit3_neighbors3.pkl', 'rb'))
# df1 = pd.DataFrame(Successes)
# print(df1)
# df1.to_csv('ssuccesses.csv')
# df1 = pd.DataFrame(Failures)
# print(df1)
# df1.to_csv('sFails.csv')

# Successes = pickle.load(open('results/edit3_query200000_neighbors8_successes.pkl', 'rb'))
# Failures = pickle.load(open('results/edit3_query200000_neighbors8_failures.pkl', 'rb'))

# # query_acc = pickle.load(open('results/queries_edit3_neighbors3.pkl', 'rb'))
# df1 = pd.DataFrame(Successes)
# print(df1)
# df1.to_csv('Acsuccesses.csv')
# df1 = pd.DataFrame(Failures)
# print(df1)
# df1.to_csv('AcFails.csv')
# Successes = pickle.load(open('results/edit3_query300000_neighbors8_successes.pkl', 'rb'))
# Failures = pickle.load(open('results/edit3_query300000_neighbors8_failures.pkl', 'rb'))

# # query_acc = pickle.load(open('results/queries_edit3_neighbors3.pkl', 'rb'))
# df1 = pd.DataFrame(Successes)
# print(df1)
# df1.to_csv('Vsuccesses.csv')
# df1 = pd.DataFrame(Failures)
# print(df1)
# df1.to_csv('VFails.csv')

# # initialize list of lists 
data = [['Semantic', 30.4], ['Acoustic', 72.5], ['Visual', 33]] 
  
# # Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Attack_Type', 'Percent Dip in Perspective Accuracy']) 

# # df = pd.DataFrame({'effe': speed}, index=index)
# ax = df.plot.bar(y='Change in Accuracy', x="Attack_Type",rot=0)

# Make a list by cycling through the colors you care about
# to match the length of your data.
my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))

# Specify this list of colors as the `color` option to `plot`.
ax = df.plot(kind='bar',title="Effectiveness of Different Attack Types", y='Percent Dip in Perspective Accuracy', x="Attack_Type", legend=False, rot=0, color=my_colors)
ax.set_ylim(0,85)
ax.set_ylabel("Percent Decrease in Perspective Accuracy")
plt.savefig('attackpower.png')
  

# df1 = pd.DataFrame(data={'Number of Neighbors Searched Per Word': neighbors, 
# 						'Dip in the Accuracy of Perspective API': neighbor_acc})
# df2 = pd.DataFrame(data={'Edit Distance (Number of Words Changed)': edits, 
# 						'Dip in the Accuracy of Perspective API': edit_acc,})
# df3 = pd.DataFrame(data={'Number of Queries Made to Perspective API': queries,
# 						 'Dip in the Accuracy of Perspective API': query_acc})
						
# plt.figure(1, figsize=(8,6))

# sns.set(style='whitegrid')
# plt.title('Effectiveness of Adversarial Examples '
# 		  + 'as a Function of Candidates Searched')
# plt.ylim(0, 0.35)
# ax = sns.barplot(x=list(df1)[0], y=list(df1)[1], data=df1)
# plt.savefig('accuracy_neighbors2.png')

# plt.figure(2, figsize=(8,6))
# plt.title('Effectiveness of Adversarial Examples '
# 		  + 'as a Function of Edit Distance')
# plt.ylim(0, 0.5)
# sns.set(style='whitegrid')
# ax = sns.barplot(x=list(df2)[0], y=list(df2)[1], data=df2)
# plt.savefig('accuracy_edits.png')


# plt.figure(3, figsize=(8,6))
# plt.title('Effectiveness of Adversarial Examples '
# 		  + 'as a Function of Queries Made')
# plt.ylim(0, 0.35)
# sns.set(style='whitegrid')
# ax = sns.barplot(x=list(df3)[0], y=list(df3)[1], data=df3)
# plt.savefig('accuracy_queries.png')



# plt.show()