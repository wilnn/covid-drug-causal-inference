import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import ast  # For handling the list format if it's a string
import sys
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('./data/N3C_data_10000_sample.csv')

# Convert the 'conditions' column to actual lists (if they are stored as strings)
rows = df['conditions'].apply(ast.literal_eval).to_list()


rows = [[str(i) for i in row] for row in rows] # convert each id integer to string type

# Join tokens back into strings
col = [" ".join(row) for row in rows]

vectorizer = CountVectorizer()


X = vectorizer.fit_transform(col)
vocab = vectorizer.vocabulary_

X = X.toarray().tolist()

#print(len(X))
X = [str(x) for x in X]
#print(type(X))
#print(type(X[0]))
df['conditions'] = X
df.to_csv('./data/BOW_word2vec_condition.csv', index=False)


# Load the data
df = pd.read_csv('./data/N3C_data_10000_sample.csv')

# Convert the 'conditions' column to actual lists (if they are stored as strings)
rows = df['conditions'].apply(ast.literal_eval).to_list()


rows = [[str(i) for i in row] for row in rows] # convert each id integer to string type

model = Word2Vec(
    rows,         # list of lists
    vector_size=5,   # dimension of the embeddings
    window=5,          # context window size
    min_count=0,       # ignore codes with low frequency
    workers=2,         # parallelization
    sg=1               # 1 for Skip-Gram, 0 for CBOW
)

# Word2Vec conversion function
def word2vec_covert():
    temp = []
    
    for row in rows:
        temp2 = []
        for id in row:
            temp2.append([float(x) for x in model.wv[id]])
        temp.append(str(temp2))
    return temp
out = word2vec_covert()
#out = str(out)
#print(type(out))
#print(len(out))

#print(type(out[0]))
#print(out[0])
df['conditions'] = out
df.to_csv('./data/word2vec_condition.csv', index=False)



# Load the data
df = pd.read_csv('./data/N3C_data_10000_sample.csv')

# Convert the 'conditions' column to actual lists (if they are stored as strings)
rows = df['conditions'].apply(ast.literal_eval).to_list()


rows = [[str(i) for i in row] for row in rows] # convert each id integer to string type


# Word2Vec conversion function
def pooled_word2vec_covert():
    temp = []
    
    for row in rows:
        temp2 = []
        for id in row:
            temp2.append(model.wv[id])
        r = np.mean(temp2, axis=0)
        r = [float(x) for x in r]
        temp.append(str(r))
    return temp

out = pooled_word2vec_covert()

#print(type(out))
#print(len(out))

#print(type(out[0]))
#print(out[0])

df['conditions'] = out
df.to_csv('./data/pooled_word2vec_condition.csv', index=False)

