# BigData
import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from nltk.tokenize import word_tokenize 
from scipy.spatial.distance import pdist, squareform

# download file
train_file = '/Users/johngolec/Documents/Fall 2018/Big Data Analytics/Semester Project/all/train.csv'
train_df = pd.read_csv(train_file, sep = ',')
train_df.head()

# split data into comments and toxic classifications
comments_df = train_df['comment_text']
toxic_df = train_df.drop(['id','comment_text'], axis=1)
comments_df.head()
toxic_df.head()

# count number of comments classified in each category and visualize the number of comments classified in each category with a bar chart
counts = []
categories = list(toxic_df.columns.values)
for i in categories:
    counts.append((i, toxic_df[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats

df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=False, figsize=(8, 5))
plt.title("Number of comments per category")
plt.ylabel(' Number of Occurrences', fontsize=12)
plt.xlabel('Category', fontsize=12)


# removing uppercase words from comment
#remove_uppercase = [word.lower() for word in word_tokens]

# clean comment text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\\n',' ',text)
    text = re.sub("\[\[User.*",'',text)
    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',text)
    text = re.sub("(http://.*?\s)|(http://.*)",'',text)
    text = re.sub("(1)|(2)|(3)|(4)|(5)|(6)|(7)|(8)|(9)|(0)|",'',text)
    
    text = text.strip(' ')
    return text

# map through comment_text column and clean every comment
train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))
train_df['comment_text'][0]



# tokenize a sentence and remove stopwords from word_tokens
sentence = train_df['comment_text'][0]
word_tokens = word_tokenize(sentence) 

filtered_sentence = [w for w in word_tokens if not w in stop_words] 
filtered_sentence = [] 

for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

print(word_tokens) 
print(filtered_sentence)



# compute hamming distances of matrix X
#distances = pdist(, 'hamming')

# convert distance vector to square matrix
#square = squareform(distances)

