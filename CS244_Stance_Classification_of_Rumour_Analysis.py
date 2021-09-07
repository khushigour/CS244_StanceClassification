#!/usr/bin/env python
# coding: utf-8

# In[140]:


import pandas as pd
import numpy as np
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

#loading dataset
filepath=r"C:\Users\mahes\Desktop\New folder (2)\Rumor_datascience\Rumor_datascience\SemEval\train\Table2SemEvalWithSource_trainText.csv"
df=pd.read_csv(filepath)
df


# #### Preprocessing and feature extraxtion from data

# In[141]:


print(type(df['branches'][0]))
print(type(df['fold_stance_labels_text'][0]))


# In[142]:


#converting string representation of list into list
import ast
for i in range(len(df.branches)):
    df.branches[i]=ast.literal_eval(df.branches[i])

print(type(df['branches'][0]))

for i in range(len(df.fold_stance_labels)):
    df.fold_stance_labels[i]=ast.literal_eval(df.fold_stance_labels[i])
    
print(type(df['branches'][0]))

for i in range(len(df.fold_stance_labels_text)):
    df.fold_stance_labels_text[i]=ast.literal_eval(df.fold_stance_labels_text[i])
    


# In[143]:


#adding relevant columns to required dataframe
df_req = pd.DataFrame()
ar=[]
for s in df['branches']:
    for i in s:
        ar.append(i)
df_req['text'] = ar

ar2=[]
for s in df['fold_stance_labels']:
    for i in s:
        ar2.append(i)
df_req['label'] = ar2

ar3=[]
for s in df['fold_stance_labels_text']:
    for i in s:
        ar3.append(i)
df_req['label_text'] = ar3

df_req


# In[144]:


df_req.info()


# In[145]:


#  nltk.download()


# #### Processing Text

# In[146]:


# from nltk.corpus import stopwords

def remove_punctuation(text):
    '''
    Removing punctuation characters from text
    '''
    without_punct="".join([i for i in text if i not in string.punctuation])
    return without_punct


def tokenize(string):
    '''
    Tokenizes the string to a list of words
    '''
    word_tokens = string.split()
    return word_tokens


stop_words = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    '''
    Removing stop words to focus on meaningful words
    '''
    output= [i for i in text if i not in stop_words]
    return output


porter_stemmer = PorterStemmer()
def stemming(text):
    '''
    Stemming of words in text
    '''
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text


wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    '''
    Lemmatizing of words in text
    '''
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def clean_text():
    df_req['text']= df_req['text'].str.lower()
    df_req['text']= df_req['text'].str.replace('"' , '')
    df_req['text']= df_req['text'].str.replace('http\S+','',case=False)
    df_req['text']= df_req['text'].apply(lambda x: remove_punctuation(x))
    df_req['text']= df_req['text'].apply(lambda x: tokenize(x))
    df_req['text']= df_req['text'].apply(lambda x:remove_stopwords(x))
    df_req['text']= df_req['text'].apply(lambda x: stemming(x))
    df_req['text']= df_req['text'].apply(lambda x: lemmatizer(x))
    df_req['text_strings'] = df_req['text'].apply(lambda x: ' '.join([str(word) for word in x]))
    
    
    

# nltk.download('wordnet')
# nltk.download('stopwords')
clean_text()
df_req.head()


# In[147]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_req['text_strings'])
X


# In[148]:


x_train = X.toarray()
x_train = np.array(x_train)
x_train.shape


# In[149]:


x_train


# #### Incase labels are not given

# In[150]:


# from sklearn.preprocessing import LabelEncoder
# label_enc = LabelEncoder()
# train_y = label_enc.fit_transform(train_y)
# test_y = label_enc.transform(test_y)
# train_y


# #### Vectorize text

# In[151]:


count=0
for i in df_req.text:
    for j in i:
        if isinstance(j, list):
            count+=1

print(count)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1')
features = tfidf_vect.fit_transform(df_req.text_strings).toarray()
features


# In[156]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(df_req.text_strings, df_req.label, test_size = 0.3, random_state =1)
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)
print(train_X)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_X)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# #### Function for calculating accuracy of model in test set

# In[153]:


from sklearn.metrics import accuracy_score

def calculte_pred_y_and_accuracy(model):
    pred_y=model.predict(count_vect.transform(test_X))
    accuracy=accuracy_score(test_y,pred_y)*100
    print('Accuracy(in %)=',accuracy)
    


# #### Naive Bayes

# In[128]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train_y)
calculte_pred_y_and_accuracy(clf)


# #### RandomForest Classifier

# In[130]:


from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 700)  #500
clf2.fit(X_train_tfidf,train_y)
calculte_pred_y_and_accuracy(clf2)


# #### KNeighbors Classifier

# In[131]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train_tfidf, train_y) 
calculte_pred_y_and_accuracy(knn)


# #### SVM For various Kernels

# In[139]:


from sklearn import svm
svm_model = svm.SVC(kernel='linear') # Linear Kernel
svm_model2 = svm.SVC(kernel='rbf') # Linear Kernel
svm_model3 = svm.SVC(kernel='poly') # Linear Kernel
svm_model.fit(X_train_tfidf, train_y)
svm_model2.fit(X_train_tfidf, train_y)
svm_model3.fit(X_train_tfidf, train_y)
calculte_pred_y_and_accuracy(svm_model)
calculte_pred_y_and_accuracy(svm_model2)
calculte_pred_y_and_accuracy(svm_model3)

