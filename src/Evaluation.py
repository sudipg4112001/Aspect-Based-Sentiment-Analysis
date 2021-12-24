import pandas as pd
import numpy as np
import re
import nltk
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
df=pd.read_csv('../input/sentimententerpret/train.csv')
test=pd.read_csv("../input/sentimententerpret/test.csv")
df.head()

df.shape

df.dropna()

def clean_html(text):
    clean=re.compile('<.*?>')
    cleantext=re.sub(clean,'',text)
    return cleantext

def clean_text1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

def clean_text2(text):
    text=re.sub('[''"",,,]','',text)
    text=re.sub('\n','',text)
    return text
    
cleaned_html=lambda x:clean_html(x)
cleaned1=lambda x:clean_text1(x)
cleaned2=lambda x:clean_text2(x)

df['text']=pd.DataFrame(df.text.apply(cleaned_html))
df['text']=pd.DataFrame(df.text.apply(cleaned1))
df['text']=pd.DataFrame(df.text.apply(cleaned2))

test['text']=pd.DataFrame(test.text.apply(cleaned_html))
test['text']=pd.DataFrame(test.text.apply(cleaned1))
test['text']=pd.DataFrame(test.text.apply(cleaned2))

df=df.drop(columns='aspect')
aspect=test['aspect']
test=test.drop(columns='aspect')
df.head()

def remove_stopword(text):
    stopword=nltk.corpus.stopwords.words('english')
    stopword.remove('not')
    a=[w for w in nltk.word_tokenize(text) if w not in stopword]
    return ' '.join(a)
df['text'] = df['text'].apply(remove_stopword)
test['text'] = test['text'].apply(remove_stopword)

df

vectr = TfidfVectorizer(ngram_range=(1,2),min_df=1)
vectr.fit(df['text'])
vect_X = vectr.transform(df['text'])

model = LogisticRegression()
clf=model.fit(vect_X,df['label'])
clf.score(vect_X,df['label'])*100

clf.predict(vectr.transform(['i am using machine learning for a project on sentiment analysis']))

filename = 'finalized_model.sav'
joblib.dump(model, filename)

pickle.dump(model, open("model.pkl", "wb"))

predict = model.predict(vectr.transform(test['text']))
output = pd.DataFrame({'text':test['text'],'aspect':aspect,'label': predict})
output.to_csv('my_submission.csv', index=False)
print("Submission saved")

clf.score(vectr.transform(test['text']),output['label'])*100

output