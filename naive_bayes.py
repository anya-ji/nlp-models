'''
Implementation of Multinomial Naive Bayes Model with sklearn package.
'''
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#populate preprocessed documents list and labels
documents = [] # a list of strings which are concatenated preprocessed tokens in
# a sentence: ['this fake','also fake news', 'what real']
labels = [] # labels corresponding to each item in documents: [0,0,1] 

# dummy data
tokenized_fake_news_training = [['this', 'is', 'fake'], ['this', 'is', 'also','fake']]
tokenized_real_news_training = [['this', 'is', 'real'], ['this', 'is', 'definitely', 'not','fake']]
tokenized_fake_news_validation = [['it', 'is', 'fake', 'wow'], ['fake', 'news']]
tokenized_real_news_validation = [['this', 'real', 'article'], ['this', 'is', 'true']]
'''
Returns: a string that concatenates a list of preprocessed tokens.
['fake','a','.', 'data'] --> 'fake data'.
'''
def preprocessed_to_string(tokens):
  p = preprocess(tokens) # preprocess.py
  s= " "
  sl = s.join(p)
  return sl

for tokens in tokenized_fake_news_training:
  documents.append(preprocessed_to_string(tokens)) 
  labels += [0]
for tokens in tokenized_real_news_training:
  documents.append(preprocessed_to_string(tokens)) 
  labels += [1]

#populate preprocessed fake and real documents list
fake_documents = []
for tokens in tokenized_fake_news_validation:
  fake_documents.append(preprocessed_to_string(tokens))

real_documents = []
for tokens in tokenized_real_news_validation:
  real_documents.append(preprocessed_to_string(tokens))
  

#vectorize all documents
vectorizer = CountVectorizer()
documents = vectorizer.fit_transform(documents)
fake_documents = vectorizer.transform(fake_documents)
real_documents = vectorizer.transform(real_documents)


#multinomialNB
clf = MultinomialNB(alpha=0.01)
clf.fit(documents,labels)
MultinomialNB()

predict_fake1 = clf.predict(fake_documents)
predict_real1 = clf.predict(real_documents)


accuracy_fake = predict_fake1.tolist().count(0)  / len(predict_fake1) 
accuracy_real = predict_real1.tolist().count(1) / len(predict_real1)

print("accuracy fake:", accuracy_fake) 
print("accuracy real:", accuracy_real) 