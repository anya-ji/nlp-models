'''
Implementation of Maximum Entropy Markov Model using Viterbi algorithm.
'''
import nltk
from nltk.classify import MaxentClassifier
nltk.download('averaged_perceptron_tagger')
import itertools

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

'''
[featurify] returns a list feature dictionaries. [{feature}, {feature}, ...]
Requires: [sentence] is a list of tokens.
'''
def featurify(sentence, feature=0):
  if feature == 0: # 1 prevtag, POS
    token_pos = nltk.pos_tag(sentence)
    feature_set = [{'word': token_pos[0][0], 'prevtag': 2, 'pos':token_pos[0][1]}]
    for i in range(1, len(sentence)):
      feature_set.append({'word': token_pos[i][0], 'prevtag': labels[i-1], 'pos':token_pos[i][1]})

  elif feature == 1: # 1 prevtag, POS, sentiment
    # sentiment tagging
    s = ' '.join(sentence)
    score = SentimentIntensityAnalyzer().polarity_scores(s)
    max_score = max(score['pos'],score['neg'],score['neu'])
    if max_score==score['pos']:
      sent = 1
    elif max_score==score['neg']:
      sent = -1
    else:
      sent = 0
    # POS
    token_pos = nltk.pos_tag(sentence)
    feature_set = [{'word': token_pos[0][0], 'prevtag': 2, 'pos':token_pos[0][1], 'sent': sent}]
    for i in range(1, len(sentence)):
      feature_set.append({'word': token_pos[i][0], 'prevtag': labels[i-1], 'pos':token_pos[i][1], 'sent': sent})

  elif feature == 2: # 1 prevtag, sentiment
    # sentiment tagging
    s = ' '.join(sentence)
    score = SentimentIntensityAnalyzer().polarity_scores(s)
    max_score = max(score['pos'],score['neg'],score['neu'])
    if max_score==score['pos']:
      sent = 1
    elif max_score==score['neg']:
      sent = -1
    else:
      sent = 0
    feature_set = [{'word': sentence[0], 'prevtag': 2, 'sent': sent}]
    for i in range(1, len(sentence)):
      feature_set.append({'word': sentence[i], 'prevtag': labels[i-1], 'sent': sent})

  elif feature ==3: # POS
    token_pos = nltk.pos_tag(sentence)
    feature_set = [{'word': token_pos[0][0], 'pos':token_pos[0][1]}]
    for i in range(1, len(sentence)):
      feature_set.append({'word': token_pos[i][0], 'pos':token_pos[i][1]})
  
  elif feature ==4: # sentiment
    # sentiment tagging
    s = ' '.join(sentence)
    score = SentimentIntensityAnalyzer().polarity_scores(s)
    max_score = max(score['pos'],score['neg'],score['neu'])
    if max_score==score['pos']:
      sent = 1
    elif max_score==score['neg']:
      sent = -1
    else:
      sent = 0
    feature_set = [{'word': sentence[0], 'sent': sent}]
    for i in range(1, len(sentence)):
      feature_set.append({'word': sentence[i], 'sent': sent})
      
  else: # POS, sentiment
    # sentiment tagging
    s = ' '.join(sentence)
    score = SentimentIntensityAnalyzer().polarity_scores(s)
    max_score = max(score['pos'],score['neg'],score['neu'])
    if max_score==score['pos']:
      sent = 1
    elif max_score==score['neg']:
      sent = -1
    else:
      sent = 0
    # POS
    token_pos = nltk.pos_tag(sentence)
    feature_set = [{'word': token_pos[0][0], 'pos':token_pos[0][1], 'sent': sent}]
    for i in range(1, len(sentence)):
      feature_set.append({'word': token_pos[i][0], 'pos':token_pos[i][1], 'sent': sent})
  return feature_set


'''
[classifier] returns a MaxEnt classifier trained on [training].
Requires: [training] is a list of lists of tokens in sentences.
[labels] is a list of lists of labels {0,1} that correspond to tokens in [training].
[max] is max number of iterations.
'''
def classifier(training, labels, max, f=0):
  ll = [zip(featurify(s,f), l) for s, l in zip(training, labels)]
  return MaxentClassifier.train(list(itertools.chain.from_iterable(ll)), max_iter=max)

'''
[prob] returns the probability of [label] predicted by the [classifier], 
given the [feature] with preceding label [prev].
'''
def prob(classifier, feature, label, prev):
  feature['prevtag'] = prev 
  return classifier.prob_classify(feature).prob(label)

'''
[viterbi] finds most probable labels (sentence is a list of words).
'''
def viterbi_memm(sentence, classifier, f=0):
  tagset = [0, 1]
  probstable, backtracker = np.zeros((len(tagset), len(sentence))), np.zeros((len(tagset), len(sentence)))
  features = featurify(sentence,f)
  ### forward pass ###
  # initialization step
  for state in tagset:
    probstable[state, 0] = np.log2(prob(classifier, features[0], state, 2))
    backtracker[state, 0] = -1
  # recursion step
  for t in range(1, len(sentence)):
    for state in tagset:
      bestval, index = -float("inf"), -1
      for prev_state in tagset:
        prev = probstable[prev_state][t-1]
        current = prev + np.log2(prob(classifier, features[t], state, prev_state))
        if current > bestval:
          bestval, index = current, prev_state
      probstable[state, t], backtracker[state, t] = bestval, index 
  # termination step
  bestprob = max(probstable[:, len(sentence)-1])
  pathptr = int(np.argmax(probstable[:, len(sentence)-1]))

  ### backward pass ###
  labels = []
  for i in range(len(sentence)-1, 0, -1):
    labels.append(pathptr)
    pathptr = int(backtracker[pathptr, i])
  labels.append(pathptr)
  return labels[::-1], bestprob