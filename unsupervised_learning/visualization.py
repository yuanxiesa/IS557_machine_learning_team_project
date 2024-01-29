from collections import Counter
import nltk
from pathlib import Path
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt

#%%
nltk_stopwords = stopwords.words('english')

dirpath = Path('./Health-Tweets')

# process each file
for file_path in dirpath.glob("*.txt"):
    words = Counter()
    tweets = []
    # parse the file
    num_tweet = 0
    num_read_error = 0
    with open(file_path, encoding='latin1') as f:
        for line in f:
            num_tweet += 1
            try:
                tweet = line.split('|')[2].lower().split()
            except:
                num_read_error += 1
            else:
                words.update(tweet)
                tweets = tweets + tweet

    # Additional processing ----- comment from here for output without word filtering
        # delete keys in the stopwords
        for stopword in nltk_stopwords:
            words.pop(stopword, None)

        # get a list of keys
        keys = list(words.keys())

        # delete keys that star with "@" or "#"
        for key in keys:
            if key.startswith('#') or key.startswith("@") or key.startswith('http:') or key.startswith('https:'):
                words.pop(key)

        # delete keys that are common to Tweets but meaningless
        words.pop('rt', None)
        words.pop('&amp;', None)
        words.pop('--', None)
        words.pop('-', None)
        # Additional processing ----- comment until here

    top_10_word = []
    top_10_prob = []
    len_tweets = len(tweets)

    for word, freq in words.most_common(10):
        top_10_word.append(word)
        top_10_prob.append(freq/len_tweets)

    # visualization
    fig, ax = plt.subplots()

    ax.bar(top_10_word, top_10_prob)
    ax.set_ylabel('Probability of Occurrence')
    ax.set_title(file_path.name)
    ax.set_xticklabels(labels=top_10_word, rotation=30, ha='right')

    figure_name = file_path.name.split('.')[0] + '.png'
    plt.savefig(figure_name, format='png')
