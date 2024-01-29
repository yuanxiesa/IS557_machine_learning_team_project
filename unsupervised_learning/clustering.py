from pathlib import Path
import pandas as pd


def process_data(path_str):
    '''
    :param path_str: the str represent the path where the data is: folder name
    :return: tweets: a list of the tweets
    :return: dataset_label: the name of the dataset that a tweet belongs to
    '''
    # create a word frequency table
    dirpath = Path(path_str)
    tweets = []
    num_error_read = 0
    dataset_label = []
    # process each file
    for file_path in dirpath.glob("*.txt"):
        # process the file
        with open(file_path, encoding='latin1') as f:
            for line in f:
                try:
                    tweet = line.split('|')[2].lower()
                except:
                    num_error_read += 1
                else:
                    tweets.append(tweet)
                    dataset_label.append(file_path.name.split(".")[0])

        print(f'{file_path.name} :{len(tweets)} tweets read, {num_error_read} not read')

    return tweets, dataset_label


def create_vocab(tweets, vector_size):
    """
    :param tweets: a list of tweets, output from process_data()
    :param vector_size: the size of the vocabulary, also the size of the vector when we vectorize the tweet
    :return: vocab: the vocabulary used by sklearn.feature_extraction.text.CountVectorizer()
    """
    from collections import Counter
    from nltk.corpus import stopwords

    words = Counter()
    for tweet in tweets:
        words.update(tweet.split())

    # filter out useless words
    # delete keys in the stopwords and numbers
    nltk_stopwords = stopwords.words('english')
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

    print(words.most_common(vector_size))

    # make vocabulary
    # keys are terms and values are indices
    terms = list(dict(words.most_common(vector_size)).keys())
    vocab = {x: y for x, y in zip(terms, range(vector_size))}

    return vocab


def vectorize_tweets(tweets, vocab):
    '''
    :param tweets: a list of tweets, the output of process_data()
    :param vocab: the vocabulary used by sklearn.feature_extraction.text.CountVectorizer(), output of create_vocab()
    :return: X: the numpy array of vectorized tweets
    '''
    # vectorize
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(tweets)
    X = X.toarray()

    return X

def main():
    tweets, label = process_data('./Health-Tweets')
    vocab = create_vocab(tweets, vector_size=200)
    X = vectorize_tweets(tweets, vocab)
    terms = list(vocab.keys())

    # kmeans
    from sklearn.cluster import KMeans
    est = KMeans(n_clusters=16, random_state=42)
    est.fit(X)
    kmeans_labels = est.labels_

    # create a dictionary for the number of members in each cluster
    from collections import Counter
    cluster_size_dict = dict(Counter(est.labels_.tolist()))

    # compute stats
    kmeans_result_df = pd.DataFrame(data={'cluster': est.labels_, 'label': label, 'num': 1})
    new_kmeans_result_df = kmeans_result_df.groupby(['cluster', 'label']).count()

    # find the most represented Twitter account in each cluster
    for idx in range(16):
        account_name = new_kmeans_result_df.loc[idx].apply(lambda x: x / cluster_size_dict[idx]).sort_values(by='num',
                                                                                                             ascending=False).head(
            1).index.values[0]

        rep_percent = int(
            new_kmeans_result_df.loc[idx].apply(lambda x: x / cluster_size_dict[idx]).sort_values(by='num',
                                                                                                  ascending=False).head(
                1).values[0][0] * 100)

        print(f'Cluster {idx} ({cluster_size_dict[idx]} tweets): {account_name} ({rep_percent}%)')

    # create a tree model to explain the clustering
    # recreate the data table with column headers (i.e. terms)
    new_input_df = pd.DataFrame(columns=terms, data=X)
    new_input_df['label'] = est.labels_
    new_input_df.head()

    # training a decision tree model
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    X = new_input_df.drop(columns=['label'])
    y = new_input_df.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt = dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 3))

    # text representation
    from sklearn.tree import export_text
    text_representation = export_text(dt, feature_names=terms)
    print(text_representation)

    # visualization
    fig = plt.figure(figsize=(25, 25))
    _ = plot_tree(dt, feature_names=terms, fontsize=12, filled=True, max_depth=3)
    plt.show()


if __name__ == '__main__':
    main()
