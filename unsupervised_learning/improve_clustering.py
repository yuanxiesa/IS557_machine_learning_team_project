from clustering import process_data
from clustering import create_vocab
from clustering import vectorize_tweets


def main():
    # basic processing
    tweets, label = process_data('./Health-Tweets')
    vocab = create_vocab(tweets, vector_size=200)
    X = vectorize_tweets(tweets, vocab)
    terms = list(vocab.keys())

    num_cluster_l = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score
    print('Calinski-Harabasz Index')
    for num_cluster in num_cluster_l:
        est = KMeans(n_clusters=num_cluster, random_state=42)
        est.fit(X)
        print(f'{num_cluster} clusters: {round(calinski_harabasz_score(X, est.labels_),2)}')

if __name__ == '__main__':
    main()



