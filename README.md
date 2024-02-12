# IS557 MLTP
This repository holds my code for IS557 Applied Machine Learning: Team Project

## Unsupervised learning
This folder holds my code for the first preliminary individual project. I use unsupervize learning to discover clusters among tweets from 16 Twitter (now X :)) account (Karami,Amir. (2018). Health News in Twitter. UCI Machine Learning Repository. [https://doi.org/10.24432/C5BW2Q](https://doi.org/10.24432/C5BW2Q). 

- visualization.py: process the files and visualize the probability of occurrence for the 10 most common words from each Twitter account
- clustering.py: perform k-means clustering (k=16) on the vectorized tweets and build a decision tree model to explain the clustering results
- improve_clustering_vary_K.py: vary the number of clusters (K) to observe the change in the quality of clustering
- graph_based_clustering.py: This file is currently incomplete. It only implemented a semantics-informed tweet similarity measure that uses WordNet. Future work includes (1) computing all pairs of tweets in the datasets, (2) creating a network based on the pair-wise similarity, (3) using community detection algorithms to identify clusters, and (4) using tweets with high centrality to represent the meaning of the cluster, and (5) verifying the methodology by examing other tweets in the network.  

## Classification with feature selection
This folder holds my code for the first group project. I will deal with a dataset to classify 9 different types of urban land cover (e.g., concrete, grass, soil) (Johnson,Brian. (2014). Urban Land Cover. UCI Machine Learning Repository. [https://doi.org/10.24432/C53S48](https://doi.org/10.24432/C53S48)).

This dataset is a bit unusual because "there are a low number of training samples for each class (14-30) and a high number of classification variables (148)," which makes it a good testing bed for feature selection methods, as good and bad feature selection methods may be better differentiated.

