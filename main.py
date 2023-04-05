import nltk
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from datacleaning import data_cleaning

stop = set(stopwords.words("english"))
nltk.download('stopwords')


def main():
    # data_cleaning() // commented out because we already cleaned data
    data = pd.read_csv('200kdatacleaned.csv')
    vectors, words = string_to_vec(data) # string to vector implementation using Word2Vec
    dict = build_word_dict(vectors, words) # create word dictionary
    X = build_sentence_embeddings(data, dict) # create sentence embeddings + max pooling
    kmeanslabels = run_kmeans_elbow(X) # run kmeans
    run_analysis(X, kmeanslabels) # PCA + plotting


def build_corpus(data):
    corpus = []
    for i in range(len(data)):
        word_list = data.iloc[i][0].split(" ")
        corpus.append(word_list)
    return corpus


def run_analysis(X, kmeanslabels):
    print("using PCA to reduce to 2D")
    # plotting
    # Create a PCA model to reduce our data to 2 dimensions for visualisation
    pca = PCA(n_components=2)
    pca.fit(X)
    # Transform the scaled data to the new PCA space
    X_reduced = pca.transform(X)
    print(X_reduced)
    X_reduced_pd = pd.DataFrame(X_reduced)
    X_reduced_pd['label'] = kmeanslabels
    X_reduced_pd.to_csv('X_reduced.csv', index=False)
    # Analysis of Results
    """
        # done in Google Colab, code here:
        X_reduced["label"] = X_reduced["label"].astype(str)
    
        fig = px.scatter(X_reduced,
                         x='0',
                         y='1',
                         color=X_reduced['label'],
                         title='Visualizing K-Means Clustering with PCA',
                         )
        fig.update_xaxes(title_font=dict(size=22), title_text="X-value")
        fig.update_yaxes(title_font=dict(size=22), title_text="Y-value")
        fig.update_layout(
            title={
                'y': 0.935,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        fig.show()
        
        sentences = pd.read_csv('200kdatacleaned.csv')
        final = X_reduced
        final['Data'] = sentences['Data'].astype(str)
        final.to_csv('final.csv', index=False)
        """


def run_kmeans_elbow(X):
    print("running k-means elbow method...")
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    for k in K:
        print("k = ", k)
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        label = kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=0)) / len(X))
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / len(X)
        mapping2[k] = kmeanModel.inertia_

        if k == 5:  # we chose k = 5 as the 'elbow'
            print(label.labels_)
            kmeanslabels = label.labels_

    print("visualizing k-means distortion")
    for key, val in mapping1.items():
        print(f'{key} : {val}')
    return kmeanslabels


def build_sentence_embeddings(data, dict):
    print("creating sentence embeddings and conducting max pooling")
    sentence_embeddings = []
    num_words = 0
    for i in range(len(data)):
        sentence = data.iloc[i][0]
        max_pool = [-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999,
                    -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999,
                    -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999,
                    -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999,
                    -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999,
                    -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999,
                    -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999,
                    -9999, -9999]
        array_of_words = sentence.split(' ')
        for j in range(len(array_of_words)):
            num_words += len(array_of_words)
            word = array_of_words[j]
            # lookup word in PCA and get vector
            if word in dict:
                word_vector = dict.get(word)
            # loop through each index in the 100-vector
            for k in range(100):
                if word_vector[k] > max_pool[k]:
                    max_pool[k] = word_vector[k]
        sentence_embeddings.append(max_pool)
    X = sentence_embeddings
    return X


def build_word_dict(vectors, words):
    print("building word dictionary")
    vectors_np = np.asarray(vectors, np.float32)
    dict = {}
    for i in range(len(vectors_np)):
        key = words[i]
        if key not in dict and key not in '':
            dict[key] = vectors_np[i]
    return dict


def string_to_vec(data):
    print("running word2vec...")
    corpus = build_corpus(data)
    word2vec = Word2Vec(corpus, vector_size=100, min_count=1)
    model = word2vec.wv
    vectors = model.vectors
    words = []
    for i in range(len(model)):
        words.append(model.index_to_key[i])
    return vectors, words


if __name__ == '__main__':
    main()
