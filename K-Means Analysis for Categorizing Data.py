import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

def loaddata():
    df = pd.read_csv('toy_dataset.csv')
    la_enc = LabelEncoder()
    df['City'] = la_enc.fit_transform(df['City'])
    df['Gender'] = la_enc.fit_transform(df['Gender'])
    df['Illness'] = la_enc.fit_transform(df['Illness'])
    cols_to_norm = ['Age','Income']
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df

def split_data(df):
    train_df = pd.DataFrame()
    train_df = df.iloc[:119999, :]
    train_df.head()

    test_df = pd.DataFrame()
    test_df = df.iloc[119999:, :]
    test_df.head()

    return train_df, test_df

def kmeans(running_df):

    km = KMeans(n_clusters=2, random_state=42)

    # Fit the KMeans model
    km.fit_predict(running_df)

    # Calculate Silhoutte Score
    score = silhouette_score(running_df, km.labels_, metric='euclidean')

    # Print the score
    print('\nAverage Silhouette Score: %.3f' % score)

    return km

def plot_score(running_df):

    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    for i in [2, 3, 4, 5]:

      km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
      q, mod = divmod(i, 2)

      visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
      ax[q-1][mod].set_title('Average Silhouette Score for k = {} Clusters'.format(i))
      visualizer.fit(running_df)

def main():
    df = loaddata()
    train_df, test_df = split_data(df)
    print("\n------------------for Training Data----------------------------- ")
    kmeans(train_df)
    plot_score(train_df)
    print("\n\n\n")
    print("\n------------------for Testing Data------------------------------ ")
    kmeans(test_df)
    plot_score(test_df)

if __name__ == '__main__':
    main()
