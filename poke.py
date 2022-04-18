import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.cluster import KMeans

def generate_data(dim,clusters,samples_per_cluster,spread=3.0, lb=0,ub=200):

  """
  Parameters
  ----------
  dim: int, default=None
       The dimension of features for each sample.

  clusters: int, defult=None
       Number of clusters to generate.

  spread: float or list [lb, up], default=3.0
          If float, variance for each feature in each cluster.
          If list [lb, up], the lower and upper bounds of variance on each feature in each cluster
  
  samples_per_cluster : list, Number of samples in each cluster

  lb,up: int, default lb=0, up=200
        Lower and upper bound of the samples generated  

  Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.
    labels : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.
    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster.
  """

  centers = np.random.uniform(lb, ub, size=(clusters,dim))
  if isinstance(spread, (float,int)):
        spread  = [spread]*2 
  dim_spread=np.random.uniform(np.sqrt(spread[0]), np.sqrt(spread[1]), size=(clusters,dim))
  X = []
  labels = []
  for i, (n, std) in enumerate(zip(samples_per_cluster, dim_spread)):
        X.append(np.random.normal(loc=centers[i], scale=std, size=(n, dim)))
        labels += [i] * n


  X = np.concatenate(X)
  labels = np.array(labels)
  total_n_samples = np.sum(samples_per_cluster)
  indices = np.arange(total_n_samples)
  np.random.shuffle(indices)
  X = X[indices]
  labels = labels[indices]
  return X, labels, centers

def dimension_reduction_LDA(X_train,y_train,X,n_components=3):
  """
  Parameters
  ---------
  X_train: training data with known cluster labels
  y_train: labels of training data
  X: data with unkown cluster labels
  n_components: number of components to retain, need to be smaller than the number of classes

  Returns
  ---------
  X: reduced data with dimension n_components 
  """
  lda=LinearDiscriminantAnalysis(n_components=n_components)
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X = sc.transform(X)
  X_train = lda.fit_transform(X_train, y_train)
  X = lda.transform(X)
  return X

def dimension_reduction_PCA(X,n_components=10):
  """
  Parameters
  ---------
  X: data with unknown cluster labels
  n_components: number of components to retain

  Returns
  ---------
  X: reduced data with n_components
  """
  pca = PCA(n_components=n_components)
  X=pca.fit_transform(X)
  return X

X,labels,centers=generate_data(dim=100,clusters=10,spread=[1,30],samples_per_cluster=[30,20,10,50,35,20,15,60,25,40])
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.8, random_state=0)
X_reduced_LDA=dimension_reduction_LDA(X_train,y_train,X_test,4) #could change up to 10

#%% DPGMM comparison 
clf = BGM(n_components=10, weight_concentration_prior_type='dirichlet_process')
clf.fit(X_train)
y_pred = clf.predict(X_train)

### TODO: dp seems not to give a good pic of the dataset ... need help

### some params
dp_mean = clf.means_,
# dp_cov = clf.covariances_

plt.figure(1, figsize=(8,8))
plt.clf()
plt.scatter(X_train[:,2],X_train[:,3])
plt.show()

#%% LDA + DPGMM
clf_comb = BGM(n_components=10, weight_concentration_prior_type='dirichlet_process')
clf_comb.fit(X_reduced_LDA)
y_pred = clf_comb.predict(X_reduced_LDA)

comb_mean = clf_comb.means_,
# comb_cov = clf_comb.covariances_

plt.figure(2, figsize=(8,8))
plt.clf()
plt.scatter(X_reduced_LDA[:,2],X_reduced_LDA[:,3])
plt.show()

#%% K-means
clf_kmeans = KMeans(n_clusters=10)
clf_kmeans.fit(X_train)
kmeans_mean = clf_kmeans.cluster_centers_