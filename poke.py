import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score

def generate_data(samples, dim, clusters, ratio_per_cluster, spread=3.0, lower_bound=0, upper_bound=200):

  """
  Parameters
  ----------
  samples: int default=None
      Number of data points generated.

  dim: int, default=None
       The dimension of features for each sample.

  clusters: int, defult=None
       Number of clusters to generate.

  spread: float or list [lower_bound, upper_bound], default=3.0
          If float, variance for each feature in each cluster.
          If list [lower_bound, upper_bound], the lower and upper bounds of variance on each feature in each cluster
  
  ratio_per_cluster : ndarray of shape (clusters) 
      ratio of samples in each cluster

  lower_bound, upper_bound: int, default lower_bound=0, upper_bound=200
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

  samples_per_cluster = np.int64(samples * ratio_per_cluster)
  print(samples_per_cluster)

  centers = np.random.uniform(lower_bound, upper_bound, size=(clusters,dim))
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
  # total_n_samples = np.sum(samples_per_cluster)
  total_n_samples = samples

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
  return X_train, X


TOTAL_POINTS = 10000
X, labels, centers = generate_data(samples=TOTAL_POINTS, dim=100, clusters=10, spread=[1,30], ratio_per_cluster=np.array([0.16,0.15,0.05,0.1,0.2,0.05,0.05,0.12,0.08,0.04]))
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
X_train_LDA, X_test_LDA = dimension_reduction_LDA(X_train,y_train,X_test, 4) #could change up to 10

#%% DPGMM comparison 
clf_dp = BGM(n_components=10, weight_concentration_prior_type='dirichlet_process')
clf_dp.fit(X_train)
y_pred_dp = clf_dp.predict(X_test)

### TODO: dp seems not to give a good pic of the dataset ... need help

### some params
# dp_mean = clf.means_,
# dp_cov = clf.covariances_

# plt.figure(1, figsize=(8,8))
# plt.clf()
# plt.scatter(X_train[:,2],X_train[:,3])
# plt.show()

#%% LDA + DPGMM
clf_comb = BGM(n_components=4, weight_concentration_prior_type='dirichlet_process')
clf_comb.fit(X_train_LDA)
y_pred_comb = clf_comb.predict(X_test_LDA)

# comb_mean = clf_comb.means_,
# comb_cov = clf_comb.covariances_

# plt.figure(2, figsize=(8,8))
# plt.clf()
# plt.scatter(X_test_LDA[:,2],X_test_LDA[:,3])
# plt.show()

#%% K-means
clf_kmeans = KMeans(n_clusters=10)
clf_kmeans.fit(X_train)
kmeans_mean = clf_kmeans.cluster_centers_
y_pred_kmeans = clf_kmeans.predict(X_test)

#%% use S-score
dp_rscore = rand_score(y_test, y_pred_dp)
comb_rscore = rand_score(y_test, y_pred_comb)
km_rscore = rand_score(y_test, y_pred_kmeans)

dp_sscore = silhouette_score(X_test, y_pred_dp)
comb_sscore = silhouette_score(X_test, y_pred_comb)
km_sscore = silhouette_score(X_test, y_pred_kmeans)

dp_arscore = adjusted_rand_score(y_test, y_pred_dp) # better than rand index as a metrics?
comb_arscore = adjusted_rand_score(y_test, y_pred_comb)
km_arscore = adjusted_rand_score(y_test, y_pred_kmeans)

print(dp_rscore)
print(comb_rscore)
print(km_rscore)

print(dp_sscore)
print(comb_sscore)
print(km_sscore)

print(dp_arscore)
print(comb_arscore)
print(km_arscore)
