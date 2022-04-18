from sqlite3 import DatabaseError
import numpy as np
import matplotlib.pyplot as plt
from regex import D
from sklearn.model_selection import train_test_split

from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score

from DataGenerator import DataGenerator
import utils

### Constants
TOTAL_POINTS = 10000
RATIOS = np.array([0.16,0.15,0.05,0.1,0.2,0.05,0.05,0.12,0.08,0.04]) # might need random assignments for ratio

myGenerator = DataGenerator(samples=TOTAL_POINTS, n_features=100, n_clusters=10, spread=[1,30], ratio_per_cluster=RATIOS)
X, labels, centers = myGenerator.generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
X_train_LDA, X_test_LDA = utils.dimension_reduction_LDA(X_train,y_train,X_test, 4) #could change up to 10

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
