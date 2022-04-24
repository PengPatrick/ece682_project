from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score
from sklearn.manifold import TSNE
###TESTS on Large Scale Data with no training labels  100000 by 100 dimension
def plot_results(X, Y_, means, covariances, title):
  _, ax = plt.subplots()
    
  for i, (mean, covar) in enumerate(zip(means, covariances)):
    v, w = linalg.eigh(covar)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])

    if not np.any(Y_ == i):
      continue
        
    ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, label = i)

  # Plot an ellipse to show the Gaussian component
  angle = np.arctan(u[1] / u[0])
  angle = 180.0 * angle / np.pi  # convert to degrees
  ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, label = i)
  # ell.set_clip_box(plt.bbox)
  ell.set_alpha(0.5)
  ax.add_artist(ell)

  plt.xticks(())
  plt.yticks(())
  plt.legend()
  plt.title(title)
  plt.rcParams["figure.figsize"] = (30, 18)
  plt.show()

  #%% try plotting
 

X_train_tSNE = dimension_reduction_TSNE(X_train, 2) #could change up to 10
X_test_tSNE  = dimension_reduction_TSNE(X_test, 2) #could change up to 10

clf_tSNE = BGM(n_components = 20, covariance_type="full", weight_concentration_prior_type='dirichlet_process')
clf_tSNE.fit(X_train_tSNE)
y_pred_tSNE = clf_tSNE.predict(X_test_tSNE)

tSNE_rscore = rand_score(y_test, y_pred_tSNE)
tSNE_sscore = silhouette_score(X_test_tSNE, y_pred_tSNE)
tSNE_arscore = adjusted_rand_score(y_test, y_pred_tSNE)

print('###### tSNE ######')
print(tSNE_rscore)
print(tSNE_sscore)
print(tSNE_arscore)

class_tSNE = set(y_pred_tSNE)
print(class_tSNE)
print(len(class_tSNE))

# if we really need plots ... 
plot_results(
  X_test_tSNE,
  y_pred_tSNE,
  clf_tSNE.means_,
  clf_tSNE.covariances_,
  title="DPBGM with tSNE",
)


X, labels, centers = generate_data_2(samples=100000, dim=100, clusters=9, spread=20,
                                   ratio_per_cluster=np.array([0.1,0.15,0.05,0.25,0.1,0.05,0.2,0.05,0.05]),lower_bound=0, upper_bound=100)

Reduced_X_PCA=dimension_reduction_PCA(X,n_components=10)
Reduced_X_tSNE=dimension_reduction_PCA(X,n_components=10)
X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(Reduced_X_PCA, labels, test_size=0.99, random_state=0)
X_train_tSNE, X_test_tSNE, y_train_tSNE, y_test_tSNE = train_test_split(Reduced_X_PCA, labels, test_size=0.01, random_state=0)
a,b,v,W=prior_estimation(20,1000,10)
clf_dp = BGM(n_components=10, weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=np.random.gamma(a, scale=(1/b)),degrees_of_freedom_prior=v)
clf_dp.fit(X_train_PCA)
y_pred_dp = clf_dp.predict(X_train_PCA)


a=np.unique(y_pred_dp)
centers,res_k_means=fit_K_means(Reduced_X_PCA,len(a))
k_means_res=np.unique(res_k_means)
count1=[]
for i in k_means_res:
  count_i = np.count_nonzero(res == i)
  count1.append(count_i)
print("number of clusters", len(a))
print("number of elements in every cluster", count1)

a,b,v,W=prior_estimation(20,1000,10)
clf_dp = BGM(n_components=10, weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=np.random.gamma(a, scale=(1/b)),degrees_of_freedom_prior=v)
clf_dp.fit(X_train_tSNE)
y_pred_dp_tSNE = clf_dp.predict(X_train_tSNE)


a=np.unique(y_pred_dp_tSNE)
centers,res=fit_K_means(Reduced_X_tSNE,len(a))
k_means_res=np.unique(res)
count2=[]
for i in k_means_res:
  count_i = np.count_nonzero(res == i)
  count2.append(count_i)
print("number of clusters", a)
print("number of elements in every cluster", count2)


TOTAL_POINTS = 10000
X, labels, centers = generate_data_2(samples=TOTAL_POINTS, dim=100, clusters=10, spread=[1,30],
                                   ratio_per_cluster=np.array([0.16,0.15,0.05,0.1,0.2,0.05,0.05,0.12,0.08,0.04]))
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
X_train_LDA, X_test_LDA = dimension_reduction_LDA(X_train,y_train,X_test, 4) #could change up to 10

#%% DPGMM comparison 
clf_dp = BGM(n_components=10, weight_concentration_prior_type='dirichlet_process')
clf_dp.fit(X_train)
y_pred_dp = clf_dp.predict(X_test)

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

# ----Need some way to determing k.
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
