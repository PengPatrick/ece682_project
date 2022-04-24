from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def experiment_dimension_reduction_large_n(dim=100,n_clusters=5,spread=20,lower_bound=0,upper_bound=100,dp_n_components=50,\
    pca_dim = 10,lda_dim = 2):
    #data gen
    TOTAL_POINTS = 100000
    ratio_per_cluster = np.array([1/n_clusters]*n_clusters)
    X, labels, centers = generate_data_2(samples=TOTAL_POINTS, dim=dim, clusters=n_clusters, spread=spread,
                                   ratio_per_cluster=np.array([0.1,0.15,0.05,0.25,0.1,0.05,0.2,0.05,0.05]),lower_bound=lower_bound, upper_bound=upper_bound)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.99, random_state=0)
    # #fit
    clusters = set(labels)
    tSNE = TSNE(n_components=2, init='pca')
    X_test_tSNE=tSNE.fit_transform(X)
    
    #plot true test clustering
    fig,((ax_true,ax_pred),(ax_pca,ax_lda)) = plt.subplots(2,2,figsize = (20,20))
    fig.suptitle(f'dimension:{dim},n_clusters:{n_clusters},spread:{spread},lb:{lower_bound},ub:{upper_bound}')
    for i in clusters:
        ax_true.scatter(X_test_tSNE[labels == i, 0], X_test_tSNE[labels == i, 1],  label = i,s=1)
    ax_true.set_title(f'Ground truth tSNE clusters \n actual num clusters:{len(clusters)}')

    
    #dimension reduction
    
    #pca
    pca = PCA(n_components=pca_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    #fit dpgmm after pca
    clf_dp_pca = BGM(n_components=dp_n_components, weight_concentration_prior_type='dirichlet_process')
    clf_dp_pca.fit(X_train_pca)
    y_pred_dp_pca = clf_dp_pca.predict(X_train_pca)
    
    #k-means for large dataset
    a=np.unique(y_pred_dp_pca)
    count1=[]
    for i in a:
      count_i = np.count_nonzero(res == i)
      count1.append(count_i)
    count1=np.asarray(count1)
    n=len(count1)
    for i in range(len(count1)):
      if count1[i]<np.sum(count1)*0.05:
        n=n-1
    
    centers,res_k_means=fit_K_means(X,n)
    

    #scores
    dp_arscore_pca = adjusted_rand_score(labels, res_k_means)
    
    pred_clusters_pca = set(res_k_means)
    for i in pred_clusters_pca:
        ax_pca.scatter(X_test_tSNE[res_k_means == i, 0], X_test_tSNE[res_k_means == i, 1],  label = i,s=1)
    ax_pca.set_title(f'DPGMM predicted tSNE clusters,fitted on PCA-processed data, pca dimension = {pca_dim}\
    \n predicted num clusters:{len(pred_clusters_pca)}\
    \n adjusted random score: {dp_arscore_pca}')


def experiment_dimension_reduction(dim=4,n_clusters=5,spread=20,lower_bound=0,upper_bound=200,dp_n_components=50,\
    pca_dim = 2,lda_dim = 2):
    #data gen
    TOTAL_POINTS = 10000
    ratio_per_cluster = np.array([1/n_clusters]*n_clusters)
    X, labels, centers = generate_data_1(samples=TOTAL_POINTS, dim=dim, clusters=n_clusters, spread=spread,
                                   ratio_per_cluster=ratio_per_cluster,lower_bound=lower_bound, upper_bound=upper_bound)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
    #fit
    clusters = set(labels)
    tSNE = TSNE(n_components=2, init='pca')
    X_test_tSNE=tSNE.fit_transform(X_test)
    
    #plot true test clustering
    fig,((ax_true,ax_pred),(ax_pca,ax_lda)) = plt.subplots(2,2,figsize = (20,20))
    fig.suptitle(f'dimension:{dim},n_clusters:{n_clusters},spread:{spread},lb:{lower_bound},ub:{upper_bound}')
    for i in clusters:
        ax_true.scatter(X_test_tSNE[y_test == i, 0], X_test_tSNE[y_test == i, 1],  label = i,s=1)
    ax_true.set_title(f'Ground truth tSNE clusters \n actual num clusters:{len(clusters)}')
    
    #fit dpgmm
    clf_dp = BGM(n_components=dp_n_components, weight_concentration_prior_type='dirichlet_process')
    clf_dp.fit(X_train)
    y_pred_dp = clf_dp.predict(X_test)
    
    #scores
    dp_arscore = adjusted_rand_score(y_test, y_pred_dp) # better than rand index as a metrics?

    
    pred_clusters = set(y_pred_dp)
    for i in pred_clusters:
        ax_pred.scatter(X_test_tSNE[y_pred_dp == i, 0], X_test_tSNE[y_pred_dp == i, 1],  label = i,s=1)
    ax_pred.set_title(f'DPGMM predicted tSNE clusters, n_components = {dp_n_components}\
    \n predicted num clusters:{len(pred_clusters)}\
    \n adjusted random score: {dp_arscore}')
    
    #dimension reduction
    
    #pca
    pca = PCA(n_components=pca_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    #fit dpgmm after pca
    clf_dp_pca = BGM(n_components=dp_n_components, weight_concentration_prior_type='dirichlet_process')
    clf_dp_pca.fit(X_train_pca)
    y_pred_dp_pca = clf_dp_pca.predict(X_test_pca)
    
    #scores
    dp_arscore_pca = adjusted_rand_score(y_test, y_pred_dp_pca)
    
    pred_clusters_pca = set(y_pred_dp_pca)
    for i in pred_clusters_pca:
        ax_pca.scatter(X_test_tSNE[y_pred_dp_pca == i, 0], X_test_tSNE[y_pred_dp_pca == i, 1],  label = i,s=1)
    ax_pca.set_title(f'DPGMM predicted tSNE clusters,fitted on PCA-processed data, pca dimension = {pca_dim}\
    \n predicted num clusters:{len(pred_clusters_pca)}\
    \n adjusted random score: {dp_arscore_pca}')
    
    
    #lda
    lda = LinearDiscriminantAnalysis(n_components=lda_dim)
    X_train_lda = lda.fit_transform(X_train,y_train)
    X_test_lda = lda.transform(X_test)
    
    #fit dpgmm after lda
    clf_dp_lda = BGM(n_components=dp_n_components, weight_concentration_prior_type='dirichlet_process')
    clf_dp_lda.fit(X_train_lda)
    y_pred_dp_lda = clf_dp_lda.predict(X_test_lda)
    
    #scores
    dp_arscore_lda = adjusted_rand_score(y_test, y_pred_dp_lda)
    
    pred_clusters_lda = set(y_pred_dp_lda)
    for i in pred_clusters_lda:
        ax_lda.scatter(X_test_tSNE[y_pred_dp_lda == i, 0], X_test_tSNE[y_pred_dp_lda == i, 1],  label = i,s=1)
    ax_lda.set_title(f'DPGMM predicted tSNE clusters,fitted on LDA-processed data, lda dimension = {lda_dim}\
    \n predicted num clusters:{len(pred_clusters_lda)}\
    \n adjusted random score: {dp_arscore_lda}')   

def experiment(dim=4,n_clusters=5,spread=20,lower_bound=0,upper_bound=200,dp_n_components=50,\
               n_clusters_mis=5,kmeans_test_start=None,kmeans_test_end=None):
    #data gen
    TOTAL_POINTS = 10000
    ratio_per_cluster = np.array([1/n_clusters]*n_clusters)
    X, labels, centers = generate_data_2(samples=TOTAL_POINTS, dim=dim, clusters=n_clusters, spread=spread,
                                   ratio_per_cluster=ratio_per_cluster,lower_bound=lower_bound, upper_bound=upper_bound)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)
    #fit
    clusters = set(labels)
    tSNE = TSNE(n_components=2, init='pca')
    X_test_tSNE=tSNE.fit_transform(X_test)
    
    #plot true test clustering
    fig,((ax_true,ax_pred),(ax_kmean_true,ax4)) = plt.subplots(2,2,figsize = (20,20))
    fig.suptitle(f'dimension:{dim},n_clusters:{n_clusters},spread:{spread},lb:{lower_bound},ub:{upper_bound}')
    for i in clusters:
        ax_true.scatter(X_test_tSNE[y_test == i, 0], X_test_tSNE[y_test == i, 1],  label = i,s=1)
    ax_true.set_title(f'Ground truth tSNE clusters \n actual num clusters:{len(clusters)}')
    #fit dpgmm
    clf_dp = BGM(n_components=dp_n_components, weight_concentration_prior_type='dirichlet_process')
    clf_dp.fit(X_train)
    y_pred_dp = clf_dp.predict(X_test)
    


    
    #scores
    dp_rscore = rand_score(y_test, y_pred_dp)
    dp_sscore = silhouette_score(X_test, y_pred_dp)
    dp_arscore = adjusted_rand_score(y_test, y_pred_dp) # better than rand index as a metrics?

    
    pred_clusters = set(y_pred_dp)
    for i in pred_clusters:
        ax_pred.scatter(X_test_tSNE[y_pred_dp == i, 0], X_test_tSNE[y_pred_dp == i, 1],  label = i,s=1)
    ax_pred.set_title(f'DPGMM predicted tSNE clusters, n_components = {dp_n_components}\
    \n predicted num clusters:{len(pred_clusters)}\
    \n adjusted random score: {dp_arscore}')
    
    
    #kmean
    clf_kmeans_true = KMeans(n_clusters=n_clusters)
    clf_kmeans_true.fit(X_train)
    y_pred_kmeans_true = clf_kmeans_true.predict(X_test)
    km_true_arscore = adjusted_rand_score(y_test, y_pred_kmeans_true)
    pred_clusters_kmeans_true = set(y_pred_kmeans_true)
    for i in pred_clusters_kmeans_true:
        ax_kmean_true.scatter(X_test_tSNE[y_pred_kmeans_true == i, 0], X_test_tSNE[y_pred_kmeans_true == i, 1],  label = i,s=1)
    ax_kmean_true.set_title(f'k mean,given true cluster size, predicted tSNE clusters\
    \n adjusted random score: {km_true_arscore}')
    
    #kmean mispecify
#     clf_kmeans_mispecify = KMeans(n_clusters=n_clusters_mis)
#     clf_kmeans_mispecify.fit(X_train)
#     y_pred_kmeans_mispecify = clf_kmeans_mispecify.predict(X_test)
#     km_mispecify_arscore = adjusted_rand_score(y_test, y_pred_kmeans_mispecify)
#     pred_clusters_kmeans_mispecify = set(y_pred_kmeans_mispecify)
#     for i in pred_clusters_kmeans_true:
#         ax_kmean_mispecify.scatter(X_test_tSNE[y_pred_kmeans_mispecify == i, 0], X_test_tSNE[y_pred_kmeans_mispecify == i, 1],  label = i,s=1)
#     ax_kmean_mispecify.set_title(f'k mean, with mispecified cluster size= {n_clusters_mis}, predicted tSNE clusters\
#     \n adjusted random score: {km_mispecify_arscore}')
    
    
    #kmean plot
    test_ks = []
    scores = []
    if not kmeans_test_start:
        kmeans_test_start=max(n_clusters-5,0)
    if not kmeans_test_end:
        kmeans_test_end=n_clusters+5
    for k in range(kmeans_test_start,kmeans_test_end+1):
        clf = KMeans(n_clusters = k)
        clf.fit(X_train)
        y_pred = clf.predict(X_test)
        ar_score = adjusted_rand_score(y_test,y_pred)
        test_ks.append(k)
        scores.append(ar_score)
    
    ax4.plot(test_ks,scores)
    ax4.set_title(f'Model selection for k means: Adjusted Random Score vs. num of clusters')
    ax4.set_xlabel('number of clusters')
    ax4.set_ylabel('Adjusted Random Score')

    ax4.hlines(y=dp_arscore, xmin=kmeans_test_start, xmax=kmeans_test_end+1, linewidth=2, color='r')


def generate_data_1(dim,clusters,samples_per_cluster,spread=3.0, lb=0,ub=200):

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


def generate_data_2(samples, dim, clusters, ratio_per_cluster, spread=3.0, lower_bound=0, upper_bound=200):

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

  indices = np.arange(samples)
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

def dimension_reduction_TSNE(X, n_components=2):
  """
  Parameters
  ---------
  X: data with unknown cluster labels
  n_components: number of components to retain
  Returns
  ---------
  X: reduced data with n_components
  """
  tSNE = TSNE(n_components=n_components, init='random')
  X=tSNE.fit_transform(X)
  return X

def select_n_dim_data(n,X,labels):
  X_test=[]
  y_test=[]
  for j in range(len(labels)):
    if labels[j] in n:
      X_test.append(X[j])
      y_test.append(labels[j])

  X_res=np.asarray(X_test)
  label_res=np.asarray(y_test)
  return X_res, label_res


def fit_K_means(X,n_clusters):
  kmeans = KMeans(n_clusters=n_clusters).fit(X)
  centers=kmeans.cluster_centers_
  labels=kmeans.labels_
  return centers, labels


def prior_estimation(est,n_observations,data_dim):
  a=est/np.log10(n_observations)
  alpha=2
  beta=alpha*a
  beta=np.floor(beta)
  if est==0:
    alpha=1
    beta=1
  v=data_dim+1
  W0=np.identity(data_dim)
  return alpha, beta,v , W0

def calculate_alpha(alpha,dp_n_components):
    clf_dp = BGM(n_components=dp_n_components, weight_concentration_prior_type='dirichlet_process')
    clf_dp.fit(X_train)
    y_pred_dp = clf_dp.predict(X_test)
    dp_arscore = adjusted_rand_score(y_test, y_pred_dp)
    
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(calculate_alpha, [i for i in np.linspace(0.1,2,100)])
    return results