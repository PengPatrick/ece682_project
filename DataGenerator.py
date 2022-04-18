import numpy as np

class DataGenerator:
  def __init__(self, samples, n_features, n_clusters, ratio_per_cluster, spread=3.0, lower_bound=0, upper_bound=200):
    '''
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

    '''

    self.samples = samples
    self.dim = n_features
    self.n_clusters = n_clusters
    self.ratio = ratio_per_cluster
    self.spread = spread
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def generate_data(self):

    '''
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.
    labels : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.
    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster.
    '''
    
    if len(self.ratio) != self.n_clusters:
      exit("Number of clusters mismatches!")

    samples_per_cluster = np.int64(self.samples * self.ratio)
    print(samples_per_cluster)

    centers = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.n_clusters, self.dim))
    
    if isinstance(self.spread, (float, int)):
      self.spread  = [self.spread] * 2 
    
    dim_spread=np.random.uniform(np.sqrt(self.spread[0]), np.sqrt(self.spread[1]), size=(self.n_clusters, self.dim))
    
    X = []
    labels = []
    for i, (n, std) in enumerate(zip(samples_per_cluster, dim_spread)):
      X.append(np.random.normal(loc=centers[i], scale=std, size=(n, self.dim)))
      labels += [i] * n

    X = np.concatenate(X)
    labels = np.array(labels)

    indices = np.arange(self.samples)
    np.random.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    return X, labels, centers
