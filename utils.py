import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from scipy import linalg

import torch 

def dimension_reduction_LDA(X_train, y_train, X, n_components):
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
  tSNE = TSNE(n_components=n_components, init='pca')
  X=tSNE.fit_transform(X)
  return X

def fit_K_means(X,n_clusters):
  """
  Helper method to 

  Parameters
  ----------
  X : ndarray
    Data with unknown cluster labels.
  n_clusters : int
    Expected number of clusters in X.

  Returns
  -------
  centers : ndarray of shape (n_clusters, X.shape[1])
    Coordinates of cluster centers.
  labels : ndarray of shape (X.shape[0],)
    Labels of each sample.

  """
  kmeans = KMeans(n_clusters=n_clusters).fit(X)
  centers=kmeans.cluster_centers_
  labels=kmeans.labels_
  return centers, labels

def prior_estimation(est, n_observations, data_dim):
  a = est / np.log10(n_observations)
  alpha=2
  beta=alpha*a
  beta=np.floor(beta)
  if est==0:
    alpha=1
    beta=1
  v=data_dim+1
  W0=np.identity(data_dim)
  return alpha, beta, v, W0

def train(model, device, train_loader, criterion, optimizer):
  model.train()

  train_loss = 0

  for _, (data, _) in enumerate(train_loader):
    
    data = data.to(device)
    optimizer.zero_grad()
    _, decoded_data = model(data)

    loss = criterion(data, decoded_data)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  avg_loss = train_loss / len(train_loader)
  
  return avg_loss

def get_datasets(model, device, train_loader, test_loader, batch_size):
  X_test = []
  X_train = []
  decode_X_train = []

  with torch.no_grad():
    for _, (data, _) in enumerate(train_loader):
      data = data.to(device)

      transformed_data, decoded_data = model(data)
      transformed_data = np.asarray(transformed_data.detach().cpu().numpy())
      decoded_data = np.asarray(decoded_data.detach().cpu().numpy())

      X_train.append(transformed_data)
      decode_X_train.append(decoded_data)
    
    for _, (data, _) in enumerate(test_loader):
      data = data.to(device)

      transformed_data, _ = model(data)
      transformed_data = np.asarray(transformed_data.detach().cpu().numpy())

      X_test.append(transformed_data)

  rep_dim = X_train[0].shape[1]

  out_dim = decode_X_train[0].shape[1]
  X_train = np.reshape(np.asarray(X_train), (len(train_loader) * batch_size, rep_dim))
  X_test = np.reshape(np.asarray(X_test), (len(test_loader) * batch_size, rep_dim))
  decode_X_train = np.reshape(np.asarray(decode_X_train), (len(train_loader) * batch_size, out_dim))

  return X_train, X_test, decode_X_train

def plot_results(X, Y_, means, covariances, title):
  """
  Modified from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html 
  Visualization of clustering results.
  """
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
  ell.set_alpha(0.5)
  ax.add_artist(ell)

  plt.xticks(())
  plt.yticks(())
  plt.legend()
  plt.title(title)
  plt.show()