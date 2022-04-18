from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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