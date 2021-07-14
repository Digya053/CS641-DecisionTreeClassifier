import numpy as np

class PCA:
	"""
	A class to implement PCA. Here the maximum variance have been found 
	out using eigen vectors.

	Attributes
	----------
	num_components = int
		the number of features to keep (num_components has been set to 14
		in this project so as to keep the features preserving 91% variance
		in data)

	Methods
	-------
	fit(X)
		Computes n highest eigenvectors

	fit_transform(X)
		Computes dot product between eigenvectors and data
	"""
	def __init__(self, num_components):
		"""
		Parameters
		----------
		num_components: int
			Number of dimensions to return
		"""
		self.n_components = num_components
		# n highest eigenvectors
		self.eigenvector_subset = None
		# data
		self.X = None

	def fit(self, X):
		"""Computes n highest eigenvectors preserving 95% of variance

		Parameters
		----------
		X: numpy.ndarray
			Features
		"""
		self.X = X - np.mean(X, axis=0)
		cov = np.cov(self.X.transpose())
		eigen_values, eigen_vectors = np.linalg.eig(cov)
		sorted_index = np.argsort(eigen_values)[::-1]
		# Sort in descending order to preserve eigen values with maximum variance
		sorted_eigenvalues = eigen_values[sorted_index]
		sorted_eigenvectors = eigen_vectors[:,sorted_index]
		self.eigenvector_subset = sorted_eigenvectors[:,0:self.n_components]
		

	def fit_transform(self, X):
		"""Computes dot product between eigenvectors and data

		Parameters
		----------
		X: numpy.ndarray
			Features

		Returns
		-------
			Dot product between eigenvectors and data
		"""

		self.fit(X)
		return np.dot(self.eigenvector_subset.transpose(),self.X.transpose()).transpose()
