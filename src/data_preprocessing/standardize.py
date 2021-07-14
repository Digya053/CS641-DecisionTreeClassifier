class Standardize:

	"""A class to standardize the features

	Attributes
	----------
	x: numpy.ndarray
		Features

	Methods
	-------
		compute_z_score
			Returns the standardized features
	"""

	def __init__(self, x):
		"""
		Parameters
		----------
		x: numpy.ndarray
			Features
		"""
		self.x = x
	
	def compute_z_score(self):
		"""
		Subtract features with the mean and divide by standard deviation

		Returns
		-------
		x_scaled: standardized data
		"""
		x_scaled = (self.x - self.x.mean())/self.x.std()
		return x_scaled
