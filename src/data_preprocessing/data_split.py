class Stratify:

	"""A class to split the data into train, test and validation set

	Attributes
	----------
		df: dataframe
			data which is to be splitted 
		column: str
			column on which the data has to be stratified
		n_percent: float
			percentage of split

	Methods
	-------
		stratify:
			Returns stratified dataframe

		train_test_split:
			Returns train and test dataframe
	"""
	
	def __init__(self, df, column, n_percent):
		self.df = df
		self.column = column
		self.n_percent = n_percent
		
	def stratify(self):
		"""Stratify the dataset
		Returns
		-------
			Stratified dataframe
		"""
		n_samples = (self.n_percent * self.df.shape[0])
		n = round(min(n_samples, self.df[self.column].value_counts().min()))
		unique, counts = np.unique(self.df[self.column], return_counts=True)
		y_count = dict(zip(unique, counts))
		df_ = self.df.groupby(self.column).apply(lambda x: x.sample(round(n/len(Counter(y_count)))))
		df_.index = df_.index.droplevel(0)
		return df_

	def train_test_split(self):
		"""Split the data into two subset

		Returns
		-------
			df_train, df_test
		"""
		df_test = self.stratify()
		df_train = self.df.drop(df_test.index.tolist()).sample(frac=1)
		return df_train, df_test	
