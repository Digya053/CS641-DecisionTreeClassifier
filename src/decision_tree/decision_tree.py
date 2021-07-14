import numpy as np

class DecisionTree:
	"""
	A class to construct decision tree

	Attributes
	----------
		min_sample_size: int
			Minimum sample size required to split the data
		max_depth: int
			Maximum possible depth 
		mode: string
			To compute "gini" or "entropy" as a measure of impurity
	"""

	def __init__(self, min_sample_size=2, max_depth=1000, mode="gini"):
		self.min_sample_size = min_sample_size
		self.max_depth = max_depth
		self.mode = mode

	def check_purity(self, data):
		"""Check if a data contains only a single class label.
		Parameters
		----------
			data: DataFrame
				Dataset with features and target
		"""
		
		if len(np.unique(data[:, -1])) == 1:
			return True
		else:
			return False

	def classify_data(self, data):
		"""Classify data into classes based on the maximum occurence
		Parameters
		----------
			data: DataFrame
				Dataset with features and target
		Returns
		--------
			class label
		"""

		unique_classes, count_unique_classes = np.unique(data[:, -1], return_counts = True)
		return unique_classes[count_unique_classes.argmax()]

	def get_potential_splits(self, data):
		"""Get all the unique value of a particular column of a 
		dataframe
		Parameters
		----------
			data: DataFrame
				Dataset with features and target
		Returns
		-------
			Dictionary of potential splits
		"""

		potential_splits = {}
		_, n_columns = data.shape
		for column in list(range(n_columns -1)):
			values = data[:, column]
			unique_values = np.unique(values)
			if len(unique_values) == 1:
				potential_splits[column] = unique_values
			else:
				potential_splits[column] = []
				for i in range(len(unique_values)):
					if i != 0:
						current_value = unique_values[i]
						previous_value = unique_values[i - 1]
						potential_splits[column].append((current_value + previous_value) / 2)
		return potential_splits

	def split_data(self, data, split_column, split_value):
		"""Split data in left and right branch

		Parameters
		----------
			data: DataFrame
				Dataset with features and target
			split_column: str
				Column to split
			split_value: float
				Value on the basis of which to split
		"""

		split_column_values = data[:, split_column]
		return data[split_column_values <= split_value], data[split_column_values > split_value]

	def calculate_impurity_parent(self, data):
		"""Calculate impurity of a parent class

		Parameters
		----------
			data: DataFrame
				Dataset with features and target
		Returns
		--------
			Impurity of a parent class (for both gini and entropy)
		"""

		_, unique_classes = np.unique(data[:, -1], return_counts=True)
		probability = unique_classes/unique_classes.sum()
		if self.mode == "entropy":
			return np.sum(probability * -np.log2(probability))
		else:
			return 1- np.sum(np.square(probability))

	def calculate_impurity_children(self, data_below, data_above):
		"""Calculate impurity of a parent class

		Parameters
		----------
			data: DataFrame
				Dataset with features and target
		Returns
		--------
			Impurity of a child class (for both gini and entropy)
		"""
		prob_data_below = len(data_below) / (len(data_below) + len(data_above))
		prob_data_above = len(data_above) / (len(data_below) + len(data_above))
		return prob_data_below * self.calculate_impurity_parent(data_below) + prob_data_above * self.calculate_impurity_parent(data_above)
	
	def calculate_information_gain(self, data, data_below, data_above):
		"""Calculate information gain by subtracting the impurity of a parent class
		with that of a child class.

		Parameters
		----------
			data: DataFrame
				Dataset with features and target
			data_below: DataFrame
				Child in the left position of tree
			data_above: DataFrame
				Child in the right position of tree
		Returns
		--------
			Information gain
		"""
		return self.calculate_impurity_parent(data) - self.calculate_impurity_children(data_below, data_above) 
	
	def determine_best_split(self, data, potential_splits):
		"""Determine best split column and split value based on information gain
		Parameters
		----------
			data: DataFrame
				Dataset with features and target
			potential_splits: dict
				Dictionary of all potential splits
		Returns
		-------
			best_split_column, best_split_value
		"""

		max_info_gain = -float("inf")
		best_split_column = 0
		best_split_value = 0
		for split_column in potential_splits:
			for split_value in potential_splits[split_column]:
				data_below, data_above = self.split_data(data, split_column, split_value)
				current_info_gain = self.calculate_information_gain(data, data_below, data_above)
				if current_info_gain >= max_info_gain:
					max_info_gain = current_info_gain
					best_split_column = split_column
					best_split_value = split_value
		return best_split_column, best_split_value

	def build_decision_tree(self, dataframe, current_depth=0):
		"""Build decision tree by checking purity of data, calculating potential splits and  
		determining best split

		Parameters:
		-----------
			dataframe: pd.DataFrame
				Dataset with features and target
			current_depth: int
				Depth of a decision tree

		Returns:
		--------
			Decision Tree
		"""
		if current_depth == 0:
			global COLUMNS
			COLUMNS = dataframe.columns
			data = dataframe.values
		else:
			data = dataframe
		if self.check_purity(data) or len(data) < self.min_sample_size or current_depth == self.max_depth:
			return self.classify_data(data)
		else:
			current_depth += 1
			potential_splits = self.get_potential_splits(data)
			split_column, split_value = self.determine_best_split(data, potential_splits)
			data_below, data_above = self.split_data(data, split_column, split_value)
			if len(data_below) == 0 or len(data_above) == 0:
				return self.classify_data(data)
			else:
				question = str(COLUMNS[split_column]) + " <= " + str(split_value)
				decision_tree = {question: []}
				positive = self.build_decision_tree(data_below, current_depth)
				negative = self.build_decision_tree(data_above, current_depth)
				if positive == negative:
					decision_tree = positive
				else:
					decision_tree[question].append(positive)
					decision_tree[question].append(negative)
				return decision_tree

	def classify(self, sample, tree):
		"""Classify the sample
		Parameters:
		-----------
		sample: pd.DataFrame
			Features
		tree: Decision tree which is to be used during classification

		Returns
		-------
			classification result
		"""

		if not isinstance(tree, dict):
			return tree
		question = list(tree.keys())[0]
		attribute, value = question.split(" <= ")
		if sample[attribute] <= float(value):
			answer = tree[question][0]
		else:
			answer = tree[question][1]
		return self.classify(sample, answer)
	
	def predictions(self, dataframe, tree):
		"""Predict the result of entire dataframe
		Parameters
		-----------
		dataframe: pd.DataFrame
			Features
		tree: dict 
			Decision tree which is to be used during classification

		Returns
		-------
			prediction result
		"""

		return dataframe.apply(self.classify, axis = 1, args = (tree,))

	def calculate_accuracy(self, predicted, actual):
		"""Calculates Accuracy
		Parameters
		----------
			predicted: numpy.ndarray
				Prediction result
			acutal: numpy.ndarray
				Actual result
		"""
		correct = predicted == actual
		return correct.mean()

	def calculate_error(self, predicted, actual):
		"""Calculates Error
		Parameters
		----------
			predicted: numpy.ndarray
				Prediction result
			acutal: numpy.ndarray
				Actual result
		"""
		incorrect = predicted != actual
		return incorrect.mean()
		



		
		
	
	
