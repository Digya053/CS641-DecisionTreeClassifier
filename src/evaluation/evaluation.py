import numpy as np

class Metrics:
	"""A class for handling evaluation metrics

	Parameters
	----------
	actual_values: np.ndarray
		Actual labels
	predicted_values: np.ndarry
		Target labels
	"""

	def __init__(self, actual_values, predicted_values):
		self.tp = 0
		self.tn = 0
		self.fp = 0
		self.fn = 0
		self.actual_values = actual_values
		self.predicted_values = predicted_values
		for actual, predicted in zip(self.actual_values, self.predicted_values):
			if predicted == actual:
				if predicted == 1:
					self.tp +=1
				else:
					self.tn +=1
			else:
				if predicted == 1:
					self.fp += 1
				else:
					self.fn +=1

	def confusion_matrix(self):
		"""Plot confusion matrix
		Returns
		-------
			Confusion matrix
		"""
		confusion_matrix = [
			[self.tn, self.fp],
			[self.fn, self.tp]
		]
		return np.array(confusion_matrix)

	def accuracy(self):
		"""Calculates accuracy
		Returns
		--------
			Accuracy
		"""
		return (self.tp + self.tn)/len(self.actual_values)

	def precision(self):
		"""Calculates precision
		Returns
		-------
			postive: precision for positive class
			negative: precision for negative class
		"""
		positive = self.tp /(self.tp + self.fp)
		negative = self.tn /(self.tn + self.fn)
		return positive, negative

	def recall(self):
		"""Calculates recall
		Returns
		-------
			postive: recall for positive class
			negative: recall for negative class
		"""
		positive = self.tp /(self.tp + self.fn)
		negative = self.tn /(self.tn + self.fp)
		return positive, negative

	def f1_score(self):
		"""Calculates f1-score
		Returns
		-------
			f1_postive: f1-score for positive class
			f1_negative: f1-score for negative class
		"""
		positive_precision, negative_precision = self.precision()
		positive_recall, negative_recall = self.recall()
		f1_positive = 2*(positive_precision * positive_recall)/(positive_precision + positive_recall)
		f1_negative = 2*(negative_precision * negative_recall)/(negative_precision + negative_recall)
		return f1_positive, f1_negative

	def report(self):
		"""Print evaluation metrics"""
		positive_precision, negative_precision = self.precision()
		positive_recall, negative_recall = self.recall()
		f1_positive, f1_negative = self.f1_score()
		print("Accuracy: " + str(np.round(self.accuracy(),2)))
		print("Precision (1): " + str(np.round(positive_precision,2)))
		print("Precision (0): " + str(np.round(negative_precision,2)))
		print("Recall (1): " + str(np.round(positive_recall,2)))
		print("Recall (0): " + str(np.round(negative_recall,2)))
		print("F1-score (1): " + str(np.round(f1_positive,2)))
		print("F1-score (0): " + str(np.round(f1_negative,2)))

