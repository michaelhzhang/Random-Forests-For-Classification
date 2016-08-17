import numpy as np
import scipy as sc
import ImpurityMeasures
from Node import Node

class DecisionTree:
	"""Decision tree for classification.

	Implementation Notes:
	- Does not store training data after training is finished to optimize memory."""
	def __init__(self,
		feature_bagging_criteria = lambda d: d,
		impurity_measure=ImpurityMeasures.entropy,
		min_impurity_decrease=0.0,
		min_impurity=0.0,
		max_percentage_in_class=0.9999,
		max_height=None,
		min_points_per_node=1,
		feature_name_map=None
		):
		"""Parameters:
		* feature_bagging_criteria: a function of the form f(d) that given the number of features, returns m = the amount of features to train on in each split
		* impurity measure: function of the form f(left_label_hist,right_label_hist) that measures the entropy of a node
		* The following are for stopping criteria:
		- min_impurity_decrease: Minimum decrease in impurity needed on a split
		- min_impurity: Minimum amount of impurity needed to continue splitting in a node
		- max_percentage_in_class: Maximum amount of a node we can allocate to any
		- max_height: Maximum height of a tree. If max_height is None, then trees can grow arbitrarily big
		- minimum number of data points we allow in a node. If less, stop splitting

		* OPTIONAL: feature_name_map: dictionary that maps feature numbers to feature names. Useful for debugging + printing trees.
		"""

		self.feature_bagging_criteria = feature_bagging_criteria

		# Stopping criteria stuff
		self.impurity_measure = impurity_measure
		self.min_impurity_decrease = min_impurity_decrease
		self.min_impurity = min_impurity
		self.max_percentage_in_class = max_percentage_in_class
		self.max_height = max_height
		self.min_points_per_node = min_points_per_node

		self.feature_name_map = feature_name_map
		self.root_node = None

	def __str__(self):
		if (self.root_node is None):
			return "DecisionTree not yet trained"
		else:
			return str(self.root_node)

	def train(self,data,labels):
		"""Grows a decision tree by constructing nodes.

		* training_data: n x d numpy matrix of data, where row = sample point, column = feature
		* training_labels: 1 dimensional vector of labels, where item i is the label for point i"""
		assert len(data.shape) == 2
		self.num_points = (data.shape)[0]
		self.num_features = (data.shape)[1]
		self.features_per_split = self.feature_bagging_criteria(self.num_features)

		assert (self.num_points == labels.shape[0])

		# Create a new root node in case we want to call train multiple times
		self.root_node = Node(depth=0,impurity_measure=self.impurity_measure,
			min_impurity_decrease=self.min_impurity_decrease,
			min_impurity=self.min_impurity,
			max_percentage_in_class=self.max_percentage_in_class,
			max_height=self.max_height,
			min_points_per_node=self.min_points_per_node,
			feature_name_map=self.feature_name_map)

		self.root_node.train(data,labels,np.arange(self.num_points),
			self.features_per_split)

	def predict(self,data):
		"""Given n x d test data, return a flat nparray of classifications."""
		predictions = []
		for i in xrange(data.shape[0]):
			point = np.transpose(data[i])
			prediction = self.predict_point(point)
			predictions.append(prediction)
		return np.array(predictions)


	def predict_point(self,data):
		"""Given a data point, traverse the tree to find the best label to classify the data point as.

		Assumes data is d x 1, i.e. is a column vector."""
		if (self.root_node is None):
			raise Exception('Decision Tree not yet trained')

		return self.root_node.predict_point(data)

	def print_splits(self,data):
		"""Given a data point, classifies it and prints the path it followed
		to the leaf node. Data should be a column vector."""
		self.root_node.print_splits(data)

	def root_split(self):
		"""Returns splitting rule of the root node"""
		if (self.root_node is not None):
			return self.root_node.split_rule

