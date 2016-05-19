from DecisionTree import DecisionTree
from Node import Node
import ImpurityMeasures
import math
import numpy as np
import scipy as sc
from numpy import random
from collections import defaultdict

class RandomForest:
	def __init__(self,num_trees,
		data_bagging_size,
		feature_bagging_criteria = lambda d: int(math.sqrt(d)),
		impurity_measure = ImpurityMeasures.entropy,
		min_impurity_decrease=0.0,
		min_impurity=0.0,
		max_percentage_in_class=0.9999,
		max_height=None,
		min_points_per_node=1,
		feature_name_map=None):
		"""Parameters:
		* num_trees: The number of decision trees to train this random forest on
		* data_bagging_size: The number of training points, sampled with replacement, with which to train each tree.
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
		self.num_trees = num_trees
		self.data_bagging_size = data_bagging_size
		self.feature_bagging_criteria = feature_bagging_criteria

		# Stopping criteria stuff
		self.impurity_measure = impurity_measure
		self.min_impurity_decrease = min_impurity_decrease
		self.min_impurity = min_impurity
		self.max_percentage_in_class = max_percentage_in_class
		self.max_height = max_height
		self.min_points_per_node = min_points_per_node

		self.feature_name_map = feature_name_map
		self.trees = []

	def train(self,data,labels):
		"""Trains the random forest using a bunch of decision trees.

		* training_data: n x d numpy matrix of data, where row = sample point, column = feature
		* training_labels: flat nparray of labels, where item i is the label for point i """
		num_points = data.shape[0]
		for i in xrange(self.num_trees):
			sample_indices = np.random.choice(num_points,size=self.data_bagging_size,replace=True)
			sample_data = data[sample_indices]
			sample_labels = labels[sample_indices]
			tree = DecisionTree(feature_bagging_criteria=self.feature_bagging_criteria,
				impurity_measure=self.impurity_measure,
				min_impurity_decrease=self.min_impurity_decrease,
				min_impurity=self.min_impurity,
				max_percentage_in_class=self.max_percentage_in_class,
				max_height=self.max_height,
				min_points_per_node=self.min_points_per_node,
				feature_name_map=self.feature_name_map)
			tree.train(sample_data,sample_labels)
			self.trees.append(tree)

	def predict(self,data):
		"""Uses voting of the constituent trees to make a classification.

		Given n x d test data, return an nparray of classifications."""
		if (len(self.trees) == 0):
			raise Exception("Forest not trained")
		
		classifications = []
		for i in xrange(data.shape[0]):
			votes = defaultdict(int)
			
			for tree in self.trees:
				to_predict = data[i]
				to_predict = np.reshape(to_predict,(1,data.shape[1]))
				prediction = tree.predict(to_predict)[0]
				votes[prediction] += 1

			classification = max(votes,key=votes.get) # argmax(votes)
			classifications.append(classification)

		return np.array(classifications)


	def print_root_splits(self):
		"""Prints the most common splits chosen by the roots of trees."""
		split_count = defaultdict(int)
		for tree in self.trees:
			rule = tree.root_split()
			if rule is None:
				split_count['Leaf node'] += 1
			else:
				split_count[rule] += 1
		sorted_counts = sorted(zip(split_count.keys(),split_count.values()),key = lambda tup: tup[1], reverse=True)
		print 'Most common root splits:'
		for i in range(min(len(sorted_counts),10)):
			key = sorted_counts[i][0]
			amount = sorted_counts[i][1]
			if (key is 'Leaf node'):
				print str(i) + '. Leaf node (' + str(amount) + " trees)"
			else:
				feature = sorted_counts[i][0][0]
				threshold = sorted_counts[i][0][1]

				if (self.feature_name_map is not None):
					feature = self.feature_name_map[feature]	
				print str(i) + '. ' + str(feature) + " <= " + str(threshold) + " (" + str(amount) + " trees)"


