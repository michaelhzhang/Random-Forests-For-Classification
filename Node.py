import numpy as np
import scipy as sc
from numpy import random
from collections import defaultdict

class Node:
	"""Node of a decision tree. Does the bulk of the work.

	Implementation Notes: 
	- If and only if self.label is not None, then this node is a leaf node and the
	label is the assigned label.
	- self.split_rule is a tuple (feature_index,split_threshold) that encodes the splitting rule
	- We pass around the indices of the data that corresponds to each node instead of taking slices of the data itself to avoid storing unnecessary copies of the data.
	- By convention, send features that match a threshold exactly to the left node"""
	
	def __init__(self, depth, impurity_measure, min_impurity_decrease,
		min_impurity, max_percentage_in_class, max_height,
		min_points_per_node, feature_name_map):
		""" Parameters:
		* depth: depth of this node
		* impurity measure: function of the form f(left_label_hist,right_label_hist) that measures the entropy of a node 
		* The following are for stopping criteria:
		- min_impurity_decrease: Minimum decrease in impurity needed on a split
		- min_impurity: Minimum amount of impurity needed to continue splitting in a node
		- max_percentage_in_class: Maximum amount of a node we can allocate to any 
		- max_height: Maximum height of a tree. If max_height is None, then trees can grow arbitrarily big
		- minimum number of data points we allow in a node. If less, stop splitting

		* Optional: feature_name_map: dictionary that maps feature numbers to feature names. Useful for debugging + printing trees. """

		self.depth = depth

		# Stopping criteria stuff
		self.impurity_measure = impurity_measure
		self.min_impurity_decrease = min_impurity_decrease
		self.min_impurity = min_impurity
		self.max_percentage_in_class = max_percentage_in_class
		self.max_height = max_height
		self.min_points_per_node = min_points_per_node

		self.feature_name_map = feature_name_map

		# Initialization 
		self.reset()

	def reset(self):
		"""Resets to an untrained state."""
		self.left = None
		self.right = None
		self.label = None 
		self.split_rule = None
		self.trained = False

	def __str__(self):
		# Does a DFS to return a string representation recursively.
		if (not self.trained):
			return "Not yet trained"

		if (self.is_leaf()):
			return str(self.label)
		else:
			this_node = self.split_rule_to_string(self.split_rule,self.feature_name_map)
			left_node = str(self.left)
			right_node = str(self.right)
			final_string = str(self.depth) + ": " + this_node + '\n'
			final_string += ('  '*self.depth)
			final_string += 'L: ' + left_node + '\n'
			final_string += ('  '*self.depth)
			final_string += 'R: ' + right_node 
			return final_string

	def split_rule_to_string(self,split_rule,feature_name_map):
		if feature_name_map is None:
			feature = split_rule[0]
		else:
			feature = feature_name_map[split_rule[0]]
		threshold = str(split_rule[1])
		return "(" + str(feature) + " > " + threshold + ")"

	def train(self,orig_training_data,orig_training_labels, data_indices, features_per_split):
		"""
		* orig_training_data: ORIGINAL numpy matrix of data passed into DecisionTree, where row = sample point, column = feature
		* orig_training_labels: ORIGINAL nparray of labels passed into DecisionTree, where item i is the label for point i
		* data_indices: Indices of data on which to train this node
		* features_per_split: number of features to split on (for feature bagging)
		"""
		# Reset things in case we train multiple times on the same node
		self.reset()
		
		# Initialization
		self.trained = True
		self.num_features = (orig_training_data.shape)[1]
		self.data_indices = data_indices;
		self.num_points = data_indices.shape[0]
		self.feature_indices = self.bag_features(features_per_split)

		self.histogram = self.compute_node_histogram(orig_training_labels,self.data_indices)
		self.node_impurity = self.compute_node_impurity(self.histogram)

		if (self.stop_splitting()):
			self.label = self.calculate_label(orig_training_labels, self.data_indices) # Mark as leaf node
		else:
			self.split_rule = self.segmenter(orig_training_data,
				orig_training_labels,
				self.data_indices,
				self.feature_indices)
			if (self.split_rule is None): # There were no valid splits to make
				self.label = self.calculate_label(orig_training_labels, self.data_indices)
			else:
				self.left = self.construct_child_node()
				self.right = self.construct_child_node()
				# Split up data between left and right nodes
				left_indices, right_indices = self.split_data(orig_training_data,
					self.data_indices,
					self.split_rule)
				self.left.train(orig_training_data,orig_training_labels,left_indices,features_per_split)
				self.right.train(orig_training_data,orig_training_labels,right_indices,features_per_split)

	def bag_features(self,features_per_split):
		"""Returns index of features that this node will train on"""
		return np.random.choice(self.num_features,features_per_split,replace=False)

	def compute_node_histogram(self,orig_training_labels,data_indices):
		"""Computes the histogram of this node"""
		training_labels = self.slice_labels(orig_training_labels, data_indices)
		histogram = defaultdict(int)
		for label in training_labels:
			histogram[label] += 1
		return histogram 

	def compute_node_impurity(self,histogram):
		"""Computes the impurity of this node."""
		left_histogram = histogram
		right_histogram = {} # Doesn't matter
		return self.impurity_measure(left_histogram,right_histogram)

	def segmenter(self,orig_data,orig_labels,data_indices,feature_indices):
		"""Chooses the best splitting rule for a node given a subset of
		data and features."""
		candidate_rules = {} # {feature: (split_threshold,information_gain)} 
		for feature in feature_indices:
			candidate_split = self.find_best_split(orig_data,orig_labels,data_indices,feature)
			if candidate_split is not None:
				candidate_rules[feature] = candidate_split
		best_rule = None
		best_info_gain = None
		for feature in candidate_rules:
			candidate_threshold = candidate_rules[feature][0]
			candidate_information_gain = candidate_rules[feature][1]
			if (best_info_gain is None) or (candidate_information_gain > best_info_gain):
				best_rule = (feature, candidate_threshold)
				best_info_gain = candidate_information_gain

		return best_rule

	def find_best_split(self,orig_data,orig_training_labels,data_indices,feature):
		"""Helper for segmenter. Finds the best split on a given feature.
		Returns None if there are no good candidate splits."""
		# Sort training data to be able to compute every candidate split in O(n log n) time,
		# vs O(n^2)

		training_data = orig_data[data_indices,feature].flatten()
		training_labels = self.slice_labels(orig_training_labels, data_indices)
		sorted_data,sorted_labels = self.sort_training_data(training_data,training_labels)

		candidate_threshold = sorted_data[0]
		best_threshold = None
		best_info_gain = None
		
		pointer, left_histogram, right_histogram, num_left_points, num_right_points = self.feature_split_first_pass(sorted_data,sorted_labels,candidate_threshold)


		best_threshold, best_info_gain = self.update_split_threshold(left_histogram,right_histogram,
			num_left_points,num_right_points,best_threshold,best_info_gain,candidate_threshold)

		# This loop tries every other split
		if (pointer < self.num_points):
			candidate_threshold = sorted_data[pointer]
		while (pointer < self.num_points):
			if (sorted_data[pointer] == candidate_threshold):
				# Move into left node
				num_left_points,num_right_points = self.move_point_left(left_histogram,right_histogram,
					num_left_points,num_right_points,sorted_labels,pointer)
			else: # Have come across a new threshold
				best_threshold, best_info_gain = self.update_split_threshold(left_histogram,right_histogram,
					num_left_points,num_right_points,best_threshold,best_info_gain,candidate_threshold)
				candidate_threshold = sorted_data[pointer]
				# Initialize new split
				num_left_points,num_right_points = self.move_point_left(left_histogram,right_histogram,
					num_left_points,num_right_points,sorted_labels,pointer)
			pointer += 1

		if (best_threshold is None) or (best_info_gain is None):
			return None
		else:
			return (best_threshold,best_info_gain)

	def feature_split_first_pass(self,sorted_data,sorted_labels,candidate_threshold):
		"""Helper for find_best_split. Performs the first split on the data and initializes variables."""
		pointer = 0
		# These store a count of each type of label in each node
		left_histogram = defaultdict(int) 
		right_histogram = defaultdict(int)
		# Num points in each node
		num_left_points = 0

		# Initial pass to initialize the histograms and try the first split
		while ((pointer < self.num_points) and (sorted_data[pointer] == candidate_threshold)):
			left_histogram[sorted_labels[pointer]] += 1
			num_left_points += 1
			pointer += 1
		# Have come across a new threshold (current pointer points to something different)
		# Everything now goes to the right
		num_right_points = self.num_points - num_left_points
		for i in xrange(pointer,self.num_points):
			right_histogram[sorted_labels[i]] += 1
		return pointer, left_histogram, right_histogram, num_left_points, num_right_points

	def update_split_threshold(self,left_histogram,right_histogram,num_left_points,num_right_points,best_threshold,best_info_gain,candidate_threshold):
		candidate_impurity = self.impurity_measure(left_histogram,right_histogram)
		candidate_information_gain = self.node_impurity - candidate_impurity
		if (self.valid_split(num_left_points,num_right_points,candidate_information_gain)
			and ((candidate_information_gain > best_info_gain) or (best_info_gain is None))):
			best_threshold = candidate_threshold
			best_info_gain = candidate_information_gain
		return best_threshold, best_info_gain

	def sort_training_data(self,training_data,training_labels):
		# Helper. training_data and training_labels should be a 1 x n numpy arrays
		sorted_indices = np.argsort(training_data,axis=None)
		sorted_data = training_data[sorted_indices]
		sorted_labels = training_labels[sorted_indices]
		return sorted_data,sorted_labels

	def valid_split(self,num_left_points,num_right_points,impurity_decrease):
		"""Helper for find_best_split. Returns False if the child nodes wouldn't have enough points 
		or the candidate split wouldn't decrease the impurity enough."""
		if (num_left_points < max(1,self.min_points_per_node)) or (num_right_points < max(1,self.min_points_per_node)):
			# Don't allow empty splits
			return False

		if (impurity_decrease < self.min_impurity_decrease):
			return False
		return True

	def stop_splitting(self):
		"""True iff we should not split on this node anymore"""
		if (self.node_impurity <= self.min_impurity) or ((self.max_height is not None) and (self.depth >= self.max_height)) or (self.num_points <= self.min_points_per_node):
			return True
		frequencies_by_class = [freq / (self.num_points * 1.0) for freq in self.histogram.values()]
		if (max(frequencies_by_class) > self.max_percentage_in_class):
			return True

		return False

	def move_point_left(self,left_histogram,right_histogram,num_left_points,num_right_points,sorted_labels,pointer):
		left_histogram[sorted_labels[pointer]] += 1
		right_histogram[sorted_labels[pointer]] -= 1
		num_left_points += 1
		num_right_points -= 1

		return num_left_points, num_right_points

	def slice_labels(self,training_labels,data_indices):
		"""Returns flat nparray of training labels of given indices"""
		return training_labels[data_indices]

	def mode(self,np_array):
		"""Computes mode of an numpy array"""
		counts = defaultdict(int)
		for val in np_array.tolist():
			counts[val] += 1
		best = 0
		mode = None
		for key in counts:
			if counts[key] > best:
				best = counts[key]
				mode = key
		return mode

	def calculate_label(self,training_labels,data_indices):
		# Makes the mode of the predictions of a class
		actual_labels = self.slice_labels(training_labels,data_indices)
		return self.mode(actual_labels)

	def construct_child_node(self):
		"""Initialize child node for this node."""
		child_node = Node(depth=self.depth+1, impurity_measure=self.impurity_measure, 
			min_impurity_decrease=self.min_impurity_decrease,
			min_impurity=self.min_impurity, 
			max_percentage_in_class=self.max_percentage_in_class,
			max_height=self.max_height,
			min_points_per_node=self.min_points_per_node,
			feature_name_map=self.feature_name_map)
		return child_node

	def split_data(self,orig_training_data,data_indices,split_rule):
		"""Splits data_indices into indices for left and right child nodes using split_rule"""
		split_feature = split_rule[0]
		threshold = split_rule[1]
		training_features = orig_training_data[data_indices,split_feature]
		threshold_vec = threshold * np.ones(training_features.shape)
		mask_vector = np.less_equal(training_features, threshold_vec)
		# <= goes to the left. > threshold goes to the right
		left_indices = data_indices[mask_vector]
		right_indices = data_indices[np.logical_not(mask_vector)]
		return left_indices, right_indices

	def predict(self,data):
		"""Given n x d test data, return a 1 x n nparray of classifications."""
		predictions = []
		for i in xrange(data.shape[0]):
			point = np.transpose(data[i,:])
			prediction = self.predict_point(point)
			predictions.append(prediction)
		return np.array([predictions])

	def predict_point(self,data): 
		"""Makes a prediction given data. Assumes data is d x 1,
		i.e. is a column vector."""
		assert (data.shape[0] == self.num_features)
		if (self.is_leaf()):
			return self.label 
		else:
			assert (self.left is not None)
			assert (self.right is not None)
			split_feature = self.split_rule[0]
			split_threshold = self.split_rule[1]
			if (data[split_feature] <= split_threshold):
				return self.left.predict_point(data)
			else:
				return self.right.predict_point(data)

	def is_leaf(self):
		"""Returns whether or not this is a leaf node"""
		return (self.label is not None)

	def print_splits(self,data): 
		"""Given a data point, classifies it and prints the path it followed
		to the leaf node. Data should be a column vector."""
		print str(self.depth) + ".",
		if (self.is_leaf()):
			print self.label
		else:
			split_feature = self.split_rule[0]
			split_threshold = self.split_rule[1]
			if (self.feature_name_map is not None):
				print self.feature_name_map[split_feature],
			else:
				print split_feature,
			if (data[split_feature] <= split_threshold):
				print "<= " + str(split_threshold)
				self.left.print_splits(data)
			else:
				print "> " + str(split_threshold)
				self.right.print_splits(data)


