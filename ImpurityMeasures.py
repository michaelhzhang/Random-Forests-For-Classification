### Library of impurity measures for use in training decision trees

import numpy as np 
import scipy as sc

def entropy(left_label_hist,right_label_hist):
	"""Computes entropy given a split, by computing the entropy for each
	and taking the weighted average."""
	left_total = count_total(left_label_hist)
	right_total = count_total(right_label_hist)
	if (left_total != 0):
		left_entropy = _entropy_helper(left_label_hist,left_total)
	else:
		left_entropy = 1 # Doesn't matter

	if (right_total != 0):
		right_entropy = _entropy_helper(right_label_hist,right_total)
	else:
		right_entropy = 1 # Doesn't matter

	assert (left_total + right_total > 0)

	total_entropy = weighted_average(left_total,left_entropy,right_total,right_entropy)

	return total_entropy

def _entropy_helper(histogram,total):
	""" - \sum_c p_c log_2(p_c)"""
	probabilities = [freq / (total * 1.0) for freq in histogram.values()]
	probabilities = [p for p in probabilities if (p != 0)] # Avoid logging 0
	probabilities = np.array(probabilities)
	log_prob = np.log2(probabilities)
	products = probabilities * log_prob
	entropy = np.sum(products)
	entropy = -1 * entropy
	return entropy

def count_total(histogram):
	total = 0.0
	for amount in histogram.values():
		total += amount
	return total

def weighted_average(left_total,left_value,right_total,right_value):
	numerator = (left_total*left_value + right_total*right_value)*1.0
	denominator = left_total + right_total
	return numerator / denominator

