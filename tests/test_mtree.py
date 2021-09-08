
#  Verify M-Tree results using a random data set

from dSalmon.trees import MTree
import numpy as np
import os
from scipy.spatial.distance import cdist

TARGET_SIZE = 10000
BLOCK_SIZE = 2000
DIMENSION = 4

def sigmoid(x):
	return 1/(1+np.exp(-x))

def test_mtree():
	tree = MTree()
	indices = np.zeros([0], dtype=np.int32)
	samples = np.zeros([0,DIMENSION], dtype=np.double)

	def draw_insert_size():
		# draw number of samples to insert randomly
		return np.random.binomial(BLOCK_SIZE, sigmoid((TARGET_SIZE-len(indices))/TARGET_SIZE/10) )
		
	def draw_remove_size():
		# draw number of samples to remove randomly
		remove_size = np.random.binomial(BLOCK_SIZE, 1-sigmoid((TARGET_SIZE-len(indices))/TARGET_SIZE/10) )
		return remove_size if remove_size < len(indices) else len(indices)
		
	for i in range(10000):
		if i % 10 == 0:
			print (i, len(indices))
		# Insert a few random points
		new_samples = np.random.rand(draw_insert_size(),DIMENSION)
		new_indices = tree.insert(new_samples)

		samples = np.concatenate([samples, new_samples])
		indices = np.concatenate([indices, new_indices])
		
		# Draw random point for querying
		query_point = np.random.rand(1,DIMENSION)
		radius = np.random.gamma(1,0.5)
		k = np.random.poisson(5)+1
		
		# Perform radius and knn query for tree
		tree_rq_res = tree.neighbors(query_point, radius)
		tree_knn_res = tree.knn(query_point, k)
		
		# Find points manually
		distances = cdist(samples, query_point)[:,0]
		mask = distances <= radius
		result_rq_ind = np.argsort(indices[mask])
		result_knn_ind = np.argsort(distances)[:k]
		
		# Compare
		tree_sorting_perm = np.argsort(tree_rq_res[0])
		assert (indices[mask][result_rq_ind] == tree_rq_res[0][tree_sorting_perm]).all()
		assert (distances[mask][result_rq_ind] == tree_rq_res[1][tree_sorting_perm]).all()
		
		assert (indices[result_knn_ind] == tree_knn_res[0]).all()
		assert (distances[result_knn_ind] == tree_knn_res[1]).all()
		
		# Randomly remove a few points
		remove_size = draw_remove_size()
		if remove_size > 0:
			to_delete = np.random.permutation(len(indices))
			del tree[indices[to_delete[:remove_size]]]
			samples = np.delete(samples, to_delete[:remove_size], axis=0)
			indices = np.delete(indices, to_delete[:remove_size], axis=0)

		assert len(tree) == samples.shape[0]
