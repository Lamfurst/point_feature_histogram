#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class point_feature_histogram(object):
    def __init__(self, radius, pc):
        # pc is a 3xn numpy array of points
        self._radius = radius
        self._pc = pc
        self._tree = KDTree(pc.T)
        
    def _getNeighbors(self, p):
        # p is a 3x1 numpy array
        # return a list of indices of points within radius of p
        # and a 3xn numpy array of the neighbors
        nb_idx = self._tree.query_ball_point(p.T, self._radius)[0]
        print(nb_idx)
        nb_point = self._pc[:,nb_idx]
        print(nb_point.shape)
        return nb_idx, nb_point
    
    def _getNormal(self, p):
        # p is a 3x1 numpy array
        # return a 3x1 numpy array representing the normal at p
        nb_idx, nb_point = self._getNeighbors(p)
        n = len(nb_idx)
        centroid = numpy.mean(nb_point, axis=1).reshape(3, 1)
        # subtract the centroid from the points
        pc_array_centered = nb_point - centroid
        # compute the covariance matrix
        Q = pc_array_centered.dot(pc_array_centered.T) / (n - 1)
        # SVD
        _, S, Vt = numpy.linalg.svd(Q)
        normal_vec = Vt.T[:, 2] #3x1
        return normal_vec
    