#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class point_feature_histogram(object):
    def __init__(self, radius, pc, div = 5):
        # pc is a 3xn numpy array of points
        self._radius = radius
        self._pc = pc
        self._tree = KDTree(pc.T)
        self._div = div

        # Setup thresholds for each feature
        self._f1_thresholds = self._generateThreshold(-1, 1)
        self._f2_thresholds = self._generateThreshold(0, self._radius)
        self._f3_thresholds = self._generateThreshold(-1, 1)
        self._f4_thresholds = self._generateThreshold(-np.pi, np.pi)

        self._pair_feature_hist = {}

    def _generateThreshold(self, start, end):
        # Helper function to generate thresholds for different features
        step_size = (end - start) / self._div
        return [start + i * step_size for i in range(1, self._div)]

    def _getNeighbors(self, p):
        # p is a 3x1 numpy array
        # return a list of indices of points within radius of p
        # and a 3xn numpy array of the neighbors
        nb_idx = self._tree.query_ball_point(p.T, self._radius)[0]
        nb_point = self._pc[:,nb_idx]
        return nb_idx, nb_point
    
    def _getNormal(self, p, nb_idx, nb_point):
        # p is a 3x1 numpy array
        # return a 3x1 numpy array representing the normal at p
        n = len(nb_idx)
        centroid = np.mean(nb_point, axis=1).reshape(3, 1)
        # subtract the centroid from the points
        pc_array_centered = nb_point - centroid
        # compute the covariance matrix
        Q = pc_array_centered.dot(pc_array_centered.T) / (n - 1)
        # SVD
        _, S, Vt = np.linalg.svd(Q)
        normal_vec = Vt.T[:, 2] #3x1

        # flip the normal if it points away from the viewpoint
        # viewpoint is the origin
        if normal_vec.T @ -p < 0:
            normal_vec = -normal_vec
        return normal_vec
    
    def _computePairFeatures(self, pi_idx, pj_idx, normal_i, normal_j, feature_num = 3):
        # pi_idx and pj_idx are the indices of the points
        # pi_normal_vec and pj_normal_vec are the normals at the points
        # return a 1x3 numpy array of the pair features
        pi = self._pc[:,pi_idx]
        pj = self._pc[:,pj_idx]
        if normal_i.T @ (pj - pi) <= normal_j.T @ (pi - pj):
            ps = pi
            pt = pj
            ns = normal_i
            nt = normal_j
        else:
            ps = pj
            pt = pi
            ns = normal_j
            nt = normal_i
        u = ns
        v = np.cross((pt - ps).T, u.T).T
        v = v / np.linalg.norm(v)
        w = np.cross(u.T, v.T).T
        f1 = v.T @ nt
        f2 = np.linalg.norm(pt - ps)
        f3 = u.T @ (pt - ps) / f2
        f4 = np.arctan((w.T @ nt) / (u.T @ nt))
        if feature_num == 3:
            return np.array([f1, f3, f4]).reshape(3, 1)
        return np.array([f1, f2, f3, f4]).reshape(4, 1)
    
    def _step(self, si, fi):
        for i in range(self._div - 1):
            if fi < si[i]:
                return i
        return self._div - 1

    def _computeHistogram(self, feature_list):
        s_list = []
        if len(feature_list) == 0:
            return np.zeros(self._div ** 4)
        _dim = feature_list[0].shape[0]
        if _dim == 3:
            s_list = [self._f1_thresholds, self._f3_thresholds, self._f4_thresholds]
        elif _dim == 4:
            s_list = [self._f1_thresholds, self._f2_thresholds, self._f3_thresholds, self._f4_thresholds]
        else:
            print(feature_list[0].shape)
            raise Exception("Invalid feature list length")
        histogram = np.zeros(self._div ** _dim)
        for feature in feature_list:
            index = 0
            for i in range(_dim):
                index += self._step(s_list[i], feature[i]) * self._div ** i
            histogram[index] += 1
        for i in range(len(histogram)):
            histogram[i] /= len(feature_list)
        return histogram
        
    def _computePointPFHSignature(self, p_idx, normal_vec_list, nb_idx_list):
        # For each pair of neighbors, compute pair features
        # and concatenate them into a single feature vector
        # return the geometric_feature (histogram) of the point p_idx
        nb_idx = nb_idx_list[p_idx]
        n_feature = len(nb_idx) * (len(nb_idx) - 1) // 2
        feature_list = [None] * n_feature
        curr_pose = 0
        for i in range(len(nb_idx)):
            for j in range(i):
                pi_idx = nb_idx[i]
                pj_idx = nb_idx[j]
                pi_normal_vec = normal_vec_list[pi_idx]
                pj_normal_vec = normal_vec_list[pj_idx]
                if self._pair_feature_hist.get((pi_idx, pj_idx)) is not None:
                    feature = self._pair_feature_hist[(pi_idx, pj_idx)]
                else:
                    feature = self._computePairFeatures(pi_idx, pj_idx, pi_normal_vec, pj_normal_vec)
                    self._pair_feature_hist[(pi_idx, pj_idx)] = feature
                feature_list[curr_pose] = feature
                curr_pose += 1
        return self._computeHistogram(feature_list)
    
    def computePFHSignatures(self):
        # Compute the PFH signature for each point
        pc_num = self._pc.shape[1]
        normal_vec_list = [None] * pc_num
        nb_idx_list = [None] * pc_num
        for idx in range(pc_num):
            p = self._pc[:,idx]
            nb_idx, nb_point = self._getNeighbors(p)
            normal_vec = self._getNormal(p, nb_idx, nb_point)
            normal_vec_list[idx] = normal_vec
            nb_idx_list[idx] = nb_idx
        pfh_list = [None] * pc_num
        for idx in range(pc_num):
            pfh = self._computePointPFHSignature(idx, normal_vec_list, nb_idx_list)
            pfh_list[idx] = pfh
        return pfh_list
    
    def solve(self, source_pc, target_pc):
        # source_pc and target_pc are 3xn numpy arrays
        # return the transformation from source_pc to target_pc
        pass


    # Steps:
    # 1. Compute the normal at each point in the point cloud
    # 2. Compute the PFH signature for each point
    # 3. Compute the transformation between the source and target point clouds

    """
    normal_vec_list = []
    nb_idx_list = []

    for idx, p in enumerate(pc_source):
        nb_idx, nb_point = getNeighbors(p)
        normal_vec = getNormal(p, nb_idx, nb_point)

        normal_vec_list.append(normal_vec)
        nb_idx_list.append(nb_idx)
    
    # Compute the PFH signature for each point
    pfh_list = []
    for idx, p in enumerate(pc_source):
        pfh = computePointPFHSignature(p, normal_vec_list, nb_idx_list)
        pfh_list.append(pfh)
    
    """
    


    