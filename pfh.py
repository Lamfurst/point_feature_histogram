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
    
def getCorrespondencesPFH(pfh_sig_source, pfh_sig_target):
    # pfh_source and pfh_target are lists of PFH signatures
    # return a list of correspondences
    pfh_sig_target_array = np.array(pfh_sig_target)

    correspondeces = []

    for i in range(len(pfh_sig_source)):
        source_pfh = pfh_sig_source[i]
        differences = pfh_sig_target_array - source_pfh
        distances = np.linalg.norm(differences, axis=1)
        min_idx = np.argmin(distances)
        correspondeces.append([i, min_idx, distances[min_idx]])
    return correspondeces

def getTransform(Cp, Cq):
    # Cp np.array(3,n)
    # Cq np.array(3,n)
    p_bar = np.mean(Cp, axis=1).reshape(3, 1)
    p_centered = Cp - p_bar
    q_bar = np.mean(Cq, axis=1).reshape(3, 1)
    q_centered = Cq - q_bar
    S = p_centered @ q_centered.T
    U, _, Vt = np.linalg.svd(S)
    detVUT = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, detVUT]) @ U.T
    t = q_bar - R @ p_bar
    return R, t

def tranformPCArray(pc_array, R, t):
        pc_array_transformed = R @ pc_array + t
        return pc_array_transformed
    

def solve_pfh(source_pc_array, target_pc_array, radius, div):
    PFH_source = point_feature_histogram(radius, source_pc_array, div)
    pfh_sig_source = PFH_source.computePFHSignatures()
    print("...Done computing PFH signatures for source point cloud")

    PFH_target = point_feature_histogram(radius, target_pc_array, div)
    pfh_sig_target = PFH_target.computePFHSignatures()
    print("...Done computing PFH signatures for target point cloud")

    C = getCorrespondencesPFH(pfh_sig_source, pfh_sig_target)

    # Sort C by distance C is a list of (source_idx, target_idx, distance)
    C.sort(key=lambda x: x[2])

    # Run RANSAC on the top 30% of the correspondences
    _len = min(len(C) * 0.3, 150)
    C = C[:int(_len)]



    # # Run the RANSAC algorithm
    # ## Cup and Plant
    # # iter = 10000
    # # threshold = 0.01
    # # N = len(C) * 0.8

    # # Face
    # iter = 20000
    # threshold = 0.05
    # N = len(C) * 0.8
    # R = None
    # t = None
    # error = np.inf

    # for _it in range(iter):
    #     # Randomly pick 5 correspondences
    #     idx = np.random.choice(len(C), 4, replace=False)
    #     Cp = []
    #     Cq = []
    #     for i in idx:
    #         p = source_pc_array[:,C[i][0]]
    #         q = target_pc_array[:,C[i][1]]
    #         Cp.append(p)
    #         Cq.append(q)
    #     Cp = np.hstack(Cp)
    #     Cq = np.hstack(Cq)
    #     R_tmp, t_tmp = getTransform(Cp, Cq)

    #     # For correspondences not used in RANSAC, compute the error
    #     # If error is less than threshold, add the correspondences to the set cp and cq
    #     Cp_tmp = []
    #     Cq_tmp = []
    #     for i in range(len(C)):
    #         if i not in idx:
    #             p = source_pc_array[:,C[i][0]]
    #             q = target_pc_array[:,C[i][1]]
    #             error_outlier = np.linalg.norm(q - R_tmp @ p - t_tmp)**2
    #             if error_outlier < threshold:
    #                 Cp_tmp.append(p)
    #                 Cq_tmp.append(q)
    #     if len(Cp_tmp) != 0:
    #         Cp_tmp = np.hstack(Cp_tmp)
    #         Cq_tmp = np.hstack(Cq_tmp)
    #         Cp = np.hstack([Cp, Cp_tmp])
    #         Cq = np.hstack([Cq, Cq_tmp])
    #     if Cp.shape[1] > N:
    #         R_tmp, t_tmp = getTransform(Cp, Cq)
    #         Cp_transformed = tranformPCArray(Cp, R_tmp, t_tmp)
    #         error_tmp = np.linalg.norm(Cq - Cp_transformed)**2
    #         if error_tmp < error:
    #             R = R_tmp
    #             t = t_tmp
    #             error = error_tmp



    # iter = 100000
    iter = 1000
    # iter = 30000
    R = None
    t = None

    # Error is iinifity
    error = np.inf
    
    for _it in range(iter):
        # Randomly pick 5 correspondences
        idx = np.random.choice(len(C), 4, replace=False)
        # idx = np.random.choice(len(C), 6, replace=False)
        Cp = []
        Cq = []
        for i in idx:
            p = source_pc_array[:,C[i][0]]
            q = target_pc_array[:,C[i][1]]
            Cp.append(p)
            Cq.append(q)
        Cp = np.hstack(Cp)
        Cq = np.hstack(Cq)
        R_tmp, t_tmp = getTransform(Cp, Cq)

        Cp_transformed = tranformPCArray(Cp, R_tmp, t_tmp)
        error_tmp = np.linalg.norm(Cq - Cp_transformed)**2
        if error_tmp < error:
            R = R_tmp
            t = t_tmp
            error = error_tmp

    source_pc_array_transformed = tranformPCArray(source_pc_array, R, t)
    return source_pc_array_transformed, error, R, t


    