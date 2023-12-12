#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import random
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    # utils.view_pc([pc])

    # Fit a plane to the data using ransac
    # pc is a list of 3x1 matrices
    def fit_plane(subset_points):
        # subset_points is a list of 3x1 matrices
        p1 = subset_points[0]
        p2 = subset_points[1]
        p3 = subset_points[2]
        v1 = p2 - p1
        v2 = p3 - p1
        normal_vec = numpy.cross(v1.T, v2.T).T
        normal_vec = normal_vec / numpy.linalg.norm(normal_vec)
        d = -normal_vec.T @ p1
        return numpy.vstack((normal_vec, d))
    
    def fit_plane_pca(subset_points):
        # subset_points is a list of 3x1 matrices
        pc_array = numpy.hstack(subset_points)
        # compute the centroid
        centroid = numpy.mean(pc_array, axis=1).reshape(3, 1)
        # subtract the centroid from the points
        pc_array_centered = pc_array - centroid
        # compute the covariance matrix
        Q = pc_array_centered @ pc_array_centered.T / (len(subset_points) - 1)
        # SVD
        _, S, Vt = numpy.linalg.svd(Q)
        normal_vec = Vt.T[:, 2]
        d = -normal_vec.T @ centroid
        normal_vec = normal_vec.reshape(3, 1)
        return numpy.vstack((normal_vec, d))
    
    def calculate_error(plane, point):
        return (numpy.abs(plane[0:3].T @ point + plane[3]) / numpy.linalg.norm(plane[0:3]))**2

    iter = 1000
    delta = 0.01
    N = 150
    error_best = numpy.inf
    plane_best = None
    # RUC = None
    # RUC_idx = None
    for _ in range(iter):
        # Randomly select 3 points
        # pc is a list of 3x1 matrices
        subset_points_idx = numpy.random.choice(len(pc), 3, replace=False).tolist()
        subset_points = []
        for idx in subset_points_idx:
            subset_points.append(pc[idx])
        plane = fit_plane(subset_points)
        C = []
        C_idx = []
        for j in range(len(pc)):
            if j not in subset_points_idx and calculate_error(plane, pc[j]) < delta:
                C.append(pc[j])
                C_idx.append(j)
        if len(C) > N:
            RUC = subset_points + C
            RUC_idx = subset_points_idx + C_idx
            plane_new = fit_plane_pca(RUC)
            error_new = 0
            for point in RUC:
                error_new += calculate_error(plane_new, point)
            if error_new < error_best:
                error_best = error_new
                plane_best = plane_new
    # Get outlier points
    outlier_points = []
    for i in range(len(pc)):
        if i not in RUC_idx:
            outlier_points.append(pc[i])
    #Show the resulting point cloud
    fig = utils.view_pc([outlier_points])
    utils.view_pc([RUC], fig=fig, color='r')
    normal_vec = plane_best[0:3].reshape(3, 1)
    centroid = (-plane_best[3,0] * normal_vec)
    utils.draw_plane(fig, normal_vec, centroid, (0,1,0,0.3))
    print("Equation for the plane: {}x + {}y + {}z + {} = 0".format(normal_vec[0,0], normal_vec[1,0], normal_vec[2,0], plane_best[3,0]))

    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
