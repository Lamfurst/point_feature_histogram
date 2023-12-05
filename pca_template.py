#!/usr/bin/env python
import utils
import numpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    fig = utils.view_pc([pc])

    #Rotate the points to align with the XY plane
    n = len(pc)
    pc_array = numpy.hstack(pc)
    # compute the centroid
    centroid = numpy.mean(pc_array, axis=1).reshape(3, 1)
    # subtract the centroid from the points
    pc_array_centered = pc_array - centroid
    # compute the covariance matrix
    Q = pc_array_centered.dot(pc_array_centered.T) / (n - 1)
    # SVD
    _, S, Vt = numpy.linalg.svd(Q)
    normal_vec = Vt.T[:, 2] #3x1
    pc_rotated = Vt @ pc_array_centered
    pc_rotated_list = []

    for i in range(n):
        pc_rotated_list.append(pc_rotated[:,i].reshape(3, 1))
    utils.view_pc([pc_rotated_list])
    print("Vt:", Vt)

    #Rotate the points to align with the XY plane AND eliminate the noise
    threshold = 1e-3
    s = S**2
    s_idx = s < threshold
    Vs = Vt.copy()
    Vs[s_idx] = 0
    pc_rotated_reduce_noise = Vs @ pc_array_centered
    pc_rotated_reduce_noise_list = []

    for i in range(n):
        pc_rotated_reduce_noise_list.append(pc_rotated_reduce_noise[:,i].reshape(3, 1))
    utils.view_pc([pc_rotated_reduce_noise_list])
    print("Vs:", Vs)

    utils.view_pc([pc])
    utils.draw_plane(fig, normal_vec, centroid, (0,1,0,0.3))
    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
