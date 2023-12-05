#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target

    def getTransform(Cp, Cq):
        # Cp np.array(3,n)
        # Cq np.array(3,n)
        p_bar = numpy.mean(Cp, axis=1).reshape(3, 1)
        p_centered = Cp - p_bar
        q_bar = numpy.mean(Cq, axis=1).reshape(3, 1)
        q_centered = Cq - q_bar
        S = p_centered @ q_centered.T
        U, _, Vt = numpy.linalg.svd(S)
        detVUT = numpy.linalg.det(Vt.T @ U.T)
        R = Vt.T @ numpy.diag([1, 1, detVUT]) @ U.T
        t = q_bar - R @ p_bar

        return R, t

    def getCorrespondences(pc_source_array, pc_target_array):
        C = []
        for i in range(pc_source_array.shape[1]):
            curr_dis = numpy.linalg.norm(pc_target_array - pc_source_array[:,i].reshape(3,1), axis = 0)
            min_idx = numpy.argmin(curr_dis)
            C.append([i, min_idx])
        return C

    def tranformPCArray(pc_array, R, t):
        pc_array_transformed = R @ pc_array + t
        return pc_array_transformed
    

    def icp(pc_source, pc_target, max_iter = 1000, threshold = 0.001):
        error_his = []
        pc_source_array = numpy.hstack(pc_source)
        pc_target_array = numpy.hstack(pc_target)

        for _ in range(max_iter):
            C = getCorrespondences(pc_source_array, pc_target_array)
            Cp = []
            Cq = []
            for i in range(len(C)):
                p = pc_source_array[:,C[i][0]]
                q = pc_target_array[:,C[i][1]]
                Cp.append(p)
                Cq.append(q)
            Cp = numpy.hstack(Cp)
            Cq = numpy.hstack(Cq)
            R, t = getTransform(Cp, Cq)

            Cp_transformed = tranformPCArray(Cp, R, t)
            error = numpy.linalg.norm(Cq - Cp_transformed)**2
            error_his.append(error)
            if error < threshold:
                break

            pc_source_array = tranformPCArray(pc_source_array, R, t)
        # Transform pc_source_array to pc_source
        pc_source = []
        for i in range(pc_source_array.shape[1]):
            pc_source.append(pc_source_array[:,i].reshape(3, 1))
        return pc_source, error_his

    pc_source, error_his = icp(pc_source, pc_target)
    # Plot Error_his vs Iteration on another figure

    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15])

    plt.figure()
    plt.plot(error_his)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error vs Iteration')
    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
