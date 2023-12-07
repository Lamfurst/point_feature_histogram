import pfh
import utils
import numpy
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    source_pc = utils.load_pc('pointcloud_data/cloud_icp_source.csv')
    print("...Done loading source point cloud. \n")
    target_pc = utils.load_pc('pointcloud_data/cloud_icp_target0.csv')
    print("...Done loading target point cloud. \n")


    idx = 200 # idx 200 on the handle
    # Display the starting data and print the point with idx as red dot
    utils.view_pc([source_pc, [source_pc[idx]]], color = ['r', 'b'],  marker=['o', 'o'])

    plt.title('Starting Point Clouds')
    plt.show()
    source_pc_array = utils.convert_pc_to_matrix(source_pc)
    target_pc_array = utils.convert_pc_to_matrix(target_pc)
    

    # Time the PFH computation
    start_time = time.time()
    PFH_source = pfh.point_feature_histogram(0.03, source_pc_array, 5)
    pfh_list = PFH_source.computePFHSignatures()
    end_time = time.time()
    print("Time elapsed: ", end_time - start_time, "seconds")

    
    pfh_sample = pfh_list[idx]
    # Use plot to plot the histogram
    plt.bar(range(len(pfh_sample)), pfh_sample)
    plt.show()

    # start_time = time.time()
    # PFH_target = pfh.point_feature_histogram(0.03, target_pc_array, 5)
    # pfh_list = PFH_target.computePFHSignatures()
    # end_time = time.time()
    # print("Time elapsed: ", end_time - start_time, "seconds")

    # pfh_sample_target = pfh_list[idx]
    # # Use plot to plot the histogram
    # plt.bar(range(len(pfh_sample_target)), pfh_sample_target)

    # plt.show()

