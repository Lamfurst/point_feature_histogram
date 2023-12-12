import pfh
import utils
import numpy
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # source_pc = utils.load_pc('pointcloud_data/cloud_icp_source.csv')
    # source_pc = utils.load_pc_np('pointcloud_data/face_source.npy')
    # source_pc = utils.load_pc('pointcloud_data/Hokuyo_0_downsampled.csv')
    source_pc = utils.load_pc_np('pointcloud_data/plant_source.npy')
    print("...Done loading source point cloud. \n")
    # target_pc = utils.load_pc('pointcloud_data/cloud_icp_target3.csv')
    # target_pc = utils.load_pc_np('pointcloud_data/face_target.npy')
    # target_pc = utils.load_pc('pointcloud_data/Hokuyo_1_downsampled.csv')
    target_pc = utils.load_pc_np('pointcloud_data/plant_target3.npy')
    print("...Done loading target point cloud. \n")


    idx = 200 # idx 200 on the handle
    # Display the starting data and print the point with idx as red dot
    # utils.view_pc([source_pc, [source_pc[idx]]], color = ['r', 'b'],  marker=['o', 'o'])
    utils.view_pc([source_pc, target_pc], color = ['r', 'b'],  marker=['o', 'o'])

    plt.title('Starting Point Clouds')
    plt.show()
    source_pc_array = utils.convert_pc_to_matrix(source_pc)
    target_pc_array = utils.convert_pc_to_matrix(target_pc)

    source_pc_array, error_track = pfh.solve_pfh(source_pc_array, target_pc_array, 0.03, 5)
    # source_pc_array, error_track = pfh.solve_pfh(source_pc_array, target_pc_array, 2, 5)
    source_pc = utils.convert_matrix_to_pc(source_pc_array)

    print("...Done computing PFH. \n")
    print("Error is ", error_track)
    utils.view_pc([source_pc, target_pc], color = ['r', 'b'],  marker=['o', 'o'])
    plt.title('Point Clouds after PFH')
    plt.show()

    

    # # Time the PFH computation
    # start_time = time.time()
    # PFH_source = pfh.point_feature_histogram(0.03, source_pc_array, 5)
    # pfh_list = PFH_source.computePFHSignatures()
    # end_time = time.time()
    # print("Time elapsed: ", end_time - start_time, "seconds")

    
    # pfh_sample = pfh_list[idx]
    # # Use plot to plot the histogram
    # plt.bar(range(len(pfh_sample)), pfh_sample)
    # plt.show()

    # start_time = time.time()
    # PFH_target = pfh.point_feature_histogram(0.03, target_pc_array, 5)
    # pfh_list = PFH_target.computePFHSignatures()
    # end_time = time.time()
    # print("Time elapsed: ", end_time - start_time, "seconds")

    # pfh_sample_target = pfh_list[idx]
    # # Use plot to plot the histogram
    # plt.bar(range(len(pfh_sample_target)), pfh_sample_target)

    # plt.show()

