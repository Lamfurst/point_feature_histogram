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


    # Display the starting data and print the point with idx as red dot
    utils.view_pc([source_pc, target_pc], color = ['r', 'b'],  marker=['o', 'o'])

    plt.title('Starting Point Clouds')
    source_pc_array = utils.convert_pc_to_matrix(source_pc)
    target_pc_array = utils.convert_pc_to_matrix(target_pc)

    print("...Computing PFH. It takes about 20 sec to Compute\n")
    print("...Please Be Patient : )\n")

    curr_time = time.time()

    # Plant and cup
    source_pc_array, error_track, R, t = pfh.solve_pfh(source_pc_array, target_pc_array, 0.03, 5)

    # Face
    # source_pc_array, error_track = pfh.solve_pfh(source_pc_array, target_pc_array, 0.03, 5)

    # Hokuyo
    # source_pc_array, error_track, R, t = pfh.solve_pfh(source_pc_array, target_pc_array, 2, 5)

    print("Time elapsed: ", time.time() - curr_time, "seconds")
    # Compute total error
    total_error = utils.get_average_error(source_pc_array, target_pc_array)
    source_pc = utils.convert_matrix_to_pc(source_pc_array)
    print("...Done computing PFH. \n")

    print("Error is ", total_error)
    utils.view_pc([source_pc, target_pc], color = ['r', 'b'],  marker=['o', 'o'])
    plt.title('Point Clouds after PFH')
    plt.show()

