import pfh
import utils
import numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    source_pc = utils.load_pc('pointcloud_data/cloud_icp_source.csv')
    print("...Done loading point cloud. \n")

    # Display the starting data
    utils.view_pc([source_pc])
    plt.title('Starting Point Clouds')
    source_pc_array = utils.convert_pc_to_matrix(source_pc)
    print(source_pc_array.shape)
    PFH = pfh.point_feature_histogram(0.01, source_pc_array)
    nb_idx, nb_point = PFH._getNeighbors(source_pc_array[:,0])
    normal = PFH._getNormal(source_pc_array[:,0])
    print(normal)
