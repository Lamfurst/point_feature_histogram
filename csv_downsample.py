import open3d as o3d
import numpy as np
import pandas as pd

# Load your CSV file
csv_file = 'pointcloud_data/Hokuyo_2.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file, usecols=[1, 2, 3])  # Assuming columns 2, 3, 4 are X, Y, Z

# Convert the DataFrame to a NumPy array
points = data.to_numpy()

# Create a PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# View the point cloud
o3d.visualization.draw_geometries([pcd])

# Downsample the point cloud
voxel_size = 0.5  # Set the voxel size
down_pcd = pcd.voxel_down_sample(voxel_size)

# View the downsampled point cloud
o3d.visualization.draw_geometries([down_pcd])

# Convert downsampled PointCloud to NumPy array
down_points = np.asarray(down_pcd.points)

# Create a DataFrame from the downsampled points
downsampled_data = pd.DataFrame(down_points, columns=['x', 'y', 'z'])

# Save the downsampled points to a new CSV file
downsampled_csv = csv_file.split('/')[0] + '/' + csv_file.split('/')[1].split('.')[0] + '_downsampled.csv'  # Replace with your new CSV file path
downsampled_data.to_csv(downsampled_csv, index=False, header=False)
