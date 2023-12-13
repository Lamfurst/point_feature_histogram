import numpy as np

def get_rotation_dis(R1, R2, MTD):
    if MTD == 'trace':
        R_rel = np.dot(np.linalg.inv(R1), R2)
        # Calculate the angular distance
        error = np.arccos((np.trace(R_rel) - 1) / 2)
        error = np.rad2deg(error)
    
    else:
        R_rel = np.dot(np.linalg.inv(R1), R2)
        error = np.linalg.norm(R_rel, 'fro')
    return error

def rot_dis(R1, R2):
    R_rel = np.dot(np.linalg.inv(R1), R2)
    # Calculate the angular distance
    angle = np.arccos((np.trace(R_rel) - 1) / 2)
    return angle

T1_gt = np.array([[0.988301,	0.152466,	0.003715,	0.421117],
              [	-0.152287,	0.987877,	-0.030139,	-0.049903],
              [	-0.008265,	0.029221,	0.999538,	0.066915],
              [	0,	0,	0,	1]])
R1_gt = T1_gt[:3,:3]
t1_gt = T1_gt[:3,3].reshape(3,1)

# Rotation matrix R
R1 = np.array([
    [0.9837643, -0.16755519, -0.06471395],
    [0.16769042, 0.9858393, -0.00337514],
    [0.06436273, -0.00753167, 0.99789815]
])

# Translation vector t
t1 = np.array([
    [-0.18888042],
    [-0.11366837],
    [-0.01329967]
])

# R and t are now numpy arrays

T2_gt = np.array([[0.978404,	0.206679,	0.003096,	0.877821],
              [	-0.206407,	0.977697,	-0.0388,	-0.110071],
              [	-0.011046,	0.037322,	0.999242,	0.138536],
              [	0,	0,	0,	1]])
R2_gt = T2_gt[:3,:3]
t2_gt = T2_gt[:3,3].reshape(3,1)
# Rotation matrix R
R2 = np.array([
    [0.96905821, -0.19350508, -0.15067827],
    [0.20321, 0.97840243, 0.03740524],
    [0.14011379, -0.06866719, 0.98787494]
])

# Translation vector t
t2 = np.array([
    [-0.49088185],
    [-0.55111829],
    [-0.35819249]
])

print(get_rotation_dis(R1_gt, R1, "trace"))
print(get_rotation_dis(R2_gt, R2, "trace"))
print(np.linalg.norm(t1_gt - t1))
print(np.linalg.norm(t2_gt - t2))