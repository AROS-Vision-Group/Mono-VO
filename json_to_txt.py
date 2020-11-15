import json
import numpy as np
import utils
import compute


def prepare_ground_truth(filepath, t_inv_matrix):
    with open(filepath, 'r') as f:
        with open('data/transformed_ground_truth_vol2.txt', 'w') as out:
            d = json.load(f)
            for data in d["Camera"]:
                data = np.array(data)
                transformed = (t_inv_matrix @ data)[:3]
                flattened = transformed.ravel()

                # Ground truth data has opposite direction for positive values than our model, for y and z axis
                # Therefore we decompose the ground truth rotation matrix and flips y and z axis before writing to file
                r11, r12, r13 = float(flattened[0]), float(flattened[1]), float(flattened[2])
                r21, r22, r23 = float(flattened[4]), float(flattened[5]), float(flattened[6])
                r31, r32, r33 = float(flattened[8]), float(flattened[9]), float(flattened[10])
                R = np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape((3, 3))
                aligned_R_flatten = flip_y_and_z_axis(R).ravel()

                flattened[0], flattened[1], flattened[2] = aligned_R_flatten[0], aligned_R_flatten[1], aligned_R_flatten[2]
                flattened[4], flattened[5], flattened[6] = aligned_R_flatten[3], aligned_R_flatten[4], aligned_R_flatten[5]
                flattened[8], flattened[9], flattened[10] = aligned_R_flatten[6], aligned_R_flatten[7], aligned_R_flatten[8]

                # We must also flip the y and z point locations
                flattened[7] *= -1
                flattened[11] *= -1

                out.write(" ".join(map(str, flattened.tolist())))
                out.write("\n")


def flip_y_and_z_axis(R):
    theta = compute.rotation_matrix_to_euler_angles(R)
    theta[1] = - theta[1]
    theta[2] = - theta[2]

    return compute.euler_angles_to_rotation_matrix(theta)


if __name__ == '__main__':
    filepath = './data/blender_underwater_env_ground_truth_poses.json'
    origin_transformation_mtx = np.array([
        [
            -0.40673887171736184,
            0.9135444653834002,
            1.2021145039395212e-06,
            0.22304523604587842
        ],
        [
            -0.14605695240041458,
            -0.0650304707959155,
            0.9871364670214371,
            -0.07497521108390426
        ],
        [
            0.9017931341996291,
            0.4015065972501812,
            0.159879940814945,
            -5.6632791527651305
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])

    prepare_ground_truth(filepath, origin_transformation_mtx)
