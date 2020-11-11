import json
import numpy as np


def prepare_ground_truth(filepath, t_inv_matrix):
    with open(filepath, 'r') as f:
        with open('data/transformed_ground_truth.txt', 'w') as out:
            d = json.load(f)
            for data in d["Camera"]:
                data = np.array(data)
                transformed = (t_inv_matrix @ data)[:3]
                flattened = transformed.ravel()
                out.write(" ".join(map(str, flattened.tolist())))
                out.write("\n")


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
