import numpy as np
import math
import pyquaternion as pq

DATA_DIR = "/root/tmp"

def joint_angle(bone1, bone2):
    vunit = lambda v : v / np.linalg.norm(v)
    ubone1 = vunit(bone1)
    ubone2 = vunit(bone2)
    return np.arccos(np.dot(ubone1, ubone2)) * (180.0 / math.pi)


def get_pose_data(model: str, joints : dict):
    data = np.load(f"{DATA_DIR}/{model}/data.npy")
    keypoints = [joints[j][model] for j in joints.keys()]

    # compose joint 3D position, joint angle, and joint quaternion data
    model_joint_angle = [None] * len(data)
    model_quat = [None] * len(data)
    joint_start = [None] * len(data)
    joint_vertex = [None] * len(data)
    joint_end = [None] * len(data)
    for i, data in enumerate(data):
        j_start = data[keypoints[0]]
        j_vertex = data[keypoints[1]]
        j_end = data[keypoints[2]]

        bone_1 = j_vertex - j_start
        bone_2 = j_end - j_vertex
        theta_hat = 180 - joint_angle(bone1=bone_1, bone2=bone_2)
        v_axis_hat = np.cross(bone_1, bone_2)
        quat_hat = pq.Quaternion(axis=v_axis_hat, angle=theta_hat)
        model_joint_angle[i] = theta_hat
        model_quat[i] = quat_hat.elements

        joint_start[i] = j_start
        joint_vertex[i] = j_vertex
        joint_end[i] = j_end

    model_pos = [joint_start, joint_vertex, joint_end]
    return model_pos, model_joint_angle, model_quat
