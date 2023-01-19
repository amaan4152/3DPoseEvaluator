import numpy as np
import pandas as pd

from GroundTruth import GroundTruth
from model_analysis import get_pose_data

DATA_DIR = "/root/tmp"
OUTPUT_DIR = "/root/output"

OKGREEN = "\033[92m"
ENDC = "\033[0m"


def pose_gen(
    model_type: str,
    joints: dict,
    kinematic_chain: str,
) -> pd.DataFrame:
    # ignoring quaternion data for now
    MODEL_pos, MODEL_theta, _ = get_pose_data(model=model_type, joints=joints)
    model_data = {"theta": MODEL_theta, "pos": MODEL_pos}

    df_m_raw = data_parse(
        model_name=model_type, 
        model_data=model_data, 
        kinematic_chain=kinematic_chain, 
        joints=joints
    )  # raw pose data
    return df_m_raw


def data_parse(
    model_name: str, 
    model_data: dict, 
    kinematic_chain: str, 
    joints: dict
) -> pd.DataFrame:
    """
    Note:
    Parse pose estimation data into DataFrames with the following order:
        `[POSITION (XYZ), THETA (joint angle), QUATERNION (orientation), ... ]`

    Parameters
    ----------
    `model_name`:       name of pose estimation algo
    `model_data`:       dictionary of estimated joint positions, joint angles, and orientations
    `kinematic_chain`:  valid kinematic chain label from 'joints_kinematic_table.yml'
    `joints`:           kinematic chain of joints

    Return
    ------
    `df`:               DataFrame object of data labeled
    """
    df = None
    joint_labels = list(joints.keys())

    # column names
    tuple_names = [
    (f"{model_name}", f"{kinematic_chain}", "Joint Angle", "Theta"),
        *[
            (f"{model_name}", f"{j}", f"{k}", f"{e}")
            for j in joint_labels
            for k, e in zip(
                ["Position"] * len(joint_labels),
                ["X", "Y", "Z"],
            )
        ]
    ]
    cols = pd.MultiIndex.from_tuples(tuple_names)

    # compile data as dataframe
    theta_data = np.array(model_data["theta"]).squeeze().reshape(-1, 1)
    pos_data = np.hstack(model_data["pos"])
    data = np.hstack([theta_data, pos_data])

    df = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)

    return df
