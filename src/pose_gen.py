from model_analysis import get_poseData2
import numpy as np
import pandas as pd
from pathlib import Path
from GroundTruth import GroundTruth

OKGREEN = "\033[92m"
ENDC = "\033[0m"


def pose_gen(
    video: str,
    ots_file: str,
    model_type: str,
    animation: bool,
    sframe: int,
    fframe: int,
    joints: dict,
):
    GT = GroundTruth(
        ots_file,
        skp_rows=3,
        header_row_list=[0, 2, 3],
        start_frame=sframe,
        end_frame=fframe,
    )
    GT_pos, GT_quat, GT_theta = GT.get_joints(joints)

    gt_data = {"theta": GT_theta, "pos": GT_pos, "quat": GT_quat}
    df_gt_raw = data_parse("GND_TRUTH", gt_data, joints)

    vid_name = Path(video).name.lower().split(".")[0]
    df_gt_raw.to_csv(f"output/gnd-{vid_name}-raw_data.csv")
    print(
        f"[{OKGREEN}GOOD{ENDC}]: Successfully extracted and compiled ground truth data"
    )

    MODEL_pos, MODEL_theta, _ = get_poseData2(video, model_type, animation, *list(joints.values()))
    model_data = {"theta": MODEL_theta, "pos": MODEL_pos}
    if animation:
        return None

    df_m_raw = data_parse(model_type, model_data, joints)  # raw pose data
    return df_m_raw


def data_parse(model_name: str, model_data: dict, joints: dict) -> pd.DataFrame:
    """
    Note:
    Parse pose estimation data into DataFrames with the following order:
        `[POSITION (XYZ), THETA (joint angle), QUATERNION (orientation), ... ]`

    Parameters
    ----------
    `model_name`:   name of pose estimation algo
    `model_data`:   dictionary of estimated joint positions, joint angles, and orientations
    `joints`:       dictionary of graphical relation of joints
                    `{'vertex' : [joint_a, vertex, joint_b]}`

    Return
    ------
    `df`:           DataFrame object of data labeled
    """
    df = None
    for v, joint_list in joints.items():
        # column names
        tuple_names = [
            (f"{model_name}", f"{v}", "Joint Angle", "Theta"),
            *[
                (f"{model_name}", f"{j}", f"{k}", f"{e}")
                for j in joint_list
                for k, e in zip(
                    ["Position"] * 3,
                    ["X", "Y", "Z"],
                )
            ],
        ]
        cols = pd.MultiIndex.from_tuples(tuple_names)
        theta_data = np.array(model_data["theta"]).squeeze().reshape(-1, 1)
        pos_data = np.hstack(model_data["pos"])
        data = np.hstack([theta_data, pos_data])

        df = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)

    return df
