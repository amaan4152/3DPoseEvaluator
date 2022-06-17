from model_analysis import get_poseData2
from pathlib import Path
from data_parser import data_parse2
from truth_analysis import GroundTruth

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
    # DO NOT NEED FRAME_INFO PARAMETER -> NEED TO FIX get_PoseData!
    #   blazepose function requires frame_info to find total # of frames
    GT = GroundTruth(
        ots_file,
        skp_rows=3,
        header_row_list=[0, 2, 3],
        start_frame=sframe,
        end_frame=fframe,
    )
    GT_pos, GT_quat, GT_theta = GT.get_joints(joints)

    gt_data = {"theta": GT_theta, "pos": GT_pos, "quat": GT_quat}
    df_gt_raw = data_parse2("GND_TRUTH", gt_data, joints)

    vid_name = Path(video).name.lower().split(".")[0]
    df_gt_raw.to_csv(f"output/gnd-{vid_name}-raw_data.csv")
    print(f"[{OKGREEN}GOOD{ENDC}]: Successfully extracted and compiled ground truth data")
    return None
    MODEL_pos, MODEL_theta, MODEL_quat = get_poseData2(video, model_type, animation)
    model_data = {"theta": MODEL_theta, "pos": MODEL_pos, "quat": MODEL_quat}
    if animation:
        return None

    df_m_raw = data_parse2(model_type, model_data, joints)  # raw pose data
    return df_m_raw
