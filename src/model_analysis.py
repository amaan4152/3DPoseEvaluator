import joblib as jb
import json
import cv2
from cv2 import (
    VideoCapture,
    cvtColor,
    COLOR_BGR2RGB,
    imshow,
    VideoWriter,
    VideoWriter_fourcc,
)
import numpy as np
import math
import mediapipe as mp
import os
import pyquaternion as pq
from shutil import copy
import subprocess as sp
from tqdm import tqdm

poseModels = {"VIBE", "GAST", "BLAZEPOSE"}
src_dir = "/root"


def get_vec(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return p2 - p1


def vunit(vec):
    return vec / np.linalg.norm(vec)


def joint_angle(bone1, bone2):
    ubone1 = vunit(bone1)
    ubone2 = vunit(bone2)
    return np.arccos(np.dot(ubone1, ubone2)) * (180.0 / math.pi)


def get_dataFile(proc, type):
    while True:
        out = proc.stdout.readline().decode("utf-8")
        if proc.poll() is not None:  # child process has terminated
            break
        if out:
            if type in out:
                dataFile = out.strip()
            print(out.strip())

    return dataFile


"""
FEATURES TO ADD:
----------------
- choose what joints to extract raw pose data from
    - must have same format and share same joints
"""

"""
from models.gast.parse_gast import parse_gast
from models.vibe.parse_vibe import parse_vibe

def get_pose(model, video, sframe, fframe, st_pose_file):
    # joints will be preconfigured for now
    joints = ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]

    f = open('/src/cfg_joints.json')

    pose_file = st_pose_file["file"]
    if not pose_file: 
        vproc = sp.Popen(
            [f"./src/{model}.sh", video], stdout=sp.PIPE
        )
        pose_file = get_dataFile(vproc, st_pose_file["type"])

    joint_angle = {}
    quat = {}
    HIP_pos = {}
    KNEE_pos = {}
    ANKL_pos = {}
    joint_map = json.load(f)
    for frame_num in range(0, len()):
"""

from models.ModelRegistry import ModelRegistry
import json
src = "./src"
def get_poseData2(video : str, model_name : str, animate : bool):
    # get joint indicies of `model_name`
    MODEL_KEY = model_name.upper()
    with open(f"{src}/models/cfg_joints.json", "r") as cfg_joints_file:
        JOINT_KEYS = json.load(cfg_joints_file)
    RIGHT_LEG_JOINTS = [
        JOINT_KEYS['RIGHT_HIP'][MODEL_KEY],
        JOINT_KEYS['RIGHT_KNEE'][MODEL_KEY],
        JOINT_KEYS['RIGHT_ANKLE'][MODEL_KEY],
    ]

    # execute model and parse output data file(s)
    MR = ModelRegistry()
    output_files = MR.exec_model(model_name, video, animate)
    model_data = MR.parse_data(model_name, output_files['data'])

    # compose joint 3D position, joint angle, and joint quaternion data
    model_joint_angle = [None] * len(model_data)
    model_quat = [None] * len(model_data)
    HIP_pos = [None] * len(model_data)
    KNEE_pos = [None] * len(model_data)
    ANKL_pos = [None] * len(model_data)
    for frame_num in range(0, len(model_data)):
        rhip = model_data[frame_num][RIGHT_LEG_JOINTS[0]]
        rknee = model_data[frame_num][RIGHT_LEG_JOINTS[1]]
        rankle = model_data[frame_num][RIGHT_LEG_JOINTS[2]]

        rfemur = rknee - rhip
        rtibia = rankle - rknee
        estim_theta = 180 - joint_angle(rfemur, rtibia)
        estim_axis = np.cross(rfemur, rtibia)
        estim_quat = pq.Quaternion(axis=estim_axis, angle=estim_theta)
        model_joint_angle[frame_num] = estim_theta
        model_quat[frame_num] = (
            estim_quat.elements[0],
            estim_quat.elements[1],
            estim_quat.elements[2],
            estim_quat.elements[3],
        )

        HIP_pos[frame_num] = rhip
        KNEE_pos[frame_num] = rknee
        ANKL_pos[frame_num] = rankle

    model_pos = [HIP_pos, KNEE_pos, ANKL_pos]
    return model_pos, model_joint_angle, model_quat


def get_poseData(video, model, sframe, fframe):
    time_stat = (sframe, fframe)
    if model in poseModels:
        if model == "VIBE":

            dir_name = video.split("/")[-1].split(".")[0]

            print(f"{src_dir}/VIBE/output/{dir_name}/")
            try:
                wdir_list = os.listdir(f"{src_dir}/VIBE/output/" + dir_name + "/")
                wdir_list = [file for file in wdir_list if ".pkl" in file]
                print(*wdir_list, sep="\n")

            except:
                print("New video file detected, proceed to execute VIBE...")

            b_vibe_gen = input(
                "Execute VIBE model? [y/...] If not, then choose which VIBE data file to use: "
            )
            if b_vibe_gen in ("Y", "y"):
                vproc = sp.Popen(
                    [f"./src/exec_models.sh", "VIBE", video], stdout=sp.PIPE
                )
                pkl_file = get_dataFile(vproc, ".pkl")

            else:
                pkl_file = os.path.abspath(
                    f"{src_dir}/VIBE/output/{dir_name}/{b_vibe_gen}"
                )

            # joints 12, 13, 14 are the left hip, left knee, and left ankle respectively
            pkl_data = jb.load(pkl_file)
            # pkl_data = 1000*pkl_data[1]['joints3d']
            VIBE_data = {
                str(k): [v[9] * 1000, v[10] * 1000, v[11] * 1000]
                for k, v in zip(
                    range(0, len(pkl_data[1]["joints3d"])), pkl_data[1]["joints3d"]
                )
            }

            VIBE_joint_angle = {}
            VIBE_quat = {}
            HIP_pos = {}
            KNEE_pos = {}
            ANKL_pos = {}

            for frame_num in range(0, len(VIBE_data)):
                # VIBE_time_data.append(frame_num/vibe_fps)
                rhip = VIBE_data[str(frame_num)][0]
                rknee = VIBE_data[str(frame_num)][1]
                rankle = VIBE_data[str(frame_num)][2]
                rfemur = rknee - rhip
                rtibia = rankle - rknee
                estim_theta = 180 - joint_angle(rfemur, rtibia)
                estim_axis = np.cross(rfemur, rtibia)
                estim_quat = pq.Quaternion(axis=estim_axis, angle=estim_theta)
                VIBE_joint_angle[str(frame_num)] = estim_theta
                VIBE_quat[str(frame_num)] = (
                    estim_quat.elements[0],
                    estim_quat.elements[1],
                    estim_quat.elements[2],
                    estim_quat.elements[3],
                )
                HIP_pos[str(frame_num)] = rhip
                KNEE_pos[str(frame_num)] = rknee
                ANKL_pos[str(frame_num)] = rankle

            VIBE_pos = [HIP_pos, KNEE_pos, ANKL_pos]
            return VIBE_pos, VIBE_joint_angle, VIBE_quat

        elif model == "GAST":
            # fix GAST dir path resolution!
            wdir_list = os.listdir(f"{src_dir}/GAST-Net-3DPoseEstimation/output/")
            wdir_list = [file for file in wdir_list if ".npz" in file]
            print(*wdir_list, sep="\n")
            b_gast_gen = input(
                "Execute GAST-NET model? [y/...] If not, then choose which GAST joint data file to use: "
            )
            if b_gast_gen in ("Y", "y"):
                copy(
                    video,
                    f"{src_dir}/GAST-Net-3DPoseEstimation/data/video/",
                )
                gproc = sp.Popen(
                    ["./src/exec_models.sh", "GAST", video.split("/")[-1]],
                    stdout=sp.PIPE,
                )
                npz_file = get_dataFile(gproc, ".npz")
            else:
                npz_file = os.path.abspath(
                    f"{src_dir}/GAST-Net-3DPoseEstimation/output/{b_gast_gen}"
                )

            # WARNING: 3D coordinate system for GAST revolves around pelvis (keypoint #0) as origin
            # 3D array: dim_03->list of frame data of vid seq; dim_02 -> indexed by # of frames; dim_01 -> indexed by keypoint #
            GAST_data = np.load(npz_file)["reconstruction"]
            GAST_data = np.array(GAST_data[0]) * 1000  # convert data to millimeters

            GAST_joint_angle = {}
            GAST_quat = {}
            HIP_pos = {}
            KNEE_pos = {}
            ANKL_pos = {}

            for frame_num in range(0, len(GAST_data)):
                # GAST_time_data.append(frame_num/GAST_fps)
                rhip = GAST_data[frame_num][1]  # 4 => lhip
                rknee = GAST_data[frame_num][2]  # 5 => lknee
                rankle = GAST_data[frame_num][3]  # 6 => lankle
                rfemur = rknee - rhip
                rtibia = rankle - rknee
                estim_theta = 180 - joint_angle(rfemur, rtibia)
                estim_axis = np.cross(rfemur, rtibia)
                estim_quat = pq.Quaternion(axis=estim_axis, angle=estim_theta)
                GAST_joint_angle[str(frame_num)] = estim_theta
                GAST_quat[str(frame_num)] = (
                    estim_quat.elements[0],
                    estim_quat.elements[1],
                    estim_quat.elements[2],
                    estim_quat.elements[3],
                )
                HIP_pos[str(frame_num)] = rhip
                KNEE_pos[str(frame_num)] = rknee
                ANKL_pos[str(frame_num)] = rankle

            GAST_pos = [HIP_pos, KNEE_pos, ANKL_pos]
            return GAST_pos, GAST_joint_angle, GAST_quat

        elif model == "BLAZEPOSE":
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            fname = f"video-lib/nist_peg_test_sub01.mp4"
            print(fname)
            cap = VideoCapture(fname)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            o_cap = VideoWriter(
                "TrialCamA_out.mp4", VideoWriter_fourcc(*"mp4v"), 15, (h, w)
            )

            total_frames = int(time_stat[1] * 25)
            pbar = tqdm(total=total_frames, ncols=100)
            pbar.set_description("Frame # Progress: ")
            frame_num = 1

            BP_joint_angle = {}
            BP_quat = {}
            HIP_pos = {}
            KNEE_pos = {}
            ANKL_pos = {}
            with mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as pose:
                while cap.isOpened():
                    s, img = cap.read()
                    if not s:
                        print("=== BROKEN FRAME ===")
                        break

                    if frame_num < 1000:
                        frame_num += 1
                        pbar.update(1)
                        continue
                    elif frame_num > 1300:
                        break

                    img.flags.writeable = False
                    img = cvtColor(img, COLOR_BGR2RGB)
                    results = pose.process(img)
                    rhip_x = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_HIP
                    ].x
                    rhip_y = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_HIP
                    ].y
                    rhip_z = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_HIP
                    ].z
                    rknee_x = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_KNEE
                    ].x
                    rknee_y = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_KNEE
                    ].y
                    rknee_z = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_KNEE
                    ].z
                    rankle_x = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_ANKLE
                    ].x
                    rankle_y = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_ANKLE
                    ].y
                    rankle_z = results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_ANKLE
                    ].z
                    rhip = np.array([rhip_x, rhip_y, rhip_z])
                    rknee = np.array([rknee_x, rknee_y, rknee_z])
                    rankle = np.array([rankle_x, rankle_y, rankle_z])
                    rfemur = rknee - rhip
                    rtibia = rankle - rknee
                    estim_theta = 180 - joint_angle(rfemur, rtibia)
                    estim_axis = np.cross(rfemur, rtibia)
                    estim_quat = pq.Quaternion(axis=estim_axis, angle=estim_theta)
                    BP_joint_angle[str(frame_num)] = estim_theta
                    BP_quat[str(frame_num)] = (
                        estim_quat.elements[0],
                        estim_quat.elements[1],
                        estim_quat.elements[2],
                        estim_quat.elements[3],
                    )
                    HIP_pos[str(frame_num)] = rhip
                    KNEE_pos[str(frame_num)] = rknee
                    ANKL_pos[str(frame_num)] = rankle

                    # Draw the pose annotation on the image.
                    img.flags.writeable = True
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    # Flip the image horizontally for a selfie-view display.
                    img = cv2.flip(img, 1)
                    cv2.imshow("MediaPipe Pose", img)
                    o_cap.write(img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                    pbar.update(1)
                    frame_num += 1

            pbar.close()
            cap.release()
            o_cap.release()
            cv2.destroyAllWindows()
            BP_pos = [HIP_pos, KNEE_pos, ANKL_pos]
            return BP_pos, BP_joint_angle, BP_quat

    else:
        print("<ERROR>: 3D Pose Estimation model not supported..")
