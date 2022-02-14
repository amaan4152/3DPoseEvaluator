import joblib as jb

from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB
import numpy as np
import math
import mediapipe as mp
import os
import pyquaternion as pq
from shutil import copy
import subprocess as sp
from tqdm import tqdm

poseModels = {"VIBE", "GAST", "BLAZEPOSE"}
src_dir = "./src"

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


def get_poseData(model, video, time_stat):
    if model in poseModels:
        if model == "VIBE":

            dir_name = video.split(".")[0]
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
                vproc = sp.Popen([f"{src_dir}/exec_models.sh", "VIBE", video], stdout=sp.PIPE)
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
                copy(video, f"{src_dir}/GAST-Net-3DPoseEstimation/data/video/")
                gproc = sp.Popen([f"{src_dir}/exec_models.sh", "GAST", video], stdout=sp.PIPE)
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
            cap = VideoCapture(video)

            total_frames = int(time_stat[1] * 25)
            pbar = tqdm(total=total_frames, ncols=100)
            pbar.set_description("Frame # Progress: ")
            with mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as _:
                while cap.isOpened():
                    s, img = cap.read()
                    if not s:
                        print("[MEDIAPIPE]: Empty frame detected...")
                        pbar.update(1)
                        continue

                    img.flags.writeable = False
                    img = cvtColor(img, COLOR_BGR2RGB)
                    pbar.update(1)

            pbar.close()
            exit(1)

    else:
        print("<ERROR>: 3D Pose Estimation model not supported..")
