from typing import List
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
from matplotlib import test
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
"""

from models.ModelRegistry import ModelRegistry
import json
from pathlib import Path

src = "./src"


def get_poseData2(video: str, model_name: str, animate: bool, joint_names : List):
    # get joint indicies of `model_name`
    MODEL_KEY = model_name.upper()
    with open(f"{src}/models/cfg_joints.json", "r") as cfg_joints_file:
        JOINT_KEYS = json.load(cfg_joints_file)
    print(joint_names)
    JOINTS = [
        JOINT_KEYS[joint_names[0]][MODEL_KEY],
        JOINT_KEYS[joint_names[1]][MODEL_KEY],
        JOINT_KEYS[joint_names[2]][MODEL_KEY],
    ]

    # execute model and parse output data file(s)
    vid_name = Path(video).name
    MR = ModelRegistry()
    MR.exec_model(model_name, video, animate)
    output_files = MR.get_output_files(model_name, vid_name.split(".")[0])
    if not animate and output_files["data"] == []:
        print(
            f"[{MR.FAIL}ERROR{MR.ENDC}]: No data output file(s) detected from algorithm, exiting..."
        )
        exit(1)

    if animate:
        return [None] * 3
    model_data = MR.parse_data(model_name, output_files["data"])

    # compose joint 3D position, joint angle, and joint quaternion data
    model_joint_angle = [None] * len(model_data)
    model_quat = [None] * len(model_data)
    HIP_pos = [None] * len(model_data)
    KNEE_pos = [None] * len(model_data)
    ANKL_pos = [None] * len(model_data)
    for i, data in enumerate(model_data):
        rhip = data[JOINTS[0]]
        rknee = data[JOINTS[1]]
        rankle = data[JOINTS[2]]

        rfemur = rknee - rhip
        rtibia = rankle - rknee
        estim_theta = 180 - joint_angle(rfemur, rtibia)
        estim_axis = np.cross(rfemur, rtibia)
        estim_quat = pq.Quaternion(axis=estim_axis, angle=estim_theta)
        model_joint_angle[i] = estim_theta
        model_quat[i] = estim_quat.elements

        HIP_pos[i] = rhip
        KNEE_pos[i] = rknee
        ANKL_pos[i] = rankle

    model_pos = [HIP_pos, KNEE_pos, ANKL_pos]
    return model_pos, model_joint_angle, model_quat
