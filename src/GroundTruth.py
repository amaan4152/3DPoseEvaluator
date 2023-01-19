import json
import pandas as pd
from numba import jit
import numpy as np
import pyquaternion as pq
import yaml

WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"

VID_FPS = 60
OTS_FPS = 120


class GroundTruth(object):
    VID_FPS = 60
    OTS_FPS = 120
    src = "./models"

    def __init__(
        self,
        file: str,
        skp_rows: int,
        header_row_list: list,
        start_frame: int,
        end_frame: int,
        joints: dict,
    ):
        # configure frame ID range for ground truth
        self.sframe = start_frame
        self.fframe = end_frame
        self.sframe *= int(self.OTS_FPS / self.VID_FPS)
        self.fframe *= int(self.OTS_FPS / self.VID_FPS)
        self.joints = joints

        # get ground truth and joints config
        self.gnd_df = (
            pd.read_csv(
                file, skiprows=skp_rows, header=header_row_list, dtype="unicode"
            )
            .astype(float)
            .iloc[self.sframe : self.fframe, :]
        )
        self.cols = np.array(list(map(list, self.gnd_df.columns)))

        # get root origin joint
        self.__get_origin()

    def get_joints(self):
        joint_labels = list(self.joints.keys())

        # get xyz positional data of each joint
        num_joints = len(joint_labels)
        gnd_pos = [None] * num_joints
        bone_rots = [None] * 2
        for i, j in enumerate(joint_labels):
            """
            Each joint has 1 bone, and could have `n` > 0 number of markers
            """
            gnd_truth_joint = self.joints[j]["truth"]
            bones = gnd_truth_joint["bones"]
            markers = gnd_truth_joint["markers"]

            # get columns that contain gnd-truth joint names for `joint`
            bone_col = self.__substrColMatcher(bones, self.cols)[0]
            marker_cols = self.__substrColMatcher(markers, self.cols)
            if i >= 1:
                bone_rots[i - 1] = self.gnd_df[bone_col]["Rotation"]
            col_names = bone_col if marker_cols == [] else marker_cols

            # get root joint (between hips) and set as origin of system
            gnd_pos[i] = self.__mean_xyz(col_names) - self.root_origin

        # compute relative quaternion and angle (degrees)
        df_bone_rots = pd.concat(bone_rots, axis=1).values
        gnd_quat = np.apply_along_axis(
            self.__compute_del_quat, axis=1, arr=df_bone_rots
        ).reshape(-1, 1)
        gnd_theta = np.apply_along_axis(
            self.__compute_theta, axis=1, arr=df_bone_rots
        ).reshape(-1, 1)

        return (np.array(gnd_pos), gnd_quat, gnd_theta)

    def __get_origin(self) -> None:
        with open("joints_kinematic_table.yml", "r") as stream:
            joints_cfg = yaml.load(stream=stream, Loader=yaml.SafeLoader)
        left_hip = joints_cfg["left-leg"]["left-hip"]["truth"]
        right_hip = joints_cfg["right-leg"]["right-hip"]["truth"]

        left_hip = (
            left_hip["bones"] if left_hip["markers"] == [] else left_hip["markers"]
        )
        right_hip = (
            right_hip["bones"] if right_hip["markers"] == [] else right_hip["markers"]
        )

        left_hip_cols = self.__substrColMatcher(left_hip, self.cols)
        right_hip_cols = self.__substrColMatcher(right_hip, self.cols)

        left_hip_xyz = self.__mean_xyz(left_hip_cols)
        right_hip_xyz = self.__mean_xyz(right_hip_cols)

        self.torso_diam = np.linalg.norm(left_hip_xyz - right_hip_xyz)
        self.root_origin = (left_hip_xyz + right_hip_xyz) / 2

    def __mean_xyz(self, cols: list) -> np.ndarray:
        df_x = np.mean([self.gnd_df[c]["Position"]["X"] for c in cols], axis=0)
        df_y = np.mean([self.gnd_df[c]["Position"]["Y"] for c in cols], axis=0)
        df_z = np.mean([self.gnd_df[c]["Position"]["Z"] for c in cols], axis=0)
        return np.vstack((df_x, df_y, df_z)).T

    def __compute_del_quat(self, r):
        q1 = pq.Quaternion(r[3], r[0], r[1], r[2])  # w,x,y,z
        q2 = pq.Quaternion(r[7], r[4], r[5], r[6])  # w,x,y,z

        del_q = q1.conjugate * q2
        return del_q

    def __compute_theta(self, r):
        del_q = self.__compute_del_quat(r)

        theta = 180 - np.abs(del_q.degrees)
        return theta

    def __substrColMatcher(self, joints: list, MICols: np.ndarray) -> list:
        return list(
            set(filter(lambda s, j=joints: any(map(s.__contains__, j)), MICols.T[0]))
        )
