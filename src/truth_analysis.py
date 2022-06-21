import json
import pandas as pd
from numba import jit
import numpy as np
import pyquaternion as pq

WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"

VID_FPS = 60
OTS_FPS = 120


class GroundTruth(object):
    VID_FPS = 60
    OTS_FPS = 120
    src = "./src/models"

    def __init__(
        self,
        file: str,
        skp_rows: int,
        header_row_list: list,
        start_frame: int,
        end_frame: int,
    ):
        print(
            f"[{WARNING}WARNING{ENDC}]: Extracting and compiling ground truth data..."
        )
        # configure frame ID range for ground truth
        self.sframe = start_frame
        self.fframe = end_frame
        self.sframe *= int(self.OTS_FPS / self.VID_FPS)
        self.fframe *= int(self.OTS_FPS / self.VID_FPS)

        # get ground truth and joints config
        self.gnd_df = (
            pd.read_csv(
                file, skiprows=skp_rows, header=header_row_list, dtype="unicode"
            )
            .astype(float)
            .iloc[self.sframe : self.fframe, :]
        )
        with open(f"{self.src}/cfg_joints.json", "r") as cfg_model_file:
            self.joints_dict = json.load(cfg_model_file)
        self.cols = np.array(list(map(list, self.gnd_df.columns)))

    def get_joints(self, adj_joints_vec: dict):
        vertex = list(adj_joints_vec.keys())[0]  # size 1
        joint_list = list(*adj_joints_vec.values())
        assert (
            len(joint_list) == 3
        ), f"{FAIL}ERROR{ENDC}: Joint adjacency list must be of size 3"
        assert (
            joint_list[1] == vertex
        ), f"{FAIL}ERROR{ENDC}: Joint adjacency list not in proper order"

        # get root origin joint
        root_origin = self.__get_origin()

        # get xyz positional data of each joint
        num_joints = len(joint_list)
        gnd_pos = [None] * num_joints
        bone_rots = [None] * 2
        for i, joint in enumerate(joint_list):
            """
            Each joint has 1 bone, and could have `n` > 0 number of markers
            """
            gnd_truth_joint = self.joints_dict[joint]["GND_TRUTH"]
            bones = gnd_truth_joint["bones"]
            markers = gnd_truth_joint["markers"]

            # get columns that contain gnd-truth joint names for `joint`
            bone_col = self.__substrColMatcher(bones, self.cols)[0]
            marker_cols = self.__substrColMatcher(markers, self.cols)
            if i >= 1:
                bone_rots[i - 1] = self.gnd_df[bone_col]["Rotation"]
            col_names = bone_col if marker_cols == [] else marker_cols

            # get root joint (between hips) and set as origin of system
            gnd_pos[i] = self.__mean_xyz(col_names) - root_origin

        # compute relative quaternion and angle (degrees)
        df_bone_rots = pd.concat(bone_rots, axis=1).values
        gnd_quat = np.apply_along_axis(
            self.__compute_del_quat, axis=1, arr=df_bone_rots
        ).reshape(-1, 1)
        gnd_theta = np.apply_along_axis(
            self.__compute_theta, axis=1, arr=df_bone_rots
        ).reshape(-1, 1)

        return (np.array(gnd_pos), gnd_quat, gnd_theta)

    def __get_origin(self) -> np.ndarray:
        left_hip = self.joints_dict["LEFT_HIP"]["GND_TRUTH"]
        right_hip = self.joints_dict["RIGHT_HIP"]["GND_TRUTH"]

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

        root_origin = (left_hip_xyz + right_hip_xyz) / 2
        return root_origin

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

    def __substrColMatcher(self, joints : list, MICols : np.ndarray) -> list:
        return list(
            set(filter(lambda s, j=joints: any(map(s.__contains__, j)), MICols.T[0]))
        )


def str_has(string, *args):
    return any([marker_label in string for marker_label in set(args)])


def get_OTSData(file, sframe, fframe):
    print("===== OTS DATA EXTRACTION =====")

    skipped_frames = []
    OTS_data = pd.read_csv(file, skiprows=3, dtype="unicode")
    OTS_data = OTS_data.fillna("NULL").drop(labels=0, axis=0)
    QUAT_data = []
    POS_data = []
    for col in range(0, len(OTS_data.columns)):
        col_dat = OTS_data.iloc[:, col]
        if col_dat.values[0] == "Rotation":
            if str_has(col_dat.name, "RThigh", "RShin"):
                QUAT_data.append(col_dat)

        elif str_has(col_dat.name, "WaistLFront", "WaistRFront", "RKnee", "RAnkle"):
            POS_data.append(col_dat)

    # get the index of the rows that contains the start frame # data and the end frame # data
    frame_stats = (int(sframe / VID_FPS) * OTS_FPS, int(fframe / VID_FPS) * OTS_FPS)
    print(frame_stats)
    ind = OTS_data.index[
        (OTS_data.iloc[:, 0] == str(frame_stats[0]))
        | (OTS_data.iloc[:, 0] == str(frame_stats[1]))
    ].tolist()
    print(f"{ind=}")

    # extract quaternion information of FEMUR and TIBIA artifacts between the start frame # and the end frame #
    # extract joint positional data for each joint in the left leg
    FEM_quat = [None] * (ind[1] - ind[0])
    TIB_quat = [None] * (ind[1] - ind[0])
    OTS_jangle = [None] * (ind[1] - ind[0])
    OTS_quat = [None] * (ind[1] - ind[0])
    HIP_pos = [None] * (ind[1] - ind[0])
    KNEE_pos = [None] * (ind[1] - ind[0])
    ANKL_pos = [None] * (ind[1] - ind[0])
    torso_width = []
    for i in range(ind[0] - 1, ind[1] - 1):
        if any(
            QUAT_data[col_num].values[i] == "NULL"
            for col_num in range(0, len(QUAT_data))
        ) or (
            any(
                POS_data[col_num].values[i] == "NULL"
                for col_num in range(0, len(POS_data))
            )
        ):

            skipped_frames.append(str(i - ind[0]))
            continue

        lf_waist = np.array(
            [float(POS_data[0][i]), float(POS_data[1][i]), float(POS_data[2][i])]
        )
        rf_waist = np.array(
            [float(POS_data[3][i]), float(POS_data[4][i]), float(POS_data[5][i])]
        )
        diff = lf_waist - rf_waist
        torso_width.append(np.sqrt((np.sum(diff)) ** 2))

        root_origin = (lf_waist + rf_waist) / 2

        """
        # uncomment this if you want to use left limbs (really not too different when switching between left and right limbs)
        lknee_out = np.array([float(POS_data[6][i]), float(POS_data[7][i]), float(POS_data[8][i])])
        lknee_in = np.array([float(POS_data[9][i]), float(POS_data[10][i]), float(POS_data[11][i])])
        lankl_out = np.array([float(POS_data[12][i]), float(POS_data[13][i]), float(POS_data[14][i])])
        lankl_in = np.array([float(POS_data[15][i]), float(POS_data[16][i]), float(POS_data[17][i])])
        lf_waist = lf_waist - root_origin
        lknee_avg = ((lknee_out + lknee_in)/2) - root_origin
        lankl_avg = ((lankl_out + lankl_in)/2) - root_origin
        """

        rknee_out = np.array(
            [float(POS_data[6][i]), float(POS_data[7][i]), float(POS_data[8][i])]
        )
        rknee_in = np.array(
            [float(POS_data[9][i]), float(POS_data[10][i]), float(POS_data[11][i])]
        )
        rankl_out = np.array(
            [float(POS_data[12][i]), float(POS_data[13][i]), float(POS_data[14][i])]
        )
        rankl_in = np.array(
            [float(POS_data[15][i]), float(POS_data[16][i]), float(POS_data[17][i])]
        )
        rf_waist = rf_waist - root_origin
        rknee_avg = ((rknee_out + rknee_in) / 2) - root_origin
        rankl_avg = ((rankl_out + rankl_in) / 2) - root_origin

        HIP_pos[i - ind[0]] = rf_waist
        KNEE_pos[i - ind[0]] = rknee_avg
        ANKL_pos[i - ind[0]] = rankl_avg

        quat_fem = pq.Quaternion(
            QUAT_data[3].values[i],
            QUAT_data[0].values[i],
            QUAT_data[1].values[i],
            QUAT_data[2].values[i],
        )
        quat_tib = pq.Quaternion(
            QUAT_data[7].values[i],
            QUAT_data[4].values[i],
            QUAT_data[5].values[i],
            QUAT_data[6].values[i],
        )

        # relative quaternion from FEMUR to TIBIA
        quat_joint = quat_fem.conjugate * quat_tib
        # joint angle: .degrees = abs(2 * atan2d(norm(q(1:3), q(4))))
        # q(1:3) => imaginary components; q(4) => real component
        OTS_jangle[i - ind[0]] = 180 - np.abs(quat_joint.degrees)

        FEM_quat[i - ind[0]] = (
            quat_fem.elements[0],
            quat_fem.elements[1],
            quat_fem.elements[2],
            quat_fem.elements[3],
        )
        TIB_quat[i - ind[0]] = (
            quat_tib.elements[0],
            quat_tib.elements[1],
            quat_tib.elements[2],
            quat_tib.elements[3],
        )
        OTS_quat[i - ind[0]] = (
            quat_joint.elements[0],
            quat_joint.elements[1],
            quat_joint.elements[2],
            quat_joint.elements[3],
        )

    OTS_pos = [HIP_pos, KNEE_pos, ANKL_pos]
    torso_width = np.array(torso_width)
    siz = torso_width.size

    return (
        OTS_pos,
        OTS_quat,
        OTS_jangle,
        skipped_frames,
        np.sum(torso_width) / siz,
        frame_stats,
    )
