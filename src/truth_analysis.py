import pandas as pd
import numpy as np
import pyquaternion as pq

OTS_FPS = 120


def str_has(string, *args):
    return any([marker_label in string for marker_label in set(args)])


def get_OTSData(file, test_type, sframe, vid_duration):
    print("===== OTS DATA EXTRACTION =====")
    FEM_quat = {}
    TIB_quat = {}
    OTS_jangle = {}
    OTS_quat = {}

    skipped_frames = []
    OTS_data = pd.read_csv(file, skiprows=3, dtype="unicode")
    OTS_data = OTS_data.fillna("NULL").drop(labels=0, axis=0)
    QUAT_data = []
    POS_data = []
    for col in range(0, len(OTS_data.columns)):
        col_dat = OTS_data.iloc[:, col]
        if col_dat.values[0] == "Rotation":
            if test_type is None or str_has(col_dat.name, "RThigh", "RShin"):
                QUAT_data.append(col_dat)

        elif str_has(col_dat.name, "WaistLFront", "WaistRFront", "RKnee", "RAnkle"):
            POS_data.append(col_dat)

    # get the index of the rows that contains the start frame # data and the end frame # data
    frame_stats = (int(sframe * 2 - 1), int((sframe * 2 - 1) + vid_duration * OTS_FPS))
    print(f"Frame stats: {frame_stats}")
    ind = OTS_data.index[
        (OTS_data.iloc[:, 0] == str(frame_stats[0]))
        | (OTS_data.iloc[:, 0] == str(frame_stats[1]))
    ].tolist()

    # extract quaternion information of FEMUR and TIBIA artifacts between the start frame # and the end frame #
    # extract joint positional data for each joint in the left leg
    HIP_pos = {}
    KNEE_pos = {}
    ANKL_pos = {}
    torso_width = []
    for i in range(ind[0], ind[1]):
        if any(
            QUAT_data[col_num].values[i] == "NULL"
            for col_num in range(0, len(QUAT_data))
        ) or (
            (test_type is not None)
            and any(
                POS_data[col_num].values[i] == "NULL"
                for col_num in range(0, len(POS_data))
            )
        ):

            skipped_frames.append(str(i - ind[0]))
            continue

        if test_type is not None:
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

            HIP_pos[str(i - ind[0])] = rf_waist
            KNEE_pos[str(i - ind[0])] = rknee_avg
            ANKL_pos[str(i - ind[0])] = rankl_avg

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
        OTS_jangle[str(i - ind[0])] = 180 - np.abs(quat_joint.degrees)

        FEM_quat[str(i - ind[0])] = (
            quat_fem.elements[0],
            quat_fem.elements[1],
            quat_fem.elements[2],
            quat_fem.elements[3],
        )
        TIB_quat[str(i - ind[0])] = (
            quat_tib.elements[0],
            quat_tib.elements[1],
            quat_tib.elements[2],
            quat_tib.elements[3],
        )
        OTS_quat[str(i - ind[0])] = (
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
