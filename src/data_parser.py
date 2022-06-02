import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from resampy import resample
from scipy import signal

MOD_FPS = 25
OTS_FPS = 120


def MPJPE(ots_joints, model_joints):
    mpjpe_dat = pd.Series([0])
    for j in range(0, len(ots_joints)):
        model_joints[j] = model_joints[j].to_numpy(copy=True)
        ots_joints[j] = ots_joints[j].to_numpy(copy=True)
        diff = model_joints[j] - ots_joints[j]
        exp = diff**2
        d = np.sqrt(exp.sum(axis=1))
        tot = d.size
        mpjpe_dat.at[j] = d.sum(axis=0) / tot

    return mpjpe_dat


def PDJ(ots_joints, model_joints, torso_diam):
    vals = []
    for j in range(0, len(ots_joints)):
        model_joints[j] = model_joints[j].to_numpy(copy=True)
        ots_joints[j] = ots_joints[j].to_numpy(copy=True)
        diff = model_joints[j] - ots_joints[j]
        exp = diff**2
        d = np.sqrt(exp.sum(axis=1))
        tot = d.size
        if (d.sum() / tot) < (0.2 * torso_diam):
            vals.append(1)
        else:
            vals.append(0)

    pdj_true = vals.count(1)
    return [((pdj_true / len(vals)) * 100)]


def arun(MODEL, OTS):
    shape = MODEL.shape  # shape[0] = dimensions of coordinate; shape[1] = # of points

    MODEL_c = MODEL.T.mean(axis=0).reshape((-1, 1))
    MODEL = MODEL - np.tile(MODEL_c, (1, shape[1]))

    OTS_c = OTS.T.mean(axis=0).reshape((-1, 1))
    OTS = OTS - np.tile(OTS_c, (1, shape[1]))

    U, _, V = np.linalg.svd(MODEL @ OTS.T)
    R = V @ U.T
    t = OTS_c - (R @ MODEL_c)

    return R, t


def calibrate(MODEL, OTS):
    diff = MODEL.shape[0] - OTS.shape[0]
    if diff > 0:
        MODEL = MODEL.iloc[: (-1 * diff)]
    elif diff < 0:
        OTS = OTS.iloc[:diff]

    OTS = OTS.to_numpy(copy=True)
    MODEL = MODEL.to_numpy(copy=True)
    R, t = arun(MODEL.T, OTS.T)
    MODEL_t = (R @ MODEL.T) + t
    MODEL_t = MODEL_t.T

    return MODEL_t


def align(df_model: DataFrame, df_ots: DataFrame):
    ots_len = df_ots.shape[0]
    mod_len = df_model.shape[0]
    sr_ratio = ots_len / mod_len

    new_model_data = []
    for c in range(len(df_model.columns)):
        ots = df_ots.iloc[:, c].values
        model = df_model.iloc[:, c].values
        r_model_data = resample(x=model, sr_orig=(OTS_FPS / sr_ratio), sr_new=OTS_FPS)

        xcorr = signal.correlate(r_model_data, ots)
        xcorr /= np.max(xcorr)
        dx = np.mean(np.diff(range(0, r_model_data.shape[0])))
        shift = np.argmax((signal.correlate(r_model_data, ots)) - ots.shape[0] - 1) * dx
        shift %= r_model_data.shape[0]
        shift = int(shift)
        r_model_data = np.concatenate(
            (r_model_data[shift:], r_model_data[:shift]), axis=0
        ).flatten()
        new_model_data.append(r_model_data)

    new_model_data = np.array(new_model_data)
    df_new_model = pd.DataFrame(new_model_data.T, columns=df_model.columns)

    return df_new_model


def data_parse2(model_name: str, model_data: dict, joints: dict) -> pd.DataFrame:
    """
    Parse pose estimation data into DataFrames with the following order:
        [POSITION (XYZ), THETA (joint angle), QUATERNION (orientation), ... ]

    Parameters
    ----------
    model_name: name of pose estimation algo
    model_data: dictionary of estimated joint positions, joint angles, and orientations
    joints: dictionary of graphical relation of joints
            {'vertex' : [joint_a, vertex, joint_b] (any order)}

    Return
    ------
    DataFrame object of estimated data
    """
    df = None
    for v, joint_list in joints.items():
        # column names
        theta_column = [f"{model_name}:{v}:theta"]
        pos_columns = [
            f"{model_name}:{j}:pos-{axis}"
            for j in joint_list
            for axis in ["X", "Y", "Z"]
        ]
        quat_columns = [
            f"{model_name}:{j}:quat-{axis}"
            for j in joint_list
            for axis in ["W", "X", "Y", "Z"]
        ]

        # make dataframes
        df_theta = pd.DataFrame(
            model_data["theta"], columns=theta_column
        )

        # Note: need to make sure that the order of model_data matches column labels!
        df_joints = None
        for i in range(len(joint_list)):
            df_pos = pd.DataFrame(
                model_data["pos"][i],
                columns=pos_columns[(3 * i) : (3 * (i + 1))],
            )
            """
            df_quat = pd.DataFrame(
                model_data["quat"], columns=quat_columns[(4*i):(4*(i + 1))]
            )
            """
            df_joints = pd.concat([df_joints, df_pos], axis=1)

        df = pd.concat([df, df_theta, df_joints], axis=1)

    return df
