import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from resampy import resample
from scipy import signal

MOD_FPS = 60
GND_FPS = 120


def MPJPE(gnd_joints, model_joints):
    mpjpe_dat = pd.Series([0])
    for j in range(0, len(gnd_joints)):
        model_joints[j] = model_joints[j].to_numpy(copy=True)
        gnd_joints[j] = gnd_joints[j].to_numpy(copy=True)
        diff = model_joints[j] - gnd_joints[j]
        exp = diff**2
        d = np.sqrt(exp.sum(axis=1))
        tot = d.size
        mpjpe_dat.at[j] = d.sum(axis=0) / tot

    return mpjpe_dat


def PDJ(gnd_joints, model_joints, torso_diam):
    vals = []
    for j in range(0, len(gnd_joints)):
        model_joints[j] = model_joints[j].to_numpy(copy=True)
        gnd_joints[j] = gnd_joints[j].to_numpy(copy=True)
        diff = model_joints[j] - gnd_joints[j]
        exp = diff**2
        d = np.sqrt(exp.sum(axis=1))
        tot = d.size
        if (d.sum() / tot) < (0.2 * torso_diam):
            vals.append(1)
        else:
            vals.append(0)

    pdj_true = vals.count(1)
    return [((pdj_true / len(vals)) * 100)]


def arun(MODEL, GND):
    shape = MODEL.shape  # shape[0] = dimensions of coordinate; shape[1] = # of points

    MODEL_c = MODEL.T.mean(axis=0).reshape((-1, 1))
    MODEL = MODEL - np.tile(MODEL_c, (1, shape[1]))

    GND_c = GND.T.mean(axis=0).reshape((-1, 1))
    GND = GND - np.tile(GND_c, (1, shape[1]))

    U, _, V = np.linalg.svd(MODEL @ GND.T)
    R = V @ U.T
    t = GND_c - (R @ MODEL_c)

    return R, t


def calibrate(R, t, df_model: DataFrame):
    """
    diff = MODEL.shape[0] - GND.shape[0]
    if diff > 0:
        MODEL = MODEL.iloc[: (-1 * diff)]
    elif diff < 0:
        GND = GND.iloc[:diff]
    """
    MODEL = df_model.to_numpy(copy=True)
    MODEL_n = (R @ MODEL.T) + t
    MODEL_n = pd.DataFrame(MODEL_n.T, columns=df_model.columns)

    return MODEL_n


def align(df_model: pd.DataFrame, df_gnd: pd.DataFrame):
    gnd_len = df_gnd.shape[0]
    mod_len = df_model.shape[0]
    sr_ratio = gnd_len / mod_len

    new_model_data = []
    for c in range(len(df_model.columns)):
        model = df_model.iloc[:, c].values
        r_model_data = resample(x=model, sr_orig=(GND_FPS / sr_ratio), sr_new=GND_FPS)
        new_model_data.append(r_model_data)

    new_model_data = np.array(new_model_data)
    df_new_model = pd.DataFrame(new_model_data.T, columns=df_model.columns)
    return df_new_model
