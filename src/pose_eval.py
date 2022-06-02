import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from resampy import resample
from scipy import signal

MOD_FPS = 60
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


def calibrate(df_model : DataFrame, df_ots : DataFrame):
    '''
    diff = MODEL.shape[0] - OTS.shape[0]
    if diff > 0:
        MODEL = MODEL.iloc[: (-1 * diff)]
    elif diff < 0:
        OTS = OTS.iloc[:diff]
    '''
    OTS = df_ots.to_numpy(copy=True)
    MODEL = df_model.to_numpy(copy=True)
    R, t = arun(MODEL.T, OTS.T)
    MODEL_t = (R @ MODEL.T) + t
    MODEL_t = pd.DataFrame(MODEL_t.T, columns=df_model.columns)

    return MODEL_t


def align(df_model: pd.DataFrame, df_ots: pd.DataFrame):
    ots_len = df_ots.shape[0]
    mod_len = df_model.shape[0]
    sr_ratio = ots_len / mod_len

    new_model_data = []
    for c in range(len(df_model.columns)):
        model = df_model.iloc[:, c].values
        r_model_data = resample(x=model, sr_orig=(OTS_FPS / sr_ratio), sr_new=OTS_FPS)
        new_model_data.append(r_model_data)

    new_model_data = np.array(new_model_data)
    df_new_model = pd.DataFrame(new_model_data.T, columns=df_model.columns)
    return df_new_model

