from dtw import *
import pandas as pd
import numpy as np
import argparse as ap
import linmdtw as ldtw
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
        exp = diff ** 2
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
        exp = diff ** 2
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


# fix for multiple models running at once
def data_parse(df, ots, model, model_name, torso_diam):
    if ots != None:
        df_first = pd.DataFrame.from_dict(
            ots[0], orient="index", columns=["OTS:KNEE_JOINT:ANGLE"]
        )
        # df_ots_q = pd.DataFrame.from_dict(ots[2], orient='index', columns=['OTS:KNEE_JOINT:W', 'OTS:KNEE_JOINT:X', 'OTS:KNEE_JOINT:Y', 'OTS:KNEE_JOINT:Z'])
        df_ots_p_h = pd.DataFrame.from_dict(
            ots[1][0], orient="index", columns=["OTS:HIP:X", "OTS:HIP:Y", "OTS:HIP:Z"]
        )
        df_ots_p_k = pd.DataFrame.from_dict(
            ots[1][1],
            orient="index",
            columns=["OTS:KNEE:X", "OTS:KNEE:Y", "OTS:KNEE:Z"],
        )
        df_ots_p_a = pd.DataFrame.from_dict(
            ots[1][2],
            orient="index",
            columns=["OTS:ANKLE:X", "OTS:ANKLE:Y", "OTS:ANKLE:Z"],
        )
        df_first = pd.concat([df_first, df_ots_p_h, df_ots_p_k, df_ots_p_a], axis=1)
        # df_ots = df_first
        # df_ots.to_csv("PARSED_OTS_00.csv", index=False)

    elif not df.empty:
        df_first = df
        # df_ots_q = df.iloc[:,1:5]
        df_ots_p_h = df.iloc[:, 5:8]
        df_ots_p_k = df.iloc[:, 8:11]
        df_ots_p_a = df.iloc[:, 11:14]
        # df_ots = pd.concat([df_first, df_ots_p_h, df_ots_p_k, df_ots_p_a], axis=1)

    df_mod_theta = pd.DataFrame.from_dict(
        model[0], orient="index", columns=[model_name + str(":KNEE_JOINT:ANGLE")]
    )
    # df_mod_p_q = pd.DataFrame.from_dict(model[2], orient='index', columns=[model_name+':KNEE_JOINT:W', model_name+':KNEE_JOINT:X', model_name+':KNEE_JOINT:Y', model_name+':KNEE_JOINT:Z'])
    df_mod_p_h = pd.DataFrame.from_dict(
        model[1][0],
        orient="index",
        columns=[model_name + ":HIP:X", model_name + ":HIP:Y", model_name + ":HIP:Z"],
    )
    df_mod_p_k = pd.DataFrame.from_dict(
        model[1][1],
        orient="index",
        columns=[
            model_name + ":KNEE:X",
            model_name + ":KNEE:Y",
            model_name + ":KNEE:Z",
        ],
    )
    df_mod_p_a = pd.DataFrame.from_dict(
        model[1][2],
        orient="index",
        columns=[
            model_name + ":ANKLE:X",
            model_name + ":ANKLE:Y",
            model_name + ":ANKLE:Z",
        ],
    )

    # calibrate model data with OTS data
    df_mod_p_h.iloc[:] = calibrate(df_mod_p_h, df_ots_p_h)
    df_mod_p_k.iloc[:] = calibrate(df_mod_p_k, df_ots_p_k)
    df_mod_p_a.iloc[:] = calibrate(df_mod_p_a, df_ots_p_a)

    # align model data with OTS data
    df_mod_theta = align(df_mod_theta, df_first)
    df_mod_p_h = align(df_mod_p_h, df_ots_p_h)
    df_mod_p_k = align(df_mod_p_k, df_ots_p_k)
    df_mod_p_a = align(df_mod_p_a, df_ots_p_a)

    # compose model data and smoothen it
    df_mod = pd.concat([df_mod_theta, df_mod_p_h, df_mod_p_k, df_mod_p_a], axis=1)
    df_mod = df_mod.apply(lambda srs: signal.savgol_filter(srs.values, 51, 7))
    df_mod_true = df_mod

    # compose newly compiled modeled data with origin dataframe
    df_cat = pd.concat(
        [df_first.reset_index(drop=True), df_mod_true.reset_index(drop=True)], axis=1
    )

    # break
    df_cat["BLANK"] = pd.Series(dtype=float)

    # error stats
    df_cat["THETA-err:" + model_name[0] + "vp:" + model_name[-1]] = (
        df_cat[model_name + str(":KNEE_JOINT:ANGLE")] - df_cat["OTS:KNEE_JOINT:ANGLE"]
    )

    # df_cat['KNEE_JOINT:W-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat[model_name+':KNEE_JOINT:W'] - df_cat['OTS:KNEE_JOINT:W']
    # df_cat['KNEE_JOINT:X-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat[model_name+':KNEE_JOINT:X'] - df_cat['OTS:KNEE_JOINT:X']
    # df_cat['KNEE_JOINT:Y-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat[model_name+':KNEE_JOINT:Y'] - df_cat['OTS:KNEE_JOINT:Y']
    # df_cat['KNEE_JOINT:Z-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat[model_name+':KNEE_JOINT:Z'] - df_cat['OTS:KNEE_JOINT:Z']

    for part in ("HIP", "KNEE", "ANKLE"):
        df_cat[part + ":X-err:" + model_name[0] + "vp:" + model_name[-1]] = (
            df_cat[model_name + ":" + part + ":X"] - df_cat["OTS:" + part + ":X"]
        )
        df_cat[part + ":Y-err:" + model_name[0] + "vp:" + model_name[-1]] = (
            df_cat[model_name + ":" + part + ":Y"] - df_cat["OTS:" + part + ":Y"]
        )
        df_cat[part + ":Z-err:" + model_name[0] + "vp:" + model_name[-1]] = (
            df_cat[model_name + ":" + part + ":Z"] - df_cat["OTS:" + part + ":Z"]
        )

        df_cat[part + ":X-err:" + model_name[0] + "vp:" + model_name[-1]] = df_cat[
            part + ":X-err:" + model_name[0] + "vp:" + model_name[-1]
        ].abs()
        df_cat[part + ":Y-err:" + model_name[0] + "vp:" + model_name[-1]] = df_cat[
            part + ":Y-err:" + model_name[0] + "vp:" + model_name[-1]
        ].abs()
        df_cat[part + ":Z-err:" + model_name[0] + "vp:" + model_name[-1]] = df_cat[
            part + ":Z-err:" + model_name[0] + "vp:" + model_name[-1]
        ].abs()

    df_cat["THETA-err:" + model_name[0] + "vp:" + model_name[-1]] = df_cat[
        "THETA-err:" + model_name[0] + "vp:" + model_name[-1]
    ].abs()

    # df_cat['KNEE_JOINT:W-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat['KNEE_JOINT:W-err:'+model_name[0]+'vp:'+model_name[-1]].abs()
    # df_cat['KNEE_JOINT:X-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat['KNEE_JOINT:X-err:'+model_name[0]+'vp:'+model_name[-1]].abs()
    # df_cat['KNEE_JOINT:Y-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat['KNEE_JOINT:Y-err:'+model_name[0]+'vp:'+model_name[-1]].abs()
    # df_cat['KNEE_JOINT:Z-err:'+model_name[0]+'vp:'+model_name[-1]] = df_cat['KNEE_JOINT:Z-err:'+model_name[0]+'vp:'+model_name[-1]].abs()

    return (
        df_cat,
        df_mod,
        MPJPE(
            [df_ots_p_h, df_ots_p_k, df_ots_p_a], [df_mod_p_h, df_mod_p_k, df_mod_p_a]
        ),
        PDJ(
            [df_ots_p_h, df_ots_p_k, df_ots_p_a],
            [df_mod_p_h, df_mod_p_k, df_mod_p_a],
            torso_diam,
        ),
    )


def cli_parse():
    pcli = ap.ArgumentParser(
        description="Evaluate GAST-NET, VIBE, and 3DMPPE 3D pose estimation algorithms on NIST video files to determine error metrics, \
                                          such as PDJ and MPJPE, against ground truth data (OTS and IMU data)."
    )

    pcli.add_argument(
        "-m",
        "--model",
        action="store",
        type=str,
        help="specific 3D pose estimation model to be used; if not specified, then all models will be executed",
    )
    pcli.add_argument(
        "-v",
        "--video",
        action="store",
        type=str,
        help="input video path to be processed",
    )
    pcli.add_argument(
        "-t",
        "--test",
        action="store",
        type=str,
        help='specify GAIT for "--test" to accomodate for GAIT OTS data',
    )
    pcli.add_argument(
        "-d", "--data", action="store", type=str, help="ground truth data"
    )
    pcli.add_argument(
        "--start",
        action="store",
        type=int,
        default=1,
        help="start frame # for video processing; default start frame is the first frame of video file",
    )
    pcli.add_argument(
        "--end",
        action="store",
        type=int,
        default=-1,
        help="end frame # for video processing; default end frame is the last frame of video file",
    )

    return pcli, pcli.parse_args()
