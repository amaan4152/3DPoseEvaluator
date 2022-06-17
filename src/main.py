import argparse as ap
from pose_gen import pose_gen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {"VIBE": "V", "GAST": "G", "BLAZEPOSE": "B"}

import subprocess

# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
HEADER = "\033[95m"
OKBLUE = "\033[94m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"


def logging(l_type: str, mssg: str):
    log_type = {
        "ERROR": FAIL,
        "WARNING": WARNING,
        "GOOD": OKGREEN,
    }
    print(f"[{log_type[l_type]}{l_type}{ENDC}]: {mssg}")


def get_duration(filename):
    # ref: https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return float(result.stdout)


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
        "--eval",
        action="store_true",
        help="Enable to generate raw pose data",
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
    pcli.add_argument(
        "--animate",
        action="store_true",
        help="generate pose animation files for specified algorithm",
    )

    return pcli, pcli.parse_args()


import matplotlib.ticker as ticker
import os
import seaborn as sns

sns.set()

"""
TO-DO:
    - apply alignment rule to the OTS data 
    - perform calibration using arun method 
    - MPJPE and PDJ error metric calcs
"""
def generate_plots(model_name : str, vid_name: str):
    subdf = lambda df, substr: df.iloc[:, df.columns.str.contains(substr)]

    df_list = {"JA": [], "JOINTS": [], "MODELS": []}
    data_files = [f for f in os.listdir("output/") if "raw_data" in f and vid_name in f]
    for file in data_files:
        if "gnd" not in file:
            df_raw = pd.read_csv(f"output/{file}")
            df_JA = subdf(df_raw, "theta")  # joint angle dataframe
            attr = np.array([col_name.split(":") for col_name in df_JA.columns])
            df_list["JA"].append(df_JA)
            df_list["JOINTS"] = attr[:, 1]

            m_name = attr[:, 0].flatten()[0] 
            df_list["MODELS"].append(m_name)

    # resample GND_TRUTH
    gnd_file = [f for f in data_files if "gnd" in f][0]
    df_gnd = pd.read_csv(f"output/{gnd_file}").iloc[:, 1]
    df_gnd = df_gnd.iloc[::2].reset_index(drop=True)
    delay = df_gnd.size - df_list["JA"][0].size - 1
    df_gnd = df_gnd.iloc[0:(-1-delay)].reset_index(drop=True)
    
    for j in df_list["JOINTS"]:
        df_models = pd.concat(
            [df[f"{m}:{j}:theta"] for df, m in zip(df_list["JA"], df_list["MODELS"])],
            axis=1,
        )
        df_models = pd.concat([df_gnd, df_models], axis=1)
        print(df_models)
        plt.figure(figsize=(27,11))
        ax = sns.lineplot(data=df_models)
        plt.legend(labels=df_list["MODELS"])
        plt.title(f"{j} Joint Models Plot")
        plt.xlabel("Frame ID")
        plt.ylabel(r"$\theta(^{\circ})$")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.xlim(left=-10)
        plt.tight_layout()
        plt.savefig(f"output/{j.upper()}_JOINTANGLE_PLOT.png")
        plt.close()


from mpl_toolkits import mplot3d
from pathlib import Path
from pose_eval import calibrate, align
from truth_analysis import get_OTSData
from data_parser import data_parse2


def main():
    # parse CLI arguments
    pcli, args = cli_parse()
    if None in (args.model, args.video):
        print("ERROR: must provide arguments for '-m' and '-v'\n")
        pcli.print_help()
        exit(1)

    joints = {"RIGHT_KNEE": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]}

    # duration = get_duration(args.video)
    # print(f"Duration: {duration}")

    # pose raw data generation
    mod_name = args.model.lower()
    vid_name = Path(args.video).name.lower().split(".")[0]
    if not args.eval:
        df_m_raw = pose_gen(
            args.video, args.data, mod_name, args.animate, args.start, args.end, joints
        )
        if not args.animate:
            # df_m_raw.to_csv(f"output/{mod_name}-{vid_name}-raw_data.csv")
            generate_plots(args.model, vid_name)
        else:
            logging(
                "WARNING",
                "No pose data and plot have been generated due to animation flag...",
            )

        logging("GOOD", "successful termination without error")

    # pose evaluation
    else:
        (
            OTS_pos,
            OTS_quat,
            OTS_theta,
            skpd_frames,
            torso_width,
            frame_stat,
        ) = get_OTSData(args.data, args.start, args.end)
        ots_data = {"theta": OTS_theta, "pos": OTS_pos, "quat": OTS_quat}
        df_ots_raw = data_parse2("OTS", ots_data, joints)
        df_ots_raw.to_csv("output/ots_data.csv")

        df_m_raw = pd.read_csv(
            f"output/{mod_name}-{vid_name}-raw_data.csv", index_col=[0]
        )
        df_ots_raw = pd.read_csv("output/ots_data.csv", index_col=[0])

        df_m_aligned = align(df_m_raw, df_ots_raw)
        df_m_cal = df_m_aligned.iloc[:, 0]
        for m_i, o_i in zip(
            range(1, len(df_m_aligned.columns), 3), range(1, len(df_ots_raw.columns), 3)
        ):
            pos_m_data = df_m_aligned.iloc[:, m_i : (m_i + 3)]
            pos_o_data = df_ots_raw.iloc[:, o_i : (o_i + 3)]
            df_cal = calibrate(pos_m_data, pos_o_data)
            print(pd.concat([pos_m_data, df_cal], axis=1))
            df_m_cal = pd.concat([df_m_cal, df_cal], axis=1)

        df_m_aligned.to_csv(f"output/{mod_name}_aligned_data.csv")
        df_m_cal.to_csv(f"output/{mod_name}_cal_data.csv")

        # dist, ix, iy = edr(df_cal.iloc[:, 0], df_cal.iloc[:, 11], 0.1)
        # print(ix.shape)
        # print(iy.shape)

        frame_ids = df_ots_raw.index.tolist()
        sns.lineplot(x=frame_ids, y=df_m_cal.iloc[:, 0].values, label="GAST")
        sns.lineplot(x=frame_ids, y=df_ots_raw.iloc[:, 0].values, label="OTS")
        plt.legend()
        # plt.plot(np.arange(1, df_cal.shape[0] + 1), df_cal.iloc[:, 0].values)
        # plt.plot(iy.flatten(), df_cal.iloc[iy.flatten(), 11].values)
        plt.savefig(f"output/{mod_name}_JA_aligned_graph.png")
        plt.close()

        alg_hip_data = df_m_aligned.iloc[:, 4:7].values
        cal_hip_data = df_m_cal.iloc[:, 4:7].values
        ots_hip = df_ots_raw.iloc[:, 4:7].values
        x_cal = cal_hip_data[:, 0]
        y_cal = cal_hip_data[:, 1]
        z_cal = cal_hip_data[:, 2]
        x_alg = alg_hip_data[:, 0]
        y_alg = alg_hip_data[:, 1]
        z_alg = alg_hip_data[:, 2]
        x_ots = ots_hip[:, 0]
        y_ots = ots_hip[:, 1]
        z_ots = ots_hip[:, 2]
        print(pd.DataFrame(x_cal - x_alg))

        plt.figure(figsize=(15, 13))
        ax = plt.axes(projection="3d")
        ax.scatter3D(y_cal, z_cal, x_cal, label="cal")
        ax.scatter3D(x_ots, y_ots, z_ots, label="ots")
        plt.legend()
        plt.savefig(f"output/{mod_name}_HIP_TRAJ_PLOT.png")
        plt.close()


if __name__ == "__main__":
    main()
