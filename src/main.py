import argparse as ap
from pose_gen import pose_gen
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

MODELS = {"VIBE": "V", "GAST": "G", "BLAZEPOSE": "B"}

import subprocess


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
def generate_plots(vid_name : str):
    subdf = lambda df,substr : df.iloc[:, df.columns.str.contains(substr)]

    df_list = {'JA': [], 'JOINTS': [], 'MODELS': []}
    data_files = [f for f in os.listdir("output/") if "raw_data" in f and vid_name in f]
    for file in data_files:
        df_raw = pd.read_csv(f"output/{file}")
        df_JA = subdf(df_raw, "theta")  # joint angle dataframe
        attr = np.array([col_name.split(':') for col_name in df_JA.columns])
        df_list['JA'].append(df_JA)
        df_list['JOINTS'] = attr[:, 1]
        df_list['MODELS'].append(*attr[:, 0].flatten())

    for j in df_list['JOINTS']:
        df_models = pd.concat([df[f"{m}:{j}:theta"] for df, m in zip(df_list['JA'], df_list['MODELS'])], axis=1)
        print(df_models)
        plt.figure(figsize=(19, 9))
        ax = sns.lineplot(data=df_models)
        plt.legend(labels=df_list["MODELS"])
        plt.title(f"{j} Joint Models Plot")
        plt.xlabel("Frame ID");
        plt.ylabel(r'$\theta(^{\circ})$')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.xlim(left=-10)
        plt.tight_layout()
        plt.savefig(f"output/{j.upper()}_JOINTANGLE_PLOT.png")
        plt.close()


from pathlib import Path
from pose_eval import calibrate, align
from edr import edr
def main():
    # parse CLI arguments
    pcli, args = cli_parse()
    if None in (args.model, args.video):
        print("ERROR: must provide arguments for '-m' and '-v'\n")
        pcli.print_help()
        exit(1)

    joints = {"KNEE": ["HIP", "KNEE", "ANKLE"]}

    # duration = get_duration(args.video)
    # print(f"Duration: {duration}")

    # pose raw data generation
    if not args.eval:
        mod_name = args.model.lower()
        vid_name = Path(args.video).name.lower()
        df_raw = pose_gen(
            args.video, None, mod_name, args.animate, None, None, joints, None
        )
        df_raw.to_csv(f"output/{mod_name}-{vid_name}-raw_data.csv")
        generate_plots(vid_name)

    # pose evaluation
    else:
        subdf = lambda df,substr : df.iloc[:, df.columns.str.contains(substr)]

        df_raw = pd.read_csv("output/raw_data.csv")
        df_ots_raw = subdf(df_raw, "OTS")
        df_m_raw = subdf(df_raw, args.model.lower()).dropna()

        df_m_aligned = align(df_m_raw, df_ots_raw)
        df_m_cal = df_m_aligned.iloc[:, 0]
        for m_i, o_i in zip(range(1, len(df_m_aligned.columns), 3), range(1, len(df_ots_raw.columns), 3)):
            pos_m_data = df_m_aligned.iloc[:, m_i:(m_i + 3)]
            pos_o_data = df_ots_raw.iloc[:, o_i:(o_i + 3)]
            df_m_cal = pd.concat([df_m_cal, calibrate(pos_m_data, pos_o_data)], axis=1)

        df_m_aligned.to_csv("output/aligned_data.csv")
        df_cal = pd.concat([df_ots_raw, df_m_cal], axis=1)
        df_cal.to_csv("output/cal_data.csv")
        dist, ix, iy = edr(df_cal.iloc[:, 0], df_cal.iloc[:, 11], 0.1)
        print(ix.shape)
        print(iy.shape)
        plt.plot(np.arange(1, df_cal.shape[0] + 1), df_cal.iloc[:, 0].values)
        plt.plot(iy.flatten(), df_cal.iloc[iy.flatten(), 11].values)
        plt.savefig("output/JA_graph.png")
        plt.close()

if __name__ == "__main__":
    main()
