import argparse as ap
from pose_gen import pose_gen
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

from pose_eval import calibrate, align
def main():
    pcli, args = cli_parse()
    if None in (args.model, args.video, args.data):
        print("ERROR: must provide arguments for '-m' and '-v'\n")
        pcli.print_help()
        exit(1)

    joints = {"KNEE": ["HIP", "KNEE", "ANKLE"]}

    duration = get_duration(args.video)
    print(f"Duration: {duration}")

    # pose raw data generation
    if not args.eval:
        df_raw, df_ots_raw, df_m_raw = pose_gen(
            args.video, args.data, args.model.lower(), args.animate, args.start, args.end, joints, duration
        )
        df_raw.to_csv("output/raw_data.csv")

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


if __name__ == "__main__":
    main()
