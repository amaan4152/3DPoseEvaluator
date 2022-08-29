import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
from GroundTruth import GroundTruth

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
        description="Evaluate GAST-NET, VIBE, and Blazepose 3D pose estimation algorithms on NIST video files to determine error metrics, \
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


import os
import seaborn as sns

sns.set()


def generate_plots(df_gnd_re: pd.DataFrame, vid_name: str, joints: dict):
    df_list = {"THETA": []}
    mod_groupings = np.array(list(os.walk("output"))[1:], dtype=object)
    models = mod_groupings[:, 0]    # subdirectory names in output dir
    data_files = mod_groupings[:, 1:].flatten().tolist()
    data_files = [f for m in data_files for f in m if f and "raw_data" in f and vid_name in f and "gnd" not in f]
    for mod_name, file in zip(models, data_files):
        df_raw = pd.read_csv(f"output/{mod_name}/{file}", header=[0, 1, 2, 3])
        df_theta = [df_raw[mod_name][j]["Joint Angle"] for j in joints.keys()]
        for df in df_theta:
            df.columns = [mod_name]
        df_list["THETA"].extend(df_theta)

    for j in joints.keys():
        df_models = pd.concat(df_list["THETA"], axis=1)
        if df_gnd_re is not None:
            df_gnd = df_gnd_re["GND_TRUTH"][j]["Joint Angle"]
            df_gnd.columns = ["ground-truth"]
            df_models = pd.concat([df_gnd, df_models], axis=1)

        joint_name = " ".join([s.capitalize() for s in j.split("_")])
        plt.figure(figsize=(15, 5))
        ax = sns.lineplot(data=df_models)
        plt.title(f"{joint_name} Joint Angle Plot")
        plt.xlabel("Frame ID")
        plt.ylabel(r"$\theta(^{\circ})$")
        plt.xlim(left=-10)
        plt.tight_layout()
        plt.savefig(f"output/{j.upper()}_JOINTANGLE_PLOT.png")
        plt.close()


def init_dirs(model : str):
    mod_name = model.lower()
    mod_dir = f"output/{mod_name}"
    gnd_dir = "output/gnd_truth"
    if not os.path.exists(mod_dir):
        os.mkdir(mod_dir)
    if not os.path.exists(gnd_dir):
        os.mkdir(gnd_dir)


from pose_gen import data_parse, pose_gen
from Evaluator import Evaluator
from pathlib import Path


def main(
    model : str,
    video : str,
    data : str,
    start_frame : int,
    end_frame : int,
    animate : bool,
    eval : bool,
):
    joints = {"RIGHT_KNEE": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]}
    joint_list = list(*joints.values())
    # duration = get_duration(video)
    # print(f"Duration: {duration}")

    # pose raw data generation
    mod_name = model.lower()
    vid_name = Path(video).name.lower().split(".")[0]
    if not eval:
        df_m_raw = pose_gen(
            video, data, mod_name, animate, start_frame, end_frame, joints
        )
        if not animate:
            df_m_raw.to_csv(f"output/{mod_name}/{mod_name}-{vid_name}-raw_data.csv")
            ETool = Evaluator(mod_name, vid_name)
            generate_plots(ETool.df_theta_gnd, vid_name, joints)
        else:
            logging(
                "WARNING",
                "No pose data and plot have been generated due to animation flag...",
            )

    # pose evaluation
    else:
        GT = GroundTruth(
            data,
            skp_rows=3,
            header_row_list=[0, 2, 3],
            start_frame=start_frame,
            end_frame=end_frame,
        )
        print(
            f"[{OKGREEN}GOOD{ENDC}]: Successfully extracted and compiled ground truth data"
        )

        # calibrate/get calibration
        N = 1000
        ETool = Evaluator(mod_name, vid_name)
        cal_file = f"{mod_name}-{vid_name}-cal_data.csv"
        cal_sim_data, cal_sim_stats = ETool.calibration_metric(
            window_size=1, num_experiments=N, sample_size=N
        )
        print(cal_sim_data)
        print(cal_sim_stats)

        sns.histplot(x=cal_sim_data.iloc[:, 0].values, stat='density')
        plt.title(f"Calibration Experiment Distribution N = {N}")
        plt.xlabel("X (mm)")
        plt.tight_layout()
        plt.savefig(f"output/cal_distr.png")
        plt.close()
        exit(0)

        if Path(f"output/{mod_name}/{cal_file}").is_file():
            df_cal = pd.read_csv(f"output/{mod_name}/{cal_file}", header=[0, 1, 2, 3], index_col=0)
        else:
            [MCAL_pos, mod_theta] = ETool.calibrate(N=1)

            CAL_data = {"theta": mod_theta, "pos": MCAL_pos, "quat": []}
            df_cal = data_parse(f"{mod_name}:<CAL>", CAL_data, joints)
            df_cal.to_csv(f"output/{mod_name}/{cal_file}")

        # MPJPE
        print(df_cal)
        mean_dists = ETool.MPJPE(df_cal.iloc[:, 1:])
        df_dists = pd.DataFrame({"MPJPE": mean_dists}, index=joint_list)

        # display MPJPE
        metrics_fname = f"output/{mod_name}/{mod_name}-{vid_name}-eval_metrics.csv"
        mpjpe_stats = df_dists.describe()
        mpjpe_stats.columns = ["Statistics"]
        print(df_dists)
        print(mpjpe_stats)

        # PDJ
        pdj = ETool.PDJ(df_cal.iloc[:, 1:], GT.torso_diam)
        df_pdj = pd.DataFrame({"PDJ": pdj}, index=joint_list)

        # display MPJPE
        pdj_stats = df_pdj.describe()
        pdj_stats.columns = ["Statistics"]
        print(df_pdj)
        print(pdj_stats)

        # save metrics
        df_metrics = pd.concat([df_dists, df_pdj], axis=1)
        df_stats = pd.concat([mpjpe_stats, pdj_stats], axis=1)
        df_metrics.to_csv(metrics_fname)
        df_stats.to_csv(metrics_fname, mode="a")

    logging("GOOD", "successful termination without error")


if __name__ == "__main__":
    # parse CLI arguments
    pcli, args = cli_parse()
    if None in (args.model, args.video):
        print("ERROR: must provide arguments for '-m' and '-v'\n")
        pcli.print_help()
        exit(1)

    init_dirs(args.model)
    main(
        model=args.model,
        video=args.video,
        data=args.data,
        start_frame=args.start,
        end_frame=args.end,
        animate=args.animate,
        eval=args.eval
    )
