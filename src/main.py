import argparse as ap
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from GroundTruth import GroundTruth

DATA_DIR = "/root/tmp"
OUTPUT_DIR = "/root/output"

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
        "-v",
        "--video",
        action="store",
        type=str,
        help="input video path to be processed",
    )
    pcli.add_argument(
        "-d", "--data", action="store", type=str, help="ground truth data"
    )
    pcli.add_argument(
        "-k",
        "--kinm_chain",
        action="store",
        type=str,
        help="kinematic chain of interest; consult 'joints_kinematic.yml' to select valid chains",
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


def generate_joint_angles_plot(df_joint_angles : pd.DataFrame, vid_name: str, kinematic_chain: str):
    joint_name = " ".join([s.upper() for s in kinematic_chain.split("-")])
    fname_joint_substr = "_".join(joint_name.split(" "))
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=df_joint_angles)
    plt.title(f"{joint_name} Joint Angle Plot")
    plt.xlabel("Frame ID")
    plt.ylabel(r"$\theta(^{\circ})$")
    plt.xlim(left=-10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/{vid_name}/PLOT-{fname_joint_substr}-joint_angle.png")
    plt.close()


def generate_gnd_truth(
    file: str, 
    vid_name : str,
    start_frame: int,
    end_frame: int,
    joints: dict,
    kinematic_chain: str 
):
    GT = GroundTruth(
        file=file,
        skp_rows=3,
        header_row_list=[0, 2, 3],
        start_frame=start_frame,
        end_frame=end_frame,
        joints=joints,
    )
    GT_pos, GT_quat, GT_theta = GT.get_joints()

    gt_data = {"theta": GT_theta, "pos": GT_pos, "quat": GT_quat}
    df_gt_raw = data_parse(
        model_name="GND_TRUTH", 
        model_data=gt_data, 
        kinematic_chain=kinematic_chain, 
        joints=joints
    )

    df_gt_raw.to_csv(f"{OUTPUT_DIR}/gnd_truth/{vid_name}/raw_data.csv")
    logging("GOOD", "Successfully extracted and compiled ground truth data")
    return df_gt_raw


import os
import yaml

from glob import glob
from pose_gen import data_parse, pose_gen
from Evaluator import Evaluator
from pathlib import Path


def main(
    video: str,
    truth_file: str,
    kinematic_chain: str,
    start_frame: int,
    end_frame: int,
    evaluate: bool = False,
    animate: bool = False,
):
    df_joint_angles_list = []
    vid_name = Path(video).name.lower().split(".")[0]

    # extract selected kinematic chain froom joints table
    with open("joints_kinematic_table.yml", "r") as stream:
        joints_cfg = yaml.load(stream=stream, Loader=yaml.SafeLoader)
    valid_chains = list(joints_cfg.keys())
    if kinematic_chain not in valid_chains:
        ValueError(
            f"Invalid kinematic chain selected. Current supported chains include: {valid_chains}"
        )
    joints = joints_cfg[kinematic_chain]
    joint_labels = list(joints.keys())

    # get model names from model directories in data-vol
    model_dirs = glob(pathname=f"{DATA_DIR}/*/", recursive=True)
    model_dirs = [m.split('/')[3] for m in model_dirs]

    # compile ground-truth data
    os.makedirs(f"{OUTPUT_DIR}/gnd_truth/{vid_name}", exist_ok=True)
    df_truth = generate_gnd_truth(
        file=truth_file,
        vid_name=vid_name,
        start_frame=start_frame,
        end_frame=end_frame,
        joints=joints,
        kinematic_chain=kinematic_chain
    )

    # iterate through each model data dir that has been executed
    for model in model_dirs:
        os.makedirs(f"{OUTPUT_DIR}/{model}/{vid_name}", exist_ok=True)
        if not evaluate:    # pose raw data generation
            raw_fname = f"{OUTPUT_DIR}/{model}/{vid_name}/raw_data.csv"
            df_model = pose_gen(
                model_type=model,
                joints=joints,
                kinematic_chain=kinematic_chain
            )
            logging("GOOD", f"Successfully extracted and compiled {model} data")
            df_model.to_csv(raw_fname)
            df_model_theta = df_model.iloc[:, 0].to_frame()
            df_model_theta.columns = [model]
            df_joint_angles_list.append(df_model_theta)
        elif animate:   # pose animation
            logging(
                "WARNING",
                "No pose data and plot have been generated due to animation flag...",
            )
        else: # pose evaluation
            logging("GOOD", f"{model.capitalize()} Evaluation")
            GT = GroundTruth(
                file=truth_file,
                skp_rows=3,
                header_row_list=[0, 2, 3],
                start_frame=start_frame,
                end_frame=end_frame,
                joints=joints,
            )

            # calibrate
            ETool = Evaluator(model=model, vid_name=vid_name)
            [MCAL_pos, mod_theta] = ETool.calibrate(N=1)

            # save calibration data
            cal_fname = f"{OUTPUT_DIR}/{model}/{vid_name}/cal_data.csv"
            CAL_data = {"theta": mod_theta, "pos": MCAL_pos, "quat": []}
            df_cal = data_parse(
                model_name=f"{model}:<CAL>",
                model_data=CAL_data,
                kinematic_chain=kinematic_chain,
                joints=joints
            )
            df_cal.to_csv(cal_fname)

            # MPJPE
            mean_dists = ETool.MPJPE(df_cal.iloc[:, 1:])
            df_dists = pd.DataFrame({"MPJPE": mean_dists}, index=joint_labels)

            # display MPJPE
            metrics_fname = f"{OUTPUT_DIR}/{model}/{vid_name}/eval_metrics.csv"
            mpjpe_stats = df_dists.describe()
            mpjpe_stats.columns = ["Statistics"]
            print(df_dists)
            print(mpjpe_stats)

            # PDJ
            pdj = ETool.PDJ(df_pos_cal=df_cal.iloc[:, 1:], torso_diam=GT.torso_diam)
            df_pdj = pd.DataFrame({"PDJ": pdj}, index=joint_labels)

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

    if not (evaluate or animate):
        # Any model can be used as reference for alignment based on the assumption that all models have the same # of frames
        df_truth_theta = Evaluator(
            df_gnd=df_truth, 
            df_model_ref=df_model
        ).df_theta_gnd  # performs ground-truth alignment assuming model FPS is 60 FPS
        df_truth_theta.columns = ["ground-truth"]   # label for plotting
        df_joint_angles_list.insert(0, df_truth_theta)

        os.makedirs(f"{OUTPUT_DIR}/plots/{vid_name}", exist_ok=True)
        df_joint_angles = pd.concat(df_joint_angles_list, axis=1)
        generate_joint_angles_plot(
            df_joint_angles=df_joint_angles,
            vid_name=vid_name,
            kinematic_chain=kinematic_chain
        )
    logging("GOOD", "successful termination without error")


if __name__ == "__main__":
    # parse CLI arguments
    _, args = cli_parse()
    if None in (args.video, args.data, args.kinm_chain, args.start, args.end):
        ValueError(
            "Must provide arguments for '--video', '--data', '--kinm_chain', '--start', and '--end'\n"
        )
    main(
        video=args.video,
        truth_file=args.data,
        kinematic_chain=args.kinm_chain,
        start_frame=args.start,
        end_frame=args.end,
        evaluate=args.eval,
        animate=args.animate,
    )
