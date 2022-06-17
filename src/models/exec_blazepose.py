import argparse as ap
import cv2
from cv2 import (
    VideoCapture,
    cvtColor,
    COLOR_BGR2RGB,
    VideoWriter,
    VideoWriter_fourcc,
)
from cv2 import COLOR_RGB2BGR
import numpy as np
import mediapipe as mp
import os
import shutil
from tqdm import tqdm

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


def cli_parse():
    pcli = ap.ArgumentParser(
        description="Evaluate GAST-NET, VIBE, and 3DMPPE 3D pose estimation algorithms on NIST video files to determine error metrics, \
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
        "--animate",
        action="store_true",
        help="generate pose animation files for specified algorithm",
    )

    return pcli, pcli.parse_args()


def init_dir(path, model_name):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        shutil.copy(__file__, path)
        os.makedirs(f"{path}/output")
        logging("WARNING", f"Initialized directory of {model_name}")


def exec_blazepose(vid_path, save_out=False):
    # init mediapipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # input video capture source
    print(f"Input video path: {vid_path}")
    cap = VideoCapture(vid_path)

    # capture output if specified in CLI
    o_cap = None
    if save_out:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        o_cap = VideoWriter(
            "output/blazepose_video.mp4", VideoWriter_fourcc(*"mp4v"), 30, (w, h)
        )

    # init progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, ncols=100)
    pbar.set_description("Frame # Progress: ")
    frame_num = 1

    bp_data = []
    with mp_pose.Pose(
        min_detection_confidence=0.75, min_tracking_confidence=0.95, model_complexity=2
    ) as pose:
        while cap.isOpened() and frame_num <= total_frames:
            s, img = cap.read()
            if not s:
                print()
                logging("ERROR", f"missing/broken frame # {frame_num}")
                exit(1)

            img.flags.writeable = False
            img = cvtColor(img, COLOR_BGR2RGB)
            results = pose.process(img)
            landmark_data = results.pose_landmarks.landmark

            frame_data = []
            for joint in landmark_data:
                frame_data.append([joint.x, joint.y, joint.y])
            bp_data.append(frame_data)

            if o_cap != None:
                # Draw the pose annotation on the image.
                img.flags.writeable = True
                img = cvtColor(img, COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                o_cap.write(img)

            pbar.update(1)
            frame_num += 1

    pbar.close()
    cap.release()
    if o_cap != None:
        o_cap.release()

    output_file = "output/blazepose_data.npz"
    bp_data = np.array(bp_data)
    logging("WARNING", f"saving Blazepose data to {output_file}...")
    np.savez_compressed(output_file, pose_data=bp_data)
    logging("GOOD", f"successfully saved file")


def main():
    model_name = "Blazepose"
    _, arg = cli_parse()
    init_dir(f"/root/{model_name}", model_name)
    exec_blazepose(arg.video, arg.animate)


if __name__ == "__main__":
    main()
