from pose_gen import pose_gen
from data_parser import data_parse2
from data_parser import cli_parse
from truth_analysis import get_OTSData
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


def main():
    pcli, args = cli_parse()
    if None in (args.model, args.video, args.data):
        print("ERROR: must provide arguments for '-m' and '-v'\n")
        pcli.print_help()
        exit(1)

    joints = {"KNEE": ["HIP", "KNEE", "ANKLE"]}

    # generate raw data
    duration = get_duration(args.video)
    print(f"Duration: {duration}")
    df_raw = pose_gen(
        args.video, args.data, args.model, args.start, args.end, joints, duration
    )
    df_raw.to_csv("output/raw_data.csv")


if __name__ == "__main__":
    main()
