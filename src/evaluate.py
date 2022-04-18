from matplotlib.lines import Line2D
import moviepy.editor as mp
import numpy as np
import pandas as pd
from model_analysis import get_poseData
from truth_analysis import get_OTSData
from data_parser import cli_parse, data_parse
from vid_proc import VidProc
from seaborn import lineplot, set_style, set
import matplotlib.pyplot as plt
import os

DURATION = 0


def getVideos(vid, start_frame, end_frame):
    cont = input("Skip video processing?[Y/n] ")
    vp = VidProc(vid, start_frame, end_frame)
    if cont in ("N", "n"):
        print("\n========== VIDEO PROCESSING UNIT (START) ==========")
        vp.gen_OutVideo(True)
        vp.ViewportProcessing(True)
        vp.close()
        print("========== VIDEO PROCESSING UNIT (END) ==========\n")

    # get video files
    vid_dir = "videos/"
    wdir_list = os.listdir(
        vid_dir
    )  # possible search a reserved dir (created by vid_proc.py) of output video files?
    wdir_list = [file for file in wdir_list if "VIEWPORT" in file]
    wdir_list.sort(
        key=lambda f: int(f[(f.rindex("VIEWPORT_") + len("VIEWPORT_")) : f.index(".")])
    )
    print("\nCURRENT OUTPUT VIDEO FILES (from video processing unit):\n")
    print(*wdir_list, sep="\n")
    print(
        "\nInput format: # # # ... (# = viewport number; spaces imply multiple videos to be used)"
    )

    while True:
        # viewport #s are like unique keys associated with each output video file
        vid_selec = input(
            "Select which generated video file(s) to evaluate from the list above or provide your own (only one of your own): "
        )
        vid_selec_l = [f"VIEWPORT_{i}" for i in vid_selec.split() if i.isdigit()]
        if vid_selec_l == []:
            break

        vid_selec = [list(filter(lambda s: i in s, wdir_list)) for i in vid_selec_l]
        print(vid_selec[0])
        vid = mp.VideoFileClip(f"{vid_dir}{vid_selec[0][0]}")
        duration = int(vid.duration)
        if [] not in vid_selec:
            break

        print("ERROR: Incorrect video selection...")

    return duration, vid_selec


def main():
    pcli, args = cli_parse()
    if None in (args.model, args.video, args.data):
        print("ERROR: must provide arguments for '-m' and '-v'\n")
        pcli.print_help()
        exit(1)

    DURATION, vid_selec = getVideos(args.video, args.start, args.end)
    print("[CHECKPOINT-ID:01] -> Video file(s) acquired\n")

    if args.model in ("all", "ALL", "aLL", "All"):
        model = ["GAST", "VIBE"]
    else:
        model = [args.model]
    OTS_pos, OTS_quat, OTS_theta, skpd_frames, torso_width, frame_stat = get_OTSData(
        args.data, args.test, args.start, DURATION
    )
    ots = (OTS_theta, OTS_pos, OTS_quat)
    print("[CHECKPOINT-ID:02] -> OTS data extracted appropriately\n")

    # extract the data
    mpjpe = {}
    pjd = {}
    df, ups_df_mod = None, []
    for vid in vid_selec:
        vid = vid[0]
        for m in model:
            MODEL_pos, MODEL_theta, MODEL_quat = get_poseData(m, vid, frame_stat)
            [MODEL_theta.pop(val) for val in skpd_frames if val in MODEL_theta.keys()]
            [MODEL_quat.pop(val) for val in skpd_frames if val in MODEL_theta.keys()]

            for i in MODEL_pos:
                [
                    MODEL_pos[i].pop(val)
                    for val in skpd_frames
                    if val in MODEL_theta.keys()
                ]

            eval_data = data_parse(
                df,
                ots,
                (MODEL_theta, MODEL_pos, MODEL_quat),
                (
                    m
                    + " vp:"
                    + vid[(vid.rindex("VIEWPORT_") + len("VIEWPORT_")) : vid.index(".")]
                ),
                torso_width,
                True,
            )

            if eval_data is not None:
                df, ups_mod, mpjpe[m], pjd[m] = eval_data
                ups_df_mod.append(ups_mod)

            ots = None

    print(ups_df_mod)

    # compile the data
    mpjpe = pd.DataFrame.from_dict(mpjpe)
    pjd = pd.DataFrame.from_dict(pjd)
    metrics = pd.concat([mpjpe, pjd], axis=1)
    df.to_csv("output/results.csv", index=False)

    print(df.describe().T)
    print(metrics.describe().T)
    if skpd_frames != []:
        print(
            "\nWARNING: ["
            + str(len(skpd_frames))
            + "/"
            + str(int(args.end) - int(args.start) + 1)
            + "] Frames have been skipped to remove missing data!"
        )

    print("[CHECKPOINT-ID:03]{SUCCESS} -> Model(s) evaluation and data extraction\n")

    # extract angle data and angle error data
    angle_data = [
        df.iloc[:, col]
        for col in range(0, len(df.columns))
        if "ANGLE" in df.iloc[:, col].name
    ]
    err_data = [
        df.iloc[:, col]
        for col in range(0, len(df.columns))
        if "THETA-err" in df.iloc[:, col].name
    ]
    print(np.array(angle_data).shape)
    angle_data = [*angle_data, *err_data]

    # color configuration
    color_keys = [df.name for df in angle_data]
    color_vals = [
        "red",  # OTS joint angle
        "purple",  # GAST joint angle
        "forestgreen",
    ]  # VIBE joint angle

    mean_colors = ["fuchsia", "lime"]  # GAST mean error  # VIBE mean error

    color_dict = dict(zip(color_keys, color_vals))
    err_color_keys = [df.name for df in err_data]
    err_color_vals = [color for color in color_vals[1:]]  # get all model colors
    err_color_dict = dict(zip(err_color_keys, err_color_vals))

    # get mean data
    mean_err = []
    for err in err_data:
        mean_err.append(err.mean())

    # begin plotting
    if len(model) > 1:
        title = ", ".join(model)
    else:
        title = args.model

    set_style("darkgrid")
    set(font_scale=1.5)

    fig, (ax_data, ax_error) = plt.subplots(nrows=2, ncols=1, figsize=(23, 16))
    plt.setp((ax_data, ax_error), xticks=np.arange(0, df.shape[0], 500))

    ax_dat = lineplot(
        data=pd.concat(angle_data[0 : (1 + len(model))], axis=1),
        palette=color_dict,
        dashes=False,
        ax=ax_data,
        legend=False,
        linewidth=2.5,
    )
    ax_dat.set(
        xlabel="Frame Number",
        ylabel="Joint Angle (Degrees)",
        title=(title + " & OTS Joint Angle Data"),
    )

    # lineplot(x=GAST_upsampled_frames[0], y=upsampled_GAST_JANGLE[0], ax=ax_data, linewidth=2.5)
    ax_err = lineplot(
        data=angle_data[1 + len(model) :],
        ax=ax_error,
        palette=err_color_dict,
        legend=False,
        dashes=False,
        linewidth=2.5,
    )
    ax_err.set(
        xlabel="Frame Number",
        ylabel="Joint Angle Error (Degrees)",
        title=(title + " Joint Angle Errors"),
    )

    for i in range(0, len(mean_err)):
        ax_error.axhline(
            mean_err[i],
            color=mean_colors[i],
            linestyle=":",
            linewidth=2.5,
            label=model[i] + " Mean Error",
        )

    legend_elements = [
        Line2D([0], [0], color="red", label="OTS", linewidth=2.5),
        Line2D([0], [0], color="purple", label="GAST", linewidth=2.5),
        Line2D([0], [0], color="forestgreen", label="VIBE", linewidth=2.5),
        Line2D(
            [0],
            [0],
            color="fuchsia",
            label="GAST Mean Error",
            linestyle=":",
            linewidth=2.5,
        ),
        Line2D(
            [0],
            [0],
            color="lime",
            label="VIBE Mean Error",
            linestyle=":",
            linewidth=2.5,
        ),
    ]

    fig.legend(handles=legend_elements, bbox_to_anchor=(0.99, 0.6), prop={"size": 18})
    plt.subplots_adjust(top=0.92, right=0.8, hspace=0.5, bottom=0.08)

    plt.tight_layout()
    plt.savefig("output/graph.png")

    return 1


if __name__ == "__main__":
    main()
