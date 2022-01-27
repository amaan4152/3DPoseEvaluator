import cv2
import moviepy as mp
import numpy as np
from tqdm import tqdm


class VidProc:
    """
    AMAAN RAHMAN

    This script generates output video files within any range of frames
    for multiple viewports or input video file.

    Press ENTER/SPACEBAR when done selecting your ROI. After selecting multiple
    ROIs, press ESC to proceed with processing output video files within each ROI viewport.

    REMARKS: Bugs with QObject threading or other inconsistent bugs could be possible depending on
    your opencv-python pip version, check the requirements file regarding this remark.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    bLCornerText = (10, 500)
    fScale = 1
    fColor = (255, 255, 255)
    lineThickness = 2

    video_path = ""
    output_videos = []
    ROIs = []  # list of rectangular ROIs
    mod_frames = []  # list of cropped frames
    out_caps = []  # list of output video captures
    START = 3350
    TOTAL = 12279
    duration = 0

    # -1 frame number implies play through of rest of video from start frame
    def __init__(self, *args):
        if len(args) >= 1:
            self.cap = cv2.VideoCapture(args[0])
            self.video_path = args[0]
            self.out_fname = args[0].split("/")[-1].split(".")[0]

            if len(args) == 1:  # only video path provided
                self.start_frame = self.START
                self.final_frame = self.START + self.TOTAL - 1
            elif len(args) == 2:  # video path and start frame number provided
                self.start_frame = int(args[1]) + self.START
                self.final_frame = self.START + self.TOTAL - 1
            elif (
                len(args) == 3
            ):  # video path, start frame num, and final frame num provided
                self.start_frame = int(args[1]) + self.START
                self.final_frame = int(args[2]) + self.START
        else:
            print("ERROR: Invlaid number of arguments provided in constructor")
            exit(-1)

    def close(self):
        self.cap.release()
        if self.out_caps:  # release all output video capture objects
            for o_cap in self.out_caps:
                o_cap.release()
        else:
            self.out_cap.release()
        cv2.destroyAllWindows()

    # generate list of cropped frames given a list of rectangular ROIs
    def ROI_Processing(self, frame):
        cropped_frames = []
        for r in self.ROIs:
            x1 = int(r[0])
            x2 = int(r[2])
            y1 = int(r[1])
            y2 = int(r[3])

            cropped_frame = frame[y1 : y1 + y2, x1 : x1 + x2]
            cropped_frames.append(cropped_frame)

        return cropped_frames

    # generate output video file(s) (functionality depends on whether input video file is a quad video or not)
    def gen_OutVideo(self, isQuad):
        if self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("ERROR: Empty frame detected, closing program...")
                exit(-1)

            frame = np.copy(frame)  # copy frame for safetys
            if isQuad:
                self.ROIs = cv2.selectROIs(
                    "Select Regions of Interests", frame, showCrosshair=False
                )  # select ROIs
                self.mod_frames = self.ROI_Processing(frame)

                viewport_num = 1

                # given quad input video, generate output video files per selected ROI by the user
                for f in self.mod_frames:
                    f_h, f_w, _ = f.shape
                    out_name = ""
                    (self.out_fname + "_VIEWPORT_" + str(viewport_num) + ".mp4")
                    self.out_caps.append(
                        cv2.VideoWriter(
                            out_name,
                            cv2.VideoWriter_fourcc(*"MP4V"),
                            self.cap.get(cv2.CAP_PROP_FPS),
                            (f_w, f_h),
                        )
                    )
                    self.output_videos.append(out_name)
                    viewport_num += 1

            else:  # when input video is not a quad
                frame_height, frame_width, _ = frame.shape
                self.out_cap = cv2.VideoWriter(
                    self.out_fname,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    self.cap.get(cv2.CAP_PROP_FPS),
                    (frame_width, frame_height),
                )

            self.cap.open(
                self.video_path
            )  # reinitialize video capture object for reset
            cv2.destroyAllWindows()
            print("<MESSAGE>: Output video file(s) generated...")

    def ViewportProcessing(self, isQuad):
        print(
            "\nKeypress Instructions:\n\tSPACEBAR = Pause video and record current frame number\n\tESC = Exit video sequence\n"
        )

        total_frames = (
            (self.START + self.TOTAL - 1)
            if self.final_frame == -1
            else self.final_frame
        )
        print(total_frames)
        pbar = tqdm(total=total_frames, ncols=100)
        pbar.set_description("Frame # Progress: ")

        frame_num = 0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("ERROR: Empty frame detected")
                break

            frame = np.copy(frame)

            # if a start frame number specified, block all processing till start frame number is reached
            if frame_num < self.start_frame:
                pbar.update(1)
                frame_num += 1
                continue

            # if a final frame number specified, exit processing
            if frame_num == self.final_frame:
                break

            if isQuad:
                if not self.mod_frames:
                    print(
                        "ERROR: No ROIs detected! Your intention might have been to process a non-quad video..."
                    )
                    exit(-1)

                self.mod_frames = self.ROI_Processing(
                    frame
                )  # generate cropped frames from list of ROIs

                # PROGRESS: Finish generating multiple output video files per ROI
                viewport_num = 0

                # for each cropped frame write it to its respective output video file and generate a window with frame counter
                for f in self.mod_frames:
                    cv2.putText(
                        f,
                        str(frame_num) + " | " + str((frame_num - self.START) / 60),
                        self.bLCornerText,
                        self.font,
                        self.fScale,
                        self.fColor,
                        self.lineThickness,
                        cv2.LINE_AA,
                    )
                    self.out_caps[viewport_num].write(f)
                    cv2.imshow(
                        self.out_fname + "_VIEWPORT_" + str(viewport_num + 1) + ".mp4",
                        f,
                    )
                    viewport_num += 1
            else:
                cv2.putText(
                    frame,
                    str(frame_num),
                    self.bLCornerText,
                    self.font,
                    self.fScale,
                    self.fColor,
                    self.lineThickness,
                    cv2.LINE_AA,
                )
                self.out_cap.write(frame)
                cv2.imshow(self.out_fname.replace("_RES.mp4", ""), frame)

            keyPress = cv2.waitKey(1) & 0xFF
            """
            if keyPress == 32: #SPACEBAR key pressed => PAUSE video sequence
                if cv2.waitKey(-1) & 0xFF == 27: #ESC key pressed
                    print("Current frame number: " + str(frame_num)) 
                    break
                print("Current frame number: " + str(frame_num)) 
            """
            if keyPress == 27:  # ESC key pressed
                print("Current frame number: " + str(frame_num))
                break

            frame_num += 1
            pbar.update(1)

        vid = mp.editor.VideoFileClip(self.out_fname + "_VIEWPORT_1" + ".mp4")
        self.duration = int(vid.duration)
        pbar.close()
