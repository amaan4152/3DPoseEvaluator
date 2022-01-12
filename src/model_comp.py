import cv2
import numpy as np

gast_path = "GAST-Net-3DPoseEstimation/output/animation_GAIT_00_VIEWPORT_1.mp4"
vibe_path = "VIBE/output/GAIT_00_VIEWPORT_1/GAIT_00_VIEWPORT_1_vibe_result.mp4"

gast_cap = cv2.VideoCapture(gast_path)
vibe_cap = cv2.VideoCapture(vibe_path)

gs, gf = gast_cap.read()
vs, vf = vibe_cap.read()
if not (gs and vs):
    print("ERROR: Empty frame...")
    exit(-1)

f_concat = np.concatenate([gf, vf], axis=1)
w, h, _ = f_concat.shape
out_cap = cv2.VideoWriter(
    "models_output.mp4",
    cv2.VideoWriter_fourcc(*"MP4V"),
    np.mean([gast_cap.get(cv2.CAP_PROP_FPS), vibe_cap.get(cv2.CAP_PROP_FPS)]),
    (w, h),
)

# reset
gast_cap = cv2.VideoCapture(gast_path)
vibe_cap = cv2.VideoCapture(vibe_path)
FRAME_NUM = 100

while gast_cap.isOpened() and vibe_cap.isOpened():
    if FRAME_NUM >= 100:
        print("Finished processing...")
        break

    gs, gf = gast_cap.read()
    vs, vf = vibe_cap.read()
    if not (gs and vs):
        print("ERROR: Empty frame...")
        break

    cv2.putText(
        gf,
        "GAST-NET: STANDING",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        7,
        cv2.LINE_AA,
    )
    cv2.putText(
        vf,
        "GAST-NET: STANDING",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        7,
        cv2.LINE_AA,
    )
    f_concat = np.concatenate([gf, vf], axis=1)
    out_cap.write(f_concat)
    cv2.imshow("RESULT", f_concat)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    FRAME_NUM += 1

gast_cap.release()
vibe_cap.release()
out_cap.release()
cv2.destroyAllWindows()
