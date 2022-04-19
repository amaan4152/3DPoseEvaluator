printf "\n ----- \e[3mInitiating\e[0;1;35m GAST \e[0m----- \n\n"
cd "/root/GAST-Net-3DPoseEstimation/" || exit 1

POSE_NUM=1  # number of poses to track
VNAME=$1
pyenv activate gast-env
python3 gen_skes.py -v "${VNAME}" -np "$POSE_NUM"
NPZ_PATH=$(find "./output/" -name "${VNAME//".mp4"/}.npz")
NPZ_PATH=$(realpath "$NPZ_PATH")
echo "$NPZ_PATH"
