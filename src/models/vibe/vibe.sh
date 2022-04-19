# provide absolute path for video file!
printf "\n ----- \e[3mInitiating\e[0;1;96m VIBE \e[0m----- \n\n"

VNAME=$1
cd "/root/VIBE/" || exit 1
pyenv activate vibe-env # python virtualenv for VIBE
# for docker: had to decrease tracker_batch_size (12 -> 8) due to mem issues
python3 demo.py --vid_file "${VNAME}" --output_folder "output/" \
                --no_render --tracker_batch_size 1 --tracker_batch_size 2 \
                --vibe_batch_size 64

VNAME="$(basename ${VNAME})"
PKL_PATH=$(realpath "./output/${VNAME//".mp4"}/vibe_output.pkl")
echo "$PKL_PATH"
