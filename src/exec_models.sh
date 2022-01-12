#!/bin/bash -i

RED="\e[1;31m"
NORMAL="\e[0;39m"

# PRECONDITION:  - Video file exists and in correct directory for appropriate model to access
#                - All model directories 
# FORMAT: ./exec_models.sh [MODEL_NAME] [VIDEO_FILE]
# Iterate through arguments -> extract model type -> acquire pose data file
for ARG in "$@"
do
    VNAME="$2"
    POSE_NUM=1 # detect only 1 subject (for GAST-NET)

    case $ARG in 
        VIBE)
            printf "\n ----- \e[3mInitiating\e[0;1;96m VIBE \e[0m----- \n\n"

            cd "./VIBE/" || exit 1
            source "$(pwd)/vibe-env/bin/activate"  # enter python virtual environment for VIBE
            python3 demo.py --vid_file "../${VNAME}" --output_folder "output/"            
            PKL_PATH=$(realpath "./output/${VNAME//".mp4"}/vibe_output.pkl")
            echo "$PKL_PATH"
            deactivate
            ;;

        GAST)
            printf "\n ----- \e[3mInitiating\e[0;1;35m GAST \e[0m----- \n\n"
            cd "./GAST-Net-3DPoseEstimation/" || exit 1
            pyenv activate gast_env
            python3 gen_skes.py -v "$VNAME" -np "$POSE_NUM"
            NPZ_PATH=$(find "./output/" -name "${VNAME//".mp4"/}.npz")
            NPZ_PATH=$(realpath "$NPZ_PATH")
            echo "$NPZ_PATH"
            ;;

        *)
            echo -e "${RED}ERROR$NORMAL: Invalid argument passed"    
            exit 1     
            ;;
    esac
done
