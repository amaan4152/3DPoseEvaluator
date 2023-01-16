#!/bin/bash -i

eval ${ACTIONS}      # execute action cmds if provided (empty str if not)
${CMD} ${@:1}   # argv[1] is the string of flags for the model command

if [[ ${MODE} == "data" ]]; then
    EXT=${DATA_EXT}
    python3 reader.py ${OUTPUT_DIR}/*.${EXT} ${TMP_DIR}/${MODEL}
elif [[ ${MODE} == "animate" ]]; then
    EXT=${VIDEO_EXT}
else
    echo "ERROR: Invalid mode specified for ${MODEL}"
    exit 1
fi

# copy selected filetype outputs to temporary volume for later temporary access
find . -name '*.${EXT}' -exec cp --parents {} ${TMP_DIR}/${MODEL} \;

