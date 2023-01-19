# !/bin/bash -i

CMD="python3 main.py --video ${1} --data /root/data/${2} --kinm_chain ${3} --start ${4} --end ${5}"
if [[ $6 == "EVAL" ]]; then
    CMD="${CMD} --eval"
elif [[ $6 == "ANIMATE" ]]; then
    CMD="${CMD} --animate"
elif [ ! -z $6 ]; then
    echo "ERROR: Incorrect 5th argument specified. Supported values: '--eval' or '--animate'"
    exit 1
fi

# execute command
${CMD}
