#!/bin/bash

MODE=$1

for DIRNAME in models/*/ ; do
    python3 main.py ${DIRNAME} ${MODE}
done
