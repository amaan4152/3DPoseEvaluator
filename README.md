# 3D Pose Estimation Evaluator

## Description
Generate 3D pose data from an arbitrary video sequence utilizing GAST-NET, VIBE, and Blazepose
as the 3D monocular pose estimation algorithms. Evaluate the 3D pose data against the ground-truth
OTS data to generate a comparison study. The 3D pose data is evaluated via the Mean Per Joint Position Error (MPJPE)
and Percent Joints Detected (PJD) to compare between the various 3D pose estimation algorithms. 

## Installation and Setup
GAIT analysis OTS and IMU data are in the data folder. 
 1. Install docker into system: https://docs.docker.com/get-docker/<br>
    Check if docker CLI is available and ready: `docker --version`
 2. Clone this github repository. Make sure to be at the top of the repository directory from here on out.
 3. Make 1 directories: 
    - `output`: directory that will contain all results after successful execution
    
 **_Note:_** store any videos for testing in the `videos` directory.
 
 4. The tree structure from the top and 2 levels deep should be as follows: 
    ```
    .
    ├── data
    │   └── GAIT_noexo_00.csv
    ├── output
    │   └── raw_data.csv
    ├── scripts
    │   └── init_pyenv.sh
    ├── src
    │   ├── __pycache__
    │   ├── models
    │   ├── cfg_joints.json
    │   ├── data_parser.py
    │   ├── edr.py
    │   ├── evaluate.py
    │   ├── exec_models.sh
    │   ├── gast-requirements.txt
    │   ├── main.py
    │   ├── model_analysis.py
    │   ├── pose_eval.py
    │   ├── pose_gen.py
    │   ├── truth_analysis.py
    │   └── vid_proc.py
    ├── videos
    │   └── SAMPLE_VIDEO_FILE
    ├── Dockerfile
    ├── Makefile
    ├── README.md
    ├── poetry.lock
    └── pyproject.toml

    ```
    - `raw_data.csv` will be inside the `output` directory after successful execution of the evaluator tool
    - `SAMPLE_VIDEO_FILE` represents any video file that will be used for analysis. There can be multiple videos, but only 1 video can be selected for analysis as can be seen in **step 6**
 5. A docker image is required to generate containers for pose estimation experiments. Thus, we are going to build our own image based on the Dockerfile in this  repository: <br>
    ```
    docker build . --compress -t eval-tool/test:latest --target add-vibe
    ```
    Make sure to be inside this repository or else the Dockerfile will not be detected. The `--compress` tag compresses the image we are going to build. `-t` flag requires the name of the image we want to build, in this case the image name is **eval-tool/test:latest**; `--target` flag is to specify up to what layer in the Dockerfile we would like to build to, in this case it's **add-vibe**, which is the last sublayer in the Dockerfile that will build everything. To check the existence of the image: `docker images`
 6. Given that the image has been built, execute the pose evaluator: 
    ```
    docker run --shm-size 10G --volume <abs path to working dir>:/home -it eval-tool/test:latest VIDEO=<video path> MODEL=VIBE
    ```
    This will generate an unamed container, mount the current working directory (you have to be inside the repository) to the container, and interactively interact with it providing it CLI arguments that are necessary for the pose evaluator to run. Here is the breakdown of the command:
    - `--shm-size <MEMORY SIZE>`: adjusts the amount of RAM allocated for the temporary filesystem that the container will use, adjust as necessary based on host machine specs. If there are memory errors that occur, increase the memory size. 
    - `--volume <host dir>:<target dir in container>`: attach host directory to a target container directory. Make sure `host dir` is absolute path of current working        directory; if on a Linux environment, use `pwd` to get absolute path of current working directory. 
    - `-it`: permit an interactive process within the terminal on execution of the container
    - `eval-tool/test`: image name to run a container from
    - `VIDEO=`: argument to specify video file to analyze. Specify relative path!
    - `MODEL=`: argument to specify model. Support models are: `GAST`, `VIBE`, `BLAZEPOSE`
    - `ANIMATION=True`: **optional** argument to enable saving `MODEL` pose into video file in `output` directory


Extra flags of interest (all flags have to be after `docker run` and preceed the name of the image): 
 - `--named <container name>`: provide name of container to reaccess it again. **_Note:_** If you would like to debug inside the docker container, make sure to **comment** the last line of the `Dockerfile` and add `bash` at the end of command `6`.   
 - `--rm`: remove the container when its job is done 
 - `--gpus all`: given that you have a supported Linux distro, use this flag to utilize GPU compute capabilities. Setup guide for NVIDIA container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

An example run of **step 6** with the example scenario in **step 4** without GPU compute and removing the container after execution:

    docker run --rm --shm-size 10G --volume /home/users/student/3DPoseEvalulator:/home -it eval-tool/test:latest VIDEO=videos/SAMPLE_VIDEO_FILE MODEL=VIBE
    
 7. A file named `{model}-{video_name}-raw_data.csv` will pop up in the `output` directory for the respective `model` that has been executed for the provided `video_name`. The columns represent the model's pose data. The data tag order is as follows: 
    1. `THETA`: joint angle
    2. `POS-X`: x-coordinate
    3. `POS-Y`: y-coordinate
    4. `POS-Z`: z-coordinate
 
    A file named `{PIVOT_JOINT}_JOINTANGLE_PLOT.png` will also be generated with the plot of the joint angle data for all executed algorithms with respect to the appropriate `PIVOT_JOINT` (e.g. _RKNEE_ for joint angle with right knee as vertex or pivot of joint angle). 
    
    **_Note:_** this tool works for tracking the right leg at the moment.
## Future Work
- Comparison study between 3D pose data and Optical Tracking System (OTS) data
- Align 3D pose data with OTS 
- Calibrate 3D pose data with OTS
- Compute 3D positional errors between model and OTS, and compute MPJPE and PJD metrics
