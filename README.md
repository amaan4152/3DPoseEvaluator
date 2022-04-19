# NIST 3D Pose Estimation Evaluator

## Description
Generate 3D pose data from an arbitrary video sequence utilizing GAST-NET, VIBE, and Blazepose
as the 3D monocular pose estimation algorithms. Evaluate the 3D pose data against the ground-truth
OTS data to generate a comparison study. The 3D pose data is evaluated via the Mean Per Joint Position Error (MPJPE)
and Percent Joints Detected (PJD) to compare between the various 3D pose estimation algorithms. 

## Installation 
GAIT analysis OTS and IMU data provided by NIST are in the data folder. 
 1. Install docker into system: https://docs.docker.com/get-docker/<br>
    Check if docker CLI is available and ready: `docker --version`
 2. Clone this github repository. Create an **output** folder inside the repository. If you want, you could also create a video directory to store all your necessary video files for analysis.
 3. A docker image is required to generate containers for pose estimation experiemnts. Thus, we are going to build our own image based on the Dockerfile in this  repository: <br>
    ```
    docker build . --compress -t eval-tool/test:latest --target add-vibe
    ```
    Make sure to be inside this repository or else the Dockerfile will not be detected. The `--compress` tag compresses the image we are going to build. `-t` flag requires the name of the image we want to build, in this case the image name is **eval-tool/test:latest**; `--target` flag is to specify up to what layer in the Dockerfile we would like to build to, in this case it's **add-vibe**, which is the last sublayer in the Dockerfile. To check the existanc of the image: `docker images`
 4. Given that the image has been built, execute the pose evaluator: 
    ```
    docker run --shm-size 10G --volume <abs path to working dir>:/home -it eval-tool/test VIDEO=<video path> MODEL=VIBE START=<start frame> END=<end frame>
    ```
    This will generate an unamed container, mount the current working directory (you have to be inside the repository) to the container, and interactively interact with it providing it CLI arguments that are necessary for the pose evaluator to run. Here is the breakdown of the command:
    - `--shm-size <MEMORY SIZE>`: adjusts the amount of RAM allocated for the temporary filesystem that the container will use, adjust as necessary based on host machine specs. If there are memory errors that occur, increase the memory size. 
    - `--volume <host dir>:<target dir in container>`: attach host directory to a target container directory
    - `-it`: permit an interactive process within the terminal on execution of the container
    - `eval-tool/test`: image name to run a container from
    - `VIDEO=`: argument to specify video file to analyze. **Must** be absolute path to video file or the program will fail
    - `MODEL=`: argument to specify model. Support models are: `GAST`, `VIBE`, `BLAZEPOSE`
    - `START=`: argument to specify the frame to start analysis (must know prior to execution of pose evaluator)
    - `END=`: argument to specify the frame to end analysis (must know prior to execution of pose evaluator)


Extra flags of interest (all flags have to be after `docker run` and preceed the name of the image: 
 - `--rm`: remove the container when its job is done 
 - `--gpus all`: given that you have a supported Linux distro, use this flag to utilize GPU compute capabilities. Setup guide for NVIDIA container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
    
