# NIST 3D Pose Estimation Evaluator

## Description
Generate 3D pose data from an arbitrary video sequence utilizing GAST-NET, VIBE, and Blazepose
as the 3D monocular pose estimation algorithms. Evaluate the 3D pose data against the ground-truth
OTS data to generate a comparison study. The 3D pose data is evaluated via the Mean Per Joint Position Error (MPJPE)
and Percent Joints Detected (PJD) to compare between the various 3D pose estimation algorithms. 

## Installation 
 1. Install docker into system: https://docs.docker.com/get-docker/<br>
    Check if docker CLI is available and ready: `docker --version`
 2. Clone this github repository
 3. A docker image is required to generate containers for pose estimation experiemnts. Thus, we are going to build our own image based on the Dockerfile in this  repository: <br>
    ```
    sudo docker build . --compress -t eval-tool/test:latest --target add-vibe
    ```
    Make sure to be inside this repository or else the Dockerfile will not be detected. The `--compress` tag compresses the image we are going to build. `-t` flag requires the name of the image we want to build, in this case the image name is **eval-tool/test:latest**; `--target` flag is to specify up to what layer in the Dockerfile we would like to build to, in this case it's **add-vibe**, which is the last sublayer in the Dockerfile. 
