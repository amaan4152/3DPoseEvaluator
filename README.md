# NIST 3D Pose Estimation Evaluator

## Description
Generate 3D pose data from an arbitrary video sequence utilizing GAST-NET, VIBE, and Blazepose
as the 3D monocular pose estimation algorithms. Evaluate the 3D pose data against the ground-truth
OTS data to generate a comparison study. The 3D pose data is evaluated via the Mean Per Joint Position Error (MPJPE)
and Percent Joints Detected (PJD) to compare between the various 3D pose estimation algorithms. 

## Setup Procedure
Docker will be utilized to create a machine-agnostic configuration to run the evaluation tool. 
Install Docker: https://docs.docker.com/get-docker/

Basic Procedure (subject to change and further details will be provided): 
 - Clone this repo 
 - If you want your own Docker image, then build the Dockerfile to construct the image
 - Generate a container from the image and the evaluation tool will execute accordingly
