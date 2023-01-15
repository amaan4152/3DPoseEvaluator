import re
import subprocess as sp
import sys
import yaml

from pathlib import Path
from typing import List

MODELS_DIR = "/root/models"
MODEL = sys.argv[1].split("/")[1]
MODE = sys.argv[2]


def flags_to_str(raw_flags: list) -> str:
    flags_list = []
    for f in raw_flags:
        if isinstance(f, dict):
            f = " ".join([str(*f.keys()), str(*f.values())])
        flags_list.append(f)
    
    flags_str = " ".join(flags_list)
    return flags_str


def compile_flags(configs : dict) -> str:
    flags = configs["flags"]
    if flags is None:                                           # set flags to empty list if primary flags don't exist
        flags = []

    # enable/disable animation
    set_animation = "enable" if MODE == "animate" else "disable"
    
    if configs["animate-opt"] is None:                        # set enable/disable animation flag to empty if it doesn't exist
        anim_flag = ""
    else:
        anim_flag = configs["animate-opt"][set_animation]
    if flags is None and configs["animate-opt"] is None:      # set flags to empty if there are no flags and enable/disable animation flags
        flags = ""
    elif configs["animate-opt"][set_animation] is not None:   # add flag for enabled/disabled animation if it exists
        anim_flag = configs["animate-opt"][set_animation]
        flags.append(anim_flag)
        flags = flags_to_str(flags)

    return flags


def create_dockerfile(configs : dict) -> List[str]:
    py_ver = configs["python-version"]
    model_name = configs["name"]
    cmd = configs["cmd"]
    output_dir = configs["output-dir"]
    data_ext = configs["filetypes"]["data"]
    video_ext = configs["filetypes"]["video"]
    

    flags = compile_flags(configs=configs)

    dockerfile_contents = [
        f"FROM python:{py_ver}-slim as base",
        "ENV DEBIAN_FRONTEND=noninteractive",
        f"ENV MODEL={model_name}",
        f"ENV CMD=\"{cmd}\"",
        f"ENV OUTPUT_DIR=/root/{model_name}/{output_dir}",
        f"ENV DATA_EXT={data_ext}",
        f"ENV VIDEO_EXT={video_ext}",
        "ENV TMP_DIR=/root/tmp",
        f"ENV MODE={MODE}"
    ]
    if configs["actions"] is not None:
        actions = "; ".join(configs["actions"])
        dockerfile_contents.append(f"ENV ACTIONS=\"{actions}\"")
    if Path(f"models/{MODEL}/requirements.txt").exists():
        dockerfile_contents.append("\nCOPY requirements.txt /")
    dockerfile_contents.extend(
        [
            "SHELL [ \"/bin/bash\", \"-c\" ]",
            "\nCOPY setup.sh /",
            "RUN bash /setup.sh",
            "RUN mkdir -p /root/${MODEL} ${TMP_DIR}/${MODEL} ${OUTPUT_DIR}",
            "WORKDIR /root/${MODEL}",
            "COPY *.py *.sh ./",                    # this does recopy setup.sh, but it will not be executed since it's for executing ACTION scripts
            f"\nENTRYPOINT [ \"bash\", \"run.sh\", \"{flags}\" ]"
        ]
    )
    return dockerfile_contents


def main():
    stream = open(f"models/{MODEL}/config.yml", "r")
    configs = yaml.load(stream=stream, Loader=yaml.SafeLoader)
    dir_name = configs["dir"]
    if dir_name != MODEL:
        ValueError("`dir` field in config.yml must match parent directory name")

    # create model dir
    sp.run(args=f"mkdir -p {MODELS_DIR}/{dir_name}", shell=True)

    # populate model dir with necessary scripts
    dockerfile = f"{MODELS_DIR}/{dir_name}/Dockerfile"
    if not Path(dockerfile).exists():
        dockerfile_contents = create_dockerfile(configs=configs)
        with open(dockerfile, "w+") as file:
            file.write("\n".join(dockerfile_contents))
        sp.run(args=f"cp model_run.sh {MODELS_DIR}/{dir_name}/run.sh", shell=True)
        sp.run(args=f"mv models/{MODEL}/*.sh {MODELS_DIR}/{dir_name}/", shell=True)
        sp.run(args=f"mv models/{MODEL}/*.py {MODELS_DIR}/{dir_name}/", shell=True)
        if Path(f"models/{MODEL}/requirements.txt").exists():
            sp.run(args=f"mv models/{MODEL}/requirements.txt {MODELS_DIR}/{dir_name}/", shell=True)
    else:
        # update model cmd flags if spawner is rerun
        curr_flags = compile_flags(configs=configs)
        with open(dockerfile, "r") as file:
            contents = file.read()
        updated_contents = re.sub(r'(ENTRYPOINT \[ \"bash\", \"run\.sh\", ).*( \])', rf'\1"{curr_flags}"\2', contents)
        with open(dockerfile, "w+") as file:
            file.write(updated_contents)
    stream.close()
    

if __name__ == "__main__":
    main()