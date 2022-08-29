import importlib as il
import json
from pathlib import Path
from shutil import copyfile
import subprocess as sp
from utils.diagnostic_tools import timeit


class ModelRegistry:
    src = "./src/models"

    # https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    def __init__(self, diagnostics=False):
        self._diagnostics = True
        with open(f"{self.src}/cfg_models.json", "r") as cfg_model_file:
            self.models = json.load(cfg_model_file)

    def reg_model(self):
        cfg_key_prompts_map = {
            "cmd": "Main command and script: ",
            "vid_flag": "Video Flag: ",
            "flags": "Flags (including video flag and arguments): ",
            "outputs": "Output Directory & File Types: ",
        }
        print(
            "\
            \n\t\t\t\t  ===== Model Registration =====\n\n\
            \tRequirements \n\
            \t------------\n\
            \tcmd:      source command and script (e.g. 'python3 gen_skes.py')\n\n\
            \tvid_flag: specify video flag syntax (with dash(s))\n\n\
            \tflags:    specify flags with arguments. If arguments are known,\n\
            \t          then add space followed by argument; else, just provide\n\
            \t          flag if argument is unkown. Sperate each set of flags\n\
            \t          and the associated arguments (if known) by comma\n\n\
            \toutputs:  target output directory, all output data file extentions, \n\
            \t          all output data video file extentions (delimited by commas)\n\
        "
        )

        # generate raw format of model registry
        model_name = input("Model GitHub Repository Name: ")
        self.models[model_name] = {
            k: input(p) for (k, p) in cfg_key_prompts_map.items()
        }

        # convert user flags input into organized dictionary of flags and arguments
        f_sets = self.models[model_name]["flags"].split(",")
        flags = [f.split()[0] for f in f_sets]
        args = [f.split()[1] for f in f_sets if len(f_sets.split()) > 1]
        self.models[model_name]["flags"] = dict(zip(flags, args))

        # convert outputs information into list
        self.models[model_name]["outputs"] = self.models[model_name]["outputs"].split(
            ","
        )  # commad-delimited-string -> list

        # update JSON config model config file
        with open(f"{self.src}/cfg_models.json", "w") as cfg_file:
            json.dump(self.models, cfg_file)

        # reload model configurations
        self.__init__()

    @timeit
    def exec_model(self, name: str, video: str, animate: bool):
        """
        Execute model `name` on `video`.

        Precondition
        ------------
        Model configured apprropriately in `cfg_models.json` and initialization encoded in `Dockerfile`

        Parameters
        ----------
        `name`:  model name
        `video`: input video file
        """
        print(f"\n\t\t\t\t  ===== {self.OKCYAN}{name.upper()}{self.ENDC} =====\n")
        path_type = self.models[name]["video_path_type"]
        vid_name = Path(video).name
        if path_type == "absolute":
            video = Path(video).absolute()
        elif path_type == "relative":
            video = Path(video).relative_to(
                Path(f"/home/{self.models[name]['dirname']}")
            )
        elif path_type == "name":
            video = vid_name

        if animate:
            a_flag = self.models[name]["animation_flag"]
            no_a_flag = self.models[name]["no_animation_flag"]
            if a_flag != "":
                self.models[name]["flags"][a_flag] = ""
                if no_a_flag != "" and no_a_flag in self.models[name]["flags"].keys():
                    self.models[name]["flags"].pop(no_a_flag)

            else:
                if no_a_flag != "" and no_a_flag in self.models[name]["flags"].keys():
                    self.models[name]["flags"].pop(no_a_flag)
        cmd = self.__make_cmd(name, video)
        proc = sp.Popen(
            cmd,
            cwd=f"/root/{self.models[name]['dirname']}",
            stdout=sp.PIPE,
            shell=True,
            executable="/bin/bash",
        )  # default executable arg is /bin/sh
        rc = self.__rt_stdout(proc)
        if rc != 0:
            print(
                f"[{self.FAIL}ERROR{self.ENDC}]: 3D pose estimation algorithm failed execution."
            )
            exit(1)

    def parse_data(self, name: str, *args):
        filename = f"{name}_reader"
        if not Path(f"{self.src}/{filename}.py").exists():
            print(
                f"[{self.FAIL}ERROR{self.ENDC}]: Either model parser file doesn't exist or incorrect name provided."
            )
            exit(1)

        parser_module = il.import_module(f"models.{name}_reader")
        parse_model = getattr(parser_module, f"parse_{name}")
        model_data = parse_model(*args)

        return model_data

    def get_output_files(self, name: str, video_name: str):
        out_dir = (
            f"/root/{self.models[name]['dirname']}/{self.models[name]['output_dir']}"
        )
        dType_list = self.models[name]["output_dataFormats"]
        vType_list = self.models[name]["output_videoFormats"]

        outputs = {"data": [], "video": []}
        for (key, ext_list) in zip(outputs.keys(), (dType_list, vType_list)):
            if not isinstance(ext_list, list):
                ext_list = [ext_list]

            for ext in ext_list:
                out_files = list(
                    Path(out_dir).rglob(f"*.{ext}")
                )  # get all output files in output dir and subdir in output dir
                if out_files == []:
                    if dType_list == []:
                        print(
                            f"[{self.OKGREEN}WARNING{self.ENDC}]: Seems like 3D pose estimation algorithm generated no data. Terminating program without error."
                        )
                        exit(0)

                    print(
                        f"[{self.WARNING}WARNING{self.ENDC}]: No .{ext} output file detected"
                    )
                    break

                out_files = sorted(
                    out_files, key=(lambda p: p.stat().st_mtime), reverse=True
                )  # sort by most recent in modification time
                if key == "video":
                    outputs[key] = [p.absolute() for p in out_files if p.is_file()]
                else:
                    outputs[key] = [
                        str(p.absolute()) for p in out_files if p.is_file()
                    ][0]

        if outputs["video"] != []:
            for vid in outputs["video"]:
                v_name = vid.name
                copyfile(
                    vid,
                    f"/home/output/{name}-{video_name.lower()}-output.{v_name.split('.')[-1]}",
                )

        return outputs

    def __make_cmd(self, name: str, video: str):
        actions = self.models[name]["actions"]
        env_cmd = self.models[name]["env"]

        # construct python virtual environment command if provided
        if env_cmd != "":
            env_cmd = f"source /root/.bashrc; pyenv activate {env_cmd};"  # must source .bashrc for pyenv to work
        v_flag = self.models[name]["vid_flag"]  # get video flag for model
        self.models[name]["flags"][v_flag] = video  # set video flag to video path

        # construct executable command
        model_cmd = self.models[name]["cmd"]
        model_flags = " ".join(
            " ".join((k, str(v))).rstrip()
            for (k, v) in self.models[name]["flags"].items()
        )
        exec_cmd = " ".join([model_cmd, model_flags])

        # include pre/post processing commands for model `name` if provided
        if actions == []:
            main_cmd = "".join([env_cmd, exec_cmd])
        else:
            actions = f"{';'.join(actions)};"
            main_cmd = "".join([actions, env_cmd, exec_cmd])

        return main_cmd

    def __rt_stdout(self, proc: sp.Popen):
        """
        Print stdout of process `proc` in realtime.

        Ref: https://www.endpointdev.com/blog/2015/01/getting-realtime-output-using-python/

        Parameters
        ----------
        `proc`: subprocess

        Return
        ------
        `rc`: return code of subprocess
        """
        while True:
            out = proc.stdout.readline().decode("utf-8")
            if proc.poll() is not None:  # child process has terminated
                break
            if out:
                print(out.strip())

        rc = proc.poll()
        return rc
