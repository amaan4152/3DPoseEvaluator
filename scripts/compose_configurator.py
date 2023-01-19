import sys
import yaml

from pathlib import Path

# TO-DO: Add GPU compute capabilities for model's that have GPU support
def main(compose_path: str, device_mode: str):
    if not Path(compose_path).exists():
        ValueError(
            "docker-compose specifications file cannot be found. Either incorrect path provided or file doesn't exist."
        )
    stream = open(compose_path, "r")
    configs = yaml.load(stream=stream, Loader=yaml.SafeLoader)
    services = configs["services"]
    if device_mode == "CPU":
        for s in services.keys():
            configs["services"][s].pop("deploy")
    elif device_mode == "GPU":
        deploy_dict = {
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}
                        ]
                    }
                }
            }
        }
        for s in services.keys():
            if "deploy" in configs["services"][s].keys():
                continue
            configs["services"][s].update(deploy_dict)
    else:
        ValueError(
            "Incorrect device mode specified. Supported device modes: CPU or GPU"
        )

    # disable YAML anchors: https://github.com/yaml/pyyaml/issues/535
    yaml.SafeDumper.ignore_aliases = lambda self, data: True
    with open(compose_path, "w+") as s:
        yaml.dump(configs, stream=s, Dumper=yaml.SafeDumper, sort_keys=False)
    stream.close()


if __name__ == "__main__":
    compose_path = sys.argv[1]
    device_mode = sys.argv[2]
    main(compose_path=compose_path, device_mode=device_mode)
