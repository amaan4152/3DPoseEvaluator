#!/bin/bash -i

sudo mv models/blazepose/*.py spawner/models/blazepose
sudo mv models/gast-net/*.py spawner/models/gast-net
sudo mv models/vibe/*.py spawner/models/vibe
sudo mv models/blazepose/*.txt spawner/models/blazepose
sudo mv models/gast-net/*.txt spawner/models/gast-net
sudo mv models/gast-net/setup.sh spawner/models/gast-net
sudo mv models/vibe/setup.sh spawner/models/vibe
sudo mv models/blazepose/setup.sh spawner/models/blazepose
sudo mv models/gast-net/set_device.sh spawner/models/gast-net
sudo mv models/vibe/set_device.sh spawner/models/vibe
sudo rm -rf models/blazepose models/gast-net models/vibe