#!/bin/bash -i

echo "eval \"\$(pyenv init -)\"" >> ~/.bashrc
echo "eval \"\$(pyenv virtualenv-init -)\"" >> ~/.bashrc
source ~/.bashrc
pyenv install 3.6.5
pyenv install 3.7.0
pyenv virtualenv 3.6.5 gast-env
pyenv virtualenv 3.7.0 vibe-env
