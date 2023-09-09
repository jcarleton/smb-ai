#!/bin/bash

# environment setup
wget -nv -O- https://lambdalabs.com/install-lambda-stack.sh | sh -
sudo apt update && sudo apt upgrade

# set up proto dir and deps
python -m venv Agent2_DQN_tf
cd Agent2_DQN_tf
source bin/activate
pip3 install pip==22.3.1 setuptools==59.1.1 wheel==0.37.1
pip3 install -r requirements.txt
deactivate

echo "all done, ready to train or play!"
echo "please go to https://wandb.ai/ to create an account for metrics collection - API key is required"
