#!/bin/bash

# environment setup
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9

# set up proto dir and deps
python3.9 -m venv Agent1_PPO
cd Agent1_PPO
source bin/activate
pip3 install pip==22.3.1 setuptools==59.1.1 wheel==0.12.5
pip3 install -r requirements.txt
deactivate

echo "all done, ready to train or play!"
echo "please go to https://wandb.ai/ to create an account for metrics collection - API key is required"
