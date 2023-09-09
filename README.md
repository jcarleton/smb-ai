# smb-ai

Super Mario Bros AI for CM3070

Two AI agent implementations:
 - Agent 1: PPO
 - Agent 2: DDQN

Agent 1 is written using the StableBaselines3 library

Agent 2 is written using Tensorflow and Keras


Please run either shell script to create the venv for each agent to run in. The
DDQN agent makes use of libraries installed via the Lambda Stack from
LambdaLabs, which provides an easy way to install several libraries and
dependencies (including Nvidia drivers). Please ensure you wish to make such a
change on your system before running this script. Learn more about the Lamba
Stack here [https://lambdalabs.com/lambda-stack-deep-learning-software]
