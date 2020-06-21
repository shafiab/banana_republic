# Banana Republic
A deep reinforcement learning agent trained on Unity ML Agents to collect bananas

## Introduction
In this project, I trained a deep reinforcement learning agent using [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) (Deep Q-Networks) algorithm to collect yellow bananas while avoiding the blue bananas on Unity ML-agent. Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. This particular setting is known as the Banana Collector Environment - a modified environment for this Udacity Deep RL project.

Here is how the trained agent behave:

[image_1]: banana_republic.gif "Trained Agents"
![Trained Agents][image_1]


## The Environment
In this environment, the agent navigate (and collect bananas) in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic. For this project, the environment is considered solved if the agent can get an average score of +13 over 100 consecutive episodes.

## How to Navigate this Repo
The repo consists of the following files:
- `main.ipynb` contains a jupyter notebook with all the codes to train and run the agent.
- `Report.md` contains description of the algorithm and hyper-parameters and performance of the algorithm.
- `checkpoint.pth` contains the weights of trained agent's neural network. See `main.ipynb` for example of how to load these weights.

## Setup / How to Run?

I trained the agent using gpu in a workspace provided by Udacity. However, the workspace does not allow to see the simulator of the environment. So, once the agent is trained, I load the trained network in a Jupyter Notebook in macbook and observed the behavior of the agent in a pre-built unity environment. The steps for the setup is as follows:

- Follow the instruction in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. 
- Once the setup is done, we can activate the environment and run notbook as follows:
```
source activate drlnd
jupyter notebook
```
This will open a notebook session in the browser.
- The pre-build unity environment `Banana.app.zip` is also included in this repo.
- So, we can just use the `main.ipynb` notebook to train and run the agent. All codes are included in that notebook.



