# Reacher PPO

This is a code for 2nd project in Udacity Deep Reinforcement Learning Nanodegree. For this environment (i.e Reacher) I chose PPO algorithm as I could see it approaching the required score threshold. I also tried DDPG , both on a CUDA based h/w (provided as Udacity workspace) and locally on a CPU only machine. In both the cases, the average score just got stuck around 1.3 (Later on I realised that I was using single agent, I have to try out if DDPG with 20 agents will give me the desired results)

![](images/train_end.gif)

## Environment

Agent was learned on Reacher environment with `20` simultanous agents for faster rollout gathering. Environment is solved when average reward over all agents in last 100 episodes is over `30.0`. 

```
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		goal_speed -> 1.0
		goal_size -> 5.0
Unity brain name: ReacherBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 33
        Number of stacked Vector Observation: 1
        Vector Action space type: continuous
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```

## Getting started

* Make sure python 3.6 is installed
* Activate the environment as described [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)
* Once drlnd environment is activated, install package tqdm : pip install tqdm
* Download and install Unity environment as described (Step 2 ) [here](https://github.com/udacity/deep-reinforcement-learning#dependencies  )

* Clone the git repository from [here](https://github.com/vimalkansal/DRLND_VKANSAL_PROJECT_2)
* Switch to cloned repository directory
* From the shell execute :   **jupyter notebook**
* Open Continuous_Control-PPO notebook
* Execute the cells by pressing "shift + enter" keys
* File ppo_model.py contains neural network used as policy estimator and ppo_agent.py contains the logic for PPO agent. Trained model weights are in '/weights' directory


## Sources

[1] Original PPO paper by Open AI [Proximal Policy Optimization Algorithm](https://arxiv.org/pdf/1707.06347.pdf)

[2] Repository of PyTorch Deep RL implementations [DeepRL](https://github.com/ShangtongZhang/DeepRL)