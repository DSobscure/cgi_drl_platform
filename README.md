# CGI DRL Platform
This repository is the official release of DRL Platform of CGI Lab (https://cgilab.nctu.edu.tw/), Department of Computer Science, National Yang Ming Chiao Tung University.
We implement some of popular deep reinforcement learning algorithm on some simulator or environemnts, especially, in video games.
Some of algorithms proposed by our lab will also be put at this repo in a specific branch:

## Publication Implementations
- Chiu-Chou Lin, Wei-Chen Chiu, I-Chen Wu. An unsupervised video game playstyle metric via state discretization. UAI 2021. [Link](https://proceedings.mlr.press/v161/lin21a.html) [branch](https://github.com/DSobscure/cgi_drl_platform/tree/playstyle_uai2021)
[Dataset and HSD Models](https://zenodo.org/record/8191453)
- Kuo-Hao Ho, Ping-Chun Hsieh, Chiu-Chou Lin, You-Ren Lou, Feng-Jian Wang, I-Chen Wu. Towards Human-Like RL: Taming Non-Naturalistic Behavior in Deep RL via Adaptive Behavioral Costs in 3D Games. ACML 2023. [Link](https://proceedings.mlr.press/v222/ho24a.html) [branch](https://github.com/DSobscure/cgi_drl_platform/tree/human_like_behavior)

## Project Information
### Banana Collector
* A example environment in [Unity ML Agent](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Examples.md)
* Banana Collector is an old version in early version of ML Agents. The new version changed to the name, Food Collector.
    * ![Food Collector](https://github.com/Unity-Technologies/ml-agents/raw/develop/docs/images/foodCollector.png)
* The corresponding setting can be found in [their offical github repo](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Learning-Environment-Examples.md).
    * Our setting
        * Set-up: A single-agent environment where the agent collect food.
        * Goal: The agents must learn to collect as many green food spheres as possible while avoiding red spheres.
        * Agents: The environment contains only 1 agents for Unity Gym interface.
        * Agent Reward Function (independent):
            * +1 for interaction with green spheres
            * -1 for interaction with red spheres
        * Behavior Parameters:
            * Actions:
                * 3 discrete actions correspond to Forward Motion
                    * 0: idle
                    * 1: forward
                    * 2: backward
                * 3 discrete actions correspond to Rotation
                    * 0: idle
                    * 1: turn left
                    * 2: turn right
            * Visual Observations: First-person camera per-agent.
                * Size: 84x84 
                * Frame stack: 1

### Paper Publication
[ACML 2023 Towards Human-Like RL: Taming Non-Naturalistic Behavior in Deep RL via Adaptive Behavioral Costs in 3D Games](https://proceedings.mlr.press/v222/ho24a.html)
* A method for learning multi-objective RL agents with adaptive scheduling to behavior costs
    * By reducing user defined non-naturalistic behaviors, agents have the ability to achieve multi-objective, especially in this case is human-like objective.
        * Playing in 3D First Person Shooting games, quick shaking and spinning long time are not human-like.

## Fraework Resource
* CGI DRL Platform: (public github repository)
    * https://github.com/DSobscure/cgi_drl_platform/tree/human_like_behavior
    * a static version at https://cgi.lab.nycu.edu.tw/~shared_data/BananaCollector/cgi_drl_platform.zip
* Banana Collector (now, Fodd Collector) model
    * https://cgilab.nctu.edu.tw/~shared_data/BananaCollector/versions.zip
* Unity ML Agent Project Asset (For scripting game environment and generating human demonstrations)
    * https://cgilab.nctu.edu.tw/~shared_data/BananaCollector/Project.zip

## Environment Installation
### System 
* Prepare a Windows environment with Python and PyTorch
    * The experiments are run on 
        * Python 3.12
        * Torch 2.3.1 + CUDA 11.8
* Install Unity ML Agent Gym Environment
    * https://github.com/Unity-Technologies/ml-agents/tree/develop
    * Open Windows PowerShall or CMD in a new project directory
```bash
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents-envs
pip install -e .
```
### Get CGI DRL Platform (Branch: human_like_behavior)
* Open Windows PowerShall or CMD in the project directory
```bash
git clone -b human_like_behavior --single-branch https://github.com/DSobscure/cgi_drl_platform.git
```
* Unzip FoodCollector(Banana Collector) game environment
    * `cgi_drl_platform\infrastructure\GameExecutables\VisualFoodCollectorSingleWindows.zip`
### Install CGI DRL Platform for customized usage
* Open Windows PowerShall or CMD in the project directory
```bash
cd cgi_drl_platform\platform
pip install -e .
cd cgi_drl
```

#### Inference pretrained training
* Extract zip file versions.zip
* FoodCollecto PPO Baseline
```bash
python run.py -k ppo_food_collector_inference
```
* FoodCollecto PPO Baseline with constant behavior cost (multi-objective for reducing shaking and spinning behaviors)
```bash
python run.py -k constant_cost_ppo_food_collector_inference
```
* FoodCollecto with ABC-RL PPO ([ACML 2023 Towards Human-Like RL: Taming Non-Naturalistic Behavior in Deep RL via Adaptive Behavioral Costs in 3D Games](https://proceedings.mlr.press/v222/ho24a.html))
```bash
python run.py -k abc_rl_ppo_food_collector_inference
```

#### Run new training
* FoodCollecto PPO Baseline
```bash
python run.py -k ppo_food_collector
```
* FoodCollecto PPO Baseline with constant behavior cost (multi-objective for reducing shaking and spinning behaviors)
```bash
python run.py -k constant_cost_ppo_food_collector
```
* FoodCollecto with ABC-RL PPO ([ACML 2023 Towards Human-Like RL: Taming Non-Naturalistic Behavior in Deep RL via Adaptive Behavioral Costs in 3D Games](https://proceedings.mlr.press/v222/ho24a.html))
```bash
python run.py -k abc_rl_ppo_food_collector
```

#### Detailed for customized usage
* human_like.yaml contains all related config key (-k) for experiments
    * You can start to set new experiment config here for selecting different execution workflow or config
*  data_storage\gae_sample_memory
    *  This plug-and-play module can stroge game trajectories (also support multi-agent) and compute GAE for RL training
*  decision_model\ppo
    *  This plug-and-play module have our version of PPO model
        *  There are PyTorch version and TF 1.15 version
            *  PyTorch version is used in the training of this release
*  environment\unity_gym
    *  This plug-and-play module wraps the interface provided by Unity ML Agent Gym and our visual observation preprocessor for logging video in training/inference that agents observed
        *  The game screen in Unity window is not what agents observed, we change the position of main camera for better human observation
*  problem\food_collector
    *  This module is the main workflow of this system
    *  You can add new algorithms or combining other moudles for customized usage