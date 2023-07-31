# CGI DRL Platform
This repository is the official release of DRL Platform of CGI Lab (https://cgilab.nctu.edu.tw/), Department of Computer Science, National Yang Ming Chiao Tung University.
We implement some of popular deep reinforcement learning algorithm on some simulator or environemnts, especially, in video games.
Some of algorithms proposed by our lab will also be put at this repo in a specific branch:

## Publication Implementations
- Chiu-Chou Lin, Wei-Chen Chiu, I-Chen Wu. An unsupervised video game playstyle metric via state discretization. UAI 2021. [Link](https://proceedings.mlr.press/v161/lin21a.html) [branch](https://github.com/DSobscure/cgi_drl_platform/tree/playstyle_uai2021)

# Environment Installation (Example for Pommerman)
## 1. Get CGI DRL Platform
```bash
cd ~
git clone https://github.com/DSobscure/cgi_drl_platform.git
```
## 2. Install Podman [official instructions](https://podman.io/getting-started/installation#linux-distributions) (if you cannot use GPU with podman on some machines, e.g., AWS, you can use docker)
```bash
# only for ubuntu
. /etc/os-release
echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" | sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
curl -L "https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/Release.key" | sudo apt-key add -
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install podman
```
## 3. Create Container
### Build local dockerfile (use a propoer CUDA driver version on the server, we use CUDA11.6 and NVIDIA built TF1.15 version as the base file)
```bash
podman build -t cgi_drl -f $PWD/cgi_drl_platform/infrastructure/dockerfiles/CUDA11.6/Dockerfile .
```
### create a podman container
```bash
podman run -d -v $PWD:/root -w /root --name cgi_drl -it cgi_drl bash
```
## 4. Enter Container and Install cgi_drl_platform
```bash
podman exec -it cgi_drl bash
cd cgi_drl_platform/platform
pip install -e .
```
## 5. Get Pommerman Environment
```bash
cd ~
git clone https://github.com/MultiAgentLearning/playground.git
cd playground
python3 setup.py install
```
## 6. Start Distributed Server for Parallel Environments
* create a tmux panel
```bash
cd ~/cgi_drl_platform/infrastructure/DistributedServer/netcoreapp2.1
dotnet Cgi.VideoGame.Distributed.Server.dll
```
* you can speficy a specific name of log file as follows
```bash
dotnet Cgi.VideoGame.Distributed.Server.dll log_name.txt
```
## 7. Run Training or Evaluation
* for evaluation with provided model, please put version.zip under cgi_drl_platform/cgi_drl and extract it

### Pommerman bot Training
```bash
cd ~/cgi_drl_platform/platform/cgi_drl
python run.py -f pommerman.yaml -k pommerman_train -i test
```

### Pommerman bot Evaluation
```bash
cd ~/cgi_drl_platform/platform/cgi_drl
python run.py -f pommerman.yaml -k pommerman_eval -i test
```
* all logs are automatically generated in cd cgi_drl_platform/platform/cgi_drl/versions
    * including AI models
    * you can use tensorboard to ckeck the result

### modify AI or develop new bot
* start from check cgi_drl/pommerman.yaml
    * this file speficy the actual enter point of problem and corresponding configs
* for change AI model in evaluation, please check cgi_drl/problem/pommerman/ppo_solver/config/solver.yaml
    * in key "eval", specify the model directory by "load_policy_model_path"
