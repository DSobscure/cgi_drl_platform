# CGI DRL Platform
This repository is the official release of DRL Platform of CGI Lab (https://cgilab.nctu.edu.tw/), Department of Computer Science, National Yang Ming Chiao Tung University.
We implement some of popular deep reinforcement learning algorithm on some simulator or environemnts, especially, in video games.
Some of algorithms proposed by our lab will also be put at this repo in a specific branch:

## Publication Implementations
- Chiu-Chou Lin, Wei-Chen Chiu, I-Chen Wu. An unsupervised video game playstyle metric via state discretization. UAI 2021. [Link](https://proceedings.mlr.press/v161/lin21a.html) [branch](https://github.com/DSobscure/cgi_drl_platform/tree/playstyle_uai2021)
[Dataset and HSD Models](https://zenodo.org/record/8191453)


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
## 5. Download Dataset and HSD Models
[Dataset and HSD Models](https://zenodo.org/record/8191453)
* upzip playstyle_uai2021.zip at /root/ in the container

# Run Experiemnts
```bash
cd ~/cgi_drl_platform/platform/cgi_drl
python run.py -k atari-metric-breakout
```
* do be bone
