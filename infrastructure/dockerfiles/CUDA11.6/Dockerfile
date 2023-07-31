FROM nvcr.io/nvidia/tensorflow:22.04-tf1-py3

RUN apt update \
    && apt install htop tmux git vim wget -y \
    && pip install numpy --upgrade \
    && pip install docker-pycreds \
    && pip install pyglet 

# dotnet 2.1
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt update -y \
    && apt-get install -y dotnet-sdk-2.1 -y\
    && pip install msgpack-numpy
	