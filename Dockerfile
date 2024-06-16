FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /home/user/project

# 必要なものを入れる
RUN apt update && apt install -y ffmpeg gcc cmake libsndfile1

# 途中でtimezone求められたりして停止しないように
ENV DEBIAN_FRONTEND="noninteractive"

### anaconda ### ref:https://www.eureka-moments-blog.com/entry/2020/02/22/160931#3-AnacondaPython37%E3%82%92%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB
# RUN set -x && \
#     apt update && \
#     apt upgrade -y
# RUN set -x && \
#     apt install -y wget && \
#     apt install -y sudo
# RUN set -x && \
#     wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
#     bash Anaconda3-2023.09-0-Linux-x86_64.sh -b && \
#     rm Anaconda3-2023.09-0-Linux-x86_64.sh
# ENV PATH $PATH:/root/anaconda3/bin
# RUN conda craete -n environment python==3.12.1 && \
#     conda activate environment
################

# pyenv全体設定 # ref https://blog.8tak4.com/post/158052756945/dockerfile-pyenv
ENV HOME /home/user/project
ENV PYENV_ROOT /home/user/.pyenv
### shimsが無いとpythonが通らない https://qiita.com/makuramoto1/items/b5aa08d5fc1ce6af0fb4
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
### pyenvに必要なものインストール https://github.com/pyenv/pyenv/blob/master/Dockerfile
RUN apt-get update -y \
    && apt-get install -y \
        make \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        git
RUN git clone https://github.com/yyuu/pyenv.git $PYENV_ROOT
RUN pyenv --version && \
    pyenv install 3.11.7 && \
    pyenv global 3.11.7  && \
    pyenv rehash
RUN pip install --upgrade pip
RUN pip install poetry
