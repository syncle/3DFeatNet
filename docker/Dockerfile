FROM ubuntu:16.04

# https://hub.docker.com/r/tleyden5iwx/ubuntu-cuda/dockerfile/

ENV http_proxy  "http://proxy-chain.intel.com:911"
ENV HTTP_PROXY  "http://proxy-chain.intel.com:911"
ENV https_proxy "http://proxy-chain.intel.com:911"
ENV HTTPS_PROXY "http://proxy-chain.intel.com:911"

######################
# Install Cuda
######################
# A docker container with the Nvidia kernel module and CUDA drivers installed
ENV CUDA_RUN https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run

RUN apt-get update && apt-get install -q -y \
  wget \
  build-essential \
  module-init-tools \
  kmod \
  sudo

RUN cd /opt && \
  wget $CUDA_RUN && \
  mv cuda_8.0.61_375.26_linux-run cuda_8.0.61_375.26_linux.run && \
  chmod +x *.run && \
  mkdir nvidia_installers && \
  ./cuda_8.0.61_375.26_linux.run -extract=`pwd`/nvidia_installers && \
  cd nvidia_installers && \
  ls && \
  ./NVIDIA-Linux-x86_64-375.26.run -s -N --no-kernel-module

RUN cd /opt/nvidia_installers && \
  ./cuda-linux64-rel-8.0.61-21551265.run -noprompt

#  python -c "import tensorflow as tf"
COPY cudnn-8.0-linux-x64-v6.0.tgz .
RUN tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz && \
  sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include && \
  sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/ && \
  sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn* && \
  rm cudnn-8.0-linux-x64-v6.0.tgz

# Ensure the CUDA libs and binaries are in the correct environment variables
#ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
ENV CUDA_HOME=/usr/local/cuda-8.0/
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
RUN nvcc --version


######################
# Install Conda
######################
ENV HOME /home/
ENV CONDA_SHELL https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh

RUN wget $CONDA_SHELL && \
  chmod +x *.sh && \
  ./Anaconda3-2018.12-Linux-x86_64.sh -b -p $HOME/conda

ENV PATH="$HOME/conda/bin:$PATH"
RUN echo ". /home/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  . ~/.bashrc && \
  conda create -n 3dfeatnet python=3.5 && \
  conda activate 3dfeatnet && \
  pip install tensorflow-gpu==1.4

RUN echo ". /home/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  . ~/.bashrc && \
  conda activate 3dfeatnet && \
  echo $PATH && \
  echo $LD_LIBRARY_PATH && \
  ls -alh /usr/local/cuda-8.0//lib64/libcudnn*

##################################
# Git clone and compile tensor_ops
##################################
RUN apt-get install -q -y git
RUN cd $HOME && \
  git clone https://github.com/yewzijian/3DFeatNet
RUN echo ". /home/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  . ~/.bashrc && \
  conda activate 3dfeatnet && \ 
  chmod +x $HOME/3DFeatNet/tf_ops/grouping/*.sh && \
  chmod +x $HOME/3DFeatNet/tf_ops/sampling/*.sh && \
  cd $HOME/3DFeatNet/tf_ops/grouping && \
  ./tf_grouping_compile.sh && \
  cd $HOME/3DFeatNet/tf_ops/sampling && \
  ./tf_sampling_compile.sh



