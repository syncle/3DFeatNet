#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
FROM tensorflow/tensorflow:1.4.0-devel-gpu-py3


ENV http_proxy  ""
ENV HTTP_PROXY  ""
ENV https_proxy ""
ENV HTTPS_PROXY ""

ENV CUDA_HOME=/usr/local/cuda-8.0/
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# this fails, but I don't know why
RUN python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'

RUN nvcc --version

RUN apt-get update && apt-get install -q -y git unzip wget
RUN pip install sklearn

ENV HOME=/home/
RUN cd $HOME && \ 
  git clone https://github.com/syncle/3DFeatNet && \
  cd $HOME/3DFeatNet/ && git checkout docker && \
  mkdir $HOME/3DFeatNet/ckpt/
COPY pretrained_model.zip $HOME/3DFeatNet/ckpt/
RUN cd $HOME/3DFeatNet/ckpt && \
  unzip pretrained_model.zip && \
  chmod +x $HOME/3DFeatNet/tf_ops/grouping/*.sh && \
  chmod +x $HOME/3DFeatNet/tf_ops/sampling/*.sh 

RUN alias python=python3 && \
  python3 -c "import tensorflow as tf" && \
  cd $HOME/3DFeatNet/ && \
  cd $HOME/3DFeatNet/tf_ops/grouping && \
  ./tf_grouping_compile.sh && \
  cd $HOME/3DFeatNet/tf_ops/sampling && \
  ./tf_sampling_compile.sh 
