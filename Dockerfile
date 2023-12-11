FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel as base
ARG FORCE_CUDA
ENV CONDA_PREFIX=/opt/conda
WORKDIR /workspace

# install diffvg
RUN apt-get update && apt-get install -y build-essential curl ffmpeg git parallel python3-dev libsm6 
RUN curl -O https://cmake.org/files/v3.12/cmake-3.12.4-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh cmake-3.12.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake-3.12.4-Linux-x86_64.sh

RUN git clone https://github.com/BachiLi/diffvg --recursive \
    && git checkout e5955dbdcb4715ff3fc6cd7d74848a3aad87ec99
WORKDIR /workspace/diffvg
# TODO: uncomment below line to increase possible number of shapes supported by diffvg
#RUN sed -i 's/constexpr auto max_hit_shapes = 256;/constexpr auto max_hit_shapes = 1024;/g' diffvg/diffvg.cpp
COPY diffvg-requirements.txt .
RUN pip install --upgrade pip && pip install -r diffvg-requirements.txt
RUN DIFFVG_CUDA=$FORCE_CUDA CMAKE_PREFIX_PATH=$CONDA_PREFIX python setup.py install

# install douga-vectorization
WORKDIR /workspace
COPY requirements.txt .
RUN apt-get install -y imagemagick time
RUN pip install -r requirements.txt && pip install .
RUN mkdir bin \
    && wget https://inkscape.org/gallery/item/26933/Inkscape-c4e8f9e-x86_64.AppImage -O bin/inkscape \
    && chmod u+x bin/inkscape
