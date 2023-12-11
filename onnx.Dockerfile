FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /workspace
RUN apt-get update \
    && apt-get install -y libcairo2 python3 python3-dev python3-pip libffi-dev time \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq autoremove \
    && apt-get -qq clean
COPY scripts/onnx_inference.py model.onnx onnx-requirements.txt .
RUN pip install -U pip wheel \
    pip install -r onnx-requirements.txt \
    pip cache purge
ENTRYPOINT /workspace/onnx_inference.py
