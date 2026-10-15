# Training image for scripts/xarm6/finetune_pi05.sh on CHTC GPU Lab.
#
# Same Python packages as the local LeRobot venv (.cache/pi05-serve-venv, lerobot 0.6.1), with
# torch / torchvision from the CUDA 12.6 index: the wheels bundle their CUDA runtime, and 12.6
# covers every GPU-Lab card (A100 / L40 / L40S / H100 / H200) on drivers >= 560.
#
# Build and push from the SimFoundry repo root:
#   docker build -t <dockerhub_user>/simfoundry_lerobot_train:0.6.1 -f scripts/xarm6/chtc/train.Dockerfile scripts/xarm6/chtc
#   docker push <dockerhub_user>/simfoundry_lerobot_train:0.6.1

FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/opt/venv
RUN uv venv --python 3.12.3 $VIRTUAL_ENV

RUN uv pip install --python $VIRTUAL_ENV/bin/python \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.11.0 torchvision==0.26.0
COPY requirements-train.txt /tmp/requirements-train.txt
RUN uv pip install --python $VIRTUAL_ENV/bin/python -r /tmp/requirements-train.txt

# CHTC runs containers as a non-root user.
RUN chmod -R a+rX $VIRTUAL_ENV /opt/uv-python

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
CMD ["/bin/bash"]
