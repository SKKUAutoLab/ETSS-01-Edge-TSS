FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl tar
RUN apt install -y ffmpeg

# Create working directory
RUN mkdir -p /usr/src/aic23-track_5

# Install pip packages
RUN alias python=python3
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache albumentations comet gsutil click
RUN pip install --no-cache torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install requirements
COPY . /usr/src/aic23-track_5
RUN pip install --no-cache --root-user-action=ignore -r /usr/src/aic23-track_5/requirements.txt

# Make workspace
WORKDIR /usr/src/aic23-track_5

# Copy contents
# COPY . /usr/src/app  (issues as not a .git directory)
#RUN git clone https://github.com/ultralytics/ultralytics /usr/src/ultralytics
#ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt /usr/src/ultralytics/

# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND teletype
