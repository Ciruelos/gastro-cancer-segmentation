FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update --fix-missing

RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 git

ENV PYTHONPATH '${PYTHONPATH}:/workspace'

WORKDIR /workspace

ADD requirements.txt .

RUN pip install -r requirements.txt
