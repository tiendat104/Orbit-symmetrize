FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# install pip
RUN apt update && apt install python3-pip -y

# install git
RUN apt install git -y

# install dependencies
RUN pip3 install numpy==1.23.5
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install xformers lightning transformers
RUN pip3 install --pre timm
RUN pip3 install easydict fastapi wandb plum-dispatch scikit-learn matplotlib
RUN pip3 install datasets
RUN pip3 install git+https://github.com/pyg-team/pytorch_geometric.git
RUN pip3 install pytorch3d

CMD ["/bin/bash"]
