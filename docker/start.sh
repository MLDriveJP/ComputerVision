#!/bin/bash

source ./common.sh

xhost +local:docker

# data dir in your own pc
KITTI_DIR=${HOME}/work/data/KITTI/

docker run \
    --gpus all \
    --net=host \
    -itd \
    --shm-size=8G \
    --privileged \
    -v ${PARENT_DIR}/working:/kaggle/working \
    -v ${KITTI_DIR}:/kaggle/input/KITTI/ \
    -v ${PARENT_DIR}/third_party:/kaggle/third_party \
    -v ${PARENT_DIR}/README.md:/kaggle/README.md \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --name $CONTAINER_NAME \
    $IMAGE_NAME

# --privileged # usbカメラを使うために必要

 #   -v ${PARENT_DIR}/input:/kaggle/input \
 #   -v ${PARENT_DIR}/.git:/kaggle/.git \
 #   -v ${PARENT_DIR}/.gitattributes:/kaggle/.gitattributes \
 #   -v ${PARENT_DIR}/.gitignore:/kaggle/.gitignore \
 #   -v ${PARENT_DIR}/.vscode:/kaggle/.vscode \
 #   -v ${PARENT_DIR}/README.md:/kaggle/README.md \
 #   -v ${PARENT_DIR}/docker:/kaggle/docker \
 #   -v ${PARENT_DIR}/input:/kaggle/input \
 #   -v ${PARENT_DIR}/working:/kaggle/working \
 #   -v ${PARENT_DIR}/code:/kaggle/code \
 #   -v ${PARENT_DIR}/upload:/kaggle/upload \
 #   -v ${PARENT_DIR}/download:/kaggle/download \
 #   -v ${PARENT_DIR}/wandb/.netrc:/home/${USER_NAME}/.netrc \
 #   -v ${PARENT_DIR}/utility:/kaggle/utility \
 #   -v ${HOME}/.kaggle:/home/${USER_NAME}/.kaggle \