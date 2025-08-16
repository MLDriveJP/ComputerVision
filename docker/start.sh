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
    -v ${PARENT_DIR}/input:/kaggle/input \
    -v ${KITTI_DIR}:/kaggle/input/KITTI/ \
    -v ${PARENT_DIR}/README.md:/kaggle/README.md \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --name $CONTAINER_NAME \
    $IMAGE_NAME