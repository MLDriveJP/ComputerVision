#!/bin/bash

source ./common.sh

docker build \
    --progress=plain \
    --build-arg USER_NAME=${USER_NAME} \
    --build-arg USER_ID=${USER_ID} \
    --build-arg GROUP_ID=${USER_GID} \
    -t $IMAGE_NAME .

# --no-cache \
# docker build --no-cache --progress=plain .
