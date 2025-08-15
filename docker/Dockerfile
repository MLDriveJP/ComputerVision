# Using Kaggle container. This version supports up to Cuda 12.4
#   ref : # https://console.cloud.google.com/artifacts/docker/kaggle-gpu-images/us/gcr.io/python?pli=1
FROM gcr.io/kaggle-gpu-images/python:v158 

# Setting environmenta variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Additional install
RUN pip install \
        open3d==0.18.0 \
        pythreejs==2.4.2 \
        tree \
        stereovision \
        ultralytics \
        depthai \
        reportlab

# User information
ARG USER_NAME
ARG USER_ID
RUN useradd -m -s /bin/bash -u $USER_ID $USER_NAME && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the permissions correctly
RUN mkdir -p /kaggle/working
RUN mkdir -p /kaggle/input
RUN chown -R $USER_NAME:$USER_NAME /kaggle/
RUN touch /tmp/kaggle.log && chmod 666 /tmp/kaggle.log

# Set the default user
USER $USER_NAME
WORKDIR /kaggle/


#======================================================#
CMD ["/bin/bash"]
