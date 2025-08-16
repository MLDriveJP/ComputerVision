# ComputerVision
Basic to Advanced Computer Vision Skills

## 1. How to setup

### 1.1 Docker Container to run computer vision scripts
1. Build a docker container
```bash
cd ./docker
./build.sh
```

2. Start the container
```bash
./start.sh
```

### 1.2 Python virtual Environment to capture images on the Host PC.
```bash
# Install virtual environment libraries
sudo apt install python3-venv -y

# Create a virtual env
python3 -m venv camera_env

# Activate the virtual env
source camera_env/bin/activate

# Install libraries
python3 -m pip install depthai-sdk
sudo apt install inkscape
```