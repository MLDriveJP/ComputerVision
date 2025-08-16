# This code can capture stereo-camera images of OAK-D S2
# 
# It's best to run this script on the host rather than in the container.
# Press 'p' to save images
# Press 'q' to finish this program

import depthai as dai
import cv2
import os
from datetime import datetime

# ==========================================
# User Settings
# ==========================================
# Preview Window Size for RGB
preview_width, preview_height = 320, 240

# ==========================================
# Save dir settings
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
save_name = 'stereo_7x4_39.33mm'
save_dir = f"{current_dir}/calibration/{save_name}"
print(f'save_dir = {save_dir}')
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# DepthAI pipeline definition
# ==========================================

# Create the pipeline (linking nodes in the processing flow)
pipeline = dai.Pipeline()

# -----------------------------
# RGB color camera node (center camera)
#   RGB camera max resolution = 4032×3040 -> 4032×3040*3 byte ~ 35.1 MB 
#   when 1920x1080 -> 1920x1080*3 byte ~ 6.2 MB 
# -----------------------------

center_rgb = pipeline.create(dai.node.ColorCamera)
center_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A) # CAM_A = center of OAK-D S2
center_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
center_rgb.setPreviewSize(preview_width, preview_height)

# Set color order to BGR (for OpenCV compatibility)
center_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Set to not interleaved (separate RGB channels, compatible with OpenCV)
center_rgb.setInterleaved(False)


# -----------------------------
# Left mono camera
#   ref : https://docs.ros.org/en/noetic/api/depthai/html/structdai_1_1MonoCameraProperties.html
#   Gray camera max resolution = 1280×800 -> 1280x800x1 byte = 1.0 MB
# -----------------------------
left_gray = pipeline.create(dai.node.MonoCamera)
left_gray.setBoardSocket(dai.CameraBoardSocket.CAM_B) # CAM_B = left of OAK-D S2
left_gray.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P) # 1280×800


# -----------------------------
# Right mono camera
# -----------------------------
right_gray = pipeline.create(dai.node.MonoCamera)
right_gray.setBoardSocket(dai.CameraBoardSocket.CAM_C) # CAM_C = right of OAK-D S2
right_gray.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P) # 1280×800


# -----------------------------
# Create output nodes and link them
# -----------------------------
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
center_rgb.preview.link(xout_rgb.input)

xout_left = pipeline.create(dai.node.XLinkOut)
xout_left.setStreamName("left")
left_gray.out.link(xout_left.input)

xout_right = pipeline.create(dai.node.XLinkOut)
xout_right.setStreamName("right")
right_gray.out.link(xout_right.input)

# ==========================================
# Device startup and real-time image display
# ==========================================
counter = 0
with dai.Device(pipeline) as device:
    # Get output queues for each camera (non-blocking with a maximum of 4 buffers)
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    while True:
        # Get frames from each camera in OpenCV format
        rgb_frame = rgb_queue.get().getCvFrame()
        left_frame = left_queue.get().getCvFrame()
        right_frame = right_queue.get().getCvFrame()

        # Display the frames in separate OpenCV windows
        cv2.imshow("RGB Camera", rgb_frame)
        cv2.imshow("Left Mono", left_frame)
        cv2.imshow("Right Mono", right_frame)

        # Check for key input (wait 1 millisecond)
        key = cv2.waitKey(1)

        # Exit if the [q] key is pressed
        if key == ord('q'):
            break

        # Save images if the [p] key is pressed
        elif key == ord('p'):
            # Get the current timestamp in "YYYYmmddHHMMSS" format
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # Create file paths for saving images (center/left/right)
            path_rgb = os.path.join(save_dir, f"{timestamp}_center.png")
            path_left = os.path.join(save_dir, f"{timestamp}_left.png")
            path_right = os.path.join(save_dir, f"{timestamp}_right.png")

            # Save the images as PNG files using OpenCV's imwrite function
            cv2.imwrite(path_rgb, rgb_frame)
            cv2.imwrite(path_left, left_frame)
            cv2.imwrite(path_right, right_frame)

            # Print a message indicating that the images have been saved
            counter += 1
            print(f"[Saved: {counter:03d}] {path_rgb}, {path_left}, {path_right}")

# Close all OpenCV windows
cv2.destroyAllWindows()
