# This code can capture multi-camera images
# 
# It's best to run this script on the host rather than in the container.
# Press 'p' to save images
# Press 'q' to finish this program

# Import Libraries
import cv2, os, time
from datetime import datetime
import sys


# ==========================================
# User Settings
# ==========================================
# task_type:
#   1=fisheye-calib, 2=fisheye-test, 3=four camera calib, 4=four camera test
task_type = 1

if task_type == 1 or task_type == 2:
    camera_device_ids = [2]   # find device id with commands : 'lsusb', 'ls /dev/video*'
    camera_names = ['fisheye']
    image_width, image_height = 1920, 1080  # 1台OK, 4台NG
elif task_type == 3 or task_type == 4:
    camera_device_ids = [4, 6, 8, 10]   # find device id with commands : 'lsusb', 'ls /dev/video*'
    camera_names = ['front', 'left', 'right', 'back']
    # image_width, image_height = 1024, 576   # 4台OK, fsp=15-27
    image_width, image_height = 960, 540  # 4台OK, fps=27

print_FPS = False
compress_image = True  # This is needed for multi camera caputre
preview_window_size = (int(image_width/3), int(image_height/3))

# ==========================================
# Save dir settings
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))

if task_type == 1:
    save_name = f'fisheye_{image_width}x{image_height}'
    save_dir = f"{current_dir}/calibration/{save_name}"
elif task_type == 2:
    save_name = f'fisheye_{image_width}x{image_height}'
    save_dir = f"{current_dir}/test/{save_name}"    
elif task_type == 3:
    save_name = f'four_camera_{image_width}x{image_height}'
    save_dir = f"{current_dir}/calibration/{save_name}"
elif task_type == 4:
    save_name = f'camera_{image_width}x{image_height}'
    save_dir = f"{current_dir}/test/{save_name}" 
    
print(f'save_dir = {save_dir}')
os.makedirs(save_dir, exist_ok=True)

print(f'image shape = ({image_width}, {image_height})')


# ==========================================
# Init usb camera
# ==========================================
# RGB camera
camera_caputures = []
fournd_camera_device_ids = []


def create_camera_caputure(device_id):
    camera_caputure = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    camera_caputure.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    camera_caputure.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    
    # 複数起動対策
    camera_caputure.set(cv2.CAP_PROP_FPS, 10)  # fps -> 効かない. FPS=30を目指している
    
    # Compress image data if usb traffic is overflowed.
    if compress_image:
        camera_caputure.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    return camera_caputure


for id in camera_device_ids:
    cap = create_camera_caputure(id)
    
    # Error check
    if not cap.isOpened() :
        print(f"Cannot open camera device-{id}. Please check device id.")
        camera_caputures.append(None)
        continue
    
    camera_caputures.append(cap)
    fournd_camera_device_ids.append(id)
    
    # デバイスの初期化の待ち時間
    time.sleep(1.0)

# Check found camera numbers
if len(camera_caputures) == 0:
    print('There is no cameras.')
    exit()
else:
    print(f'Found {len(camera_caputures)}  cameras')


# ==========================================
# FPS測定用初期化
# ==========================================
frame_times = {id: time.time() for id in fournd_camera_device_ids}
fps_log = {id: [] for id in fournd_camera_device_ids}

# ==========================================
# Create priview windows
# ==========================================
for i in range(len(camera_caputures)):
    # Preview
    # ウィンドウをサイズ変更可能にする
    cv2.namedWindow(f"Device_{i}", cv2.WINDOW_NORMAL)

    # ウィンドウサイズを指定（幅=640, 高さ=480）
    cv2.resizeWindow(f"Device_{i}", preview_window_size[0], preview_window_size[1])

# ==========================================
# Main loop to captuer images
# ==========================================
try:
    while True:
        key = cv2.waitKey(1) # wait 1 ms
        p_is_pressed = False

        # [q] key to finish 
        if key == ord('q'):
            break

        # [p] key to capture
        elif key == ord('p'):
            p_is_pressed = True
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for i, cap in enumerate(camera_caputures):
            # Error check
            if cap is None:
                continue

            # Capture
            ret, frame = cap.read()  # res = (ret, frame)
            device_id = fournd_camera_device_ids[i]

            # Error check
            if not (ret):
                print(f"ID = {device_id} : Failed to caputer an image")
                continue            

            # Save the image
            if ret and p_is_pressed:
                # Save images
                save_path = os.path.join(save_dir, f"{timestamp}_{camera_names[i]}.png")
                cv2.imwrite(save_path, frame)
                print(f"[Saved] {save_path}")  

            cv2.imshow(f"Device_{i}", frame)

            if print_FPS:
                # Measure FPS
                now = time.time()
                delta = now - frame_times[device_id]
                fps = 1.0 / delta if delta > 0 else 0.0
                fps_log[device_id].append(fps)
                frame_times[device_id] = now                
                print(f'Divice {device_id} : FPS={fps}')

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Releasing camera...")

finally:
    for cap in camera_caputures:
        cap.release()

    cv2.destroyAllWindows()

    print("All resources released.")