

from tqdm import tqdm
import numpy as np
import cv2
import open3d as o3d
import pythreejs as p3js
from matplotlib import pyplot as plt

def create_csv_line(params: np.ndarray) -> str:
    params = params.flatten()
    line = ''
    for i, param in enumerate(params):
        if i == 0:
            line = f'{param: .6e}'
        else:
            line += f', {param: .6e}'
    return line


def stereoCalibrate(file_paths_left, file_paths_right, board, detector, 
                    board_world_points, detect_thresh=0.9, use_id_matching=True):
    # modify its type to np
    board_world_points = np.array(board_world_points) # shape=(N,3)
    num_corners = board_world_points.shape[0]
    detect_thresh_points_num = np.floor(num_corners * detect_thresh)
    
    # results of detector
    detected_corner_ids_left = []
    detected_corner_ids_right = []
    detected_corner_points_left = []
    detected_corner_points_right = []    
    
    # results of matcher
    matched_img_points_left = [] # 2D points in pixel
    matched_img_points_right = [] # 2D points in pixel
    matched_world_points_left = [] # 3D points in real world
    matched_world_points_right = [] # 3D points in real world
    matched_file_ids = []
    
    # =================== Step-1 & 2. Detect and Match =================== #
    for i, (file_path_left, file_path_right) in enumerate( tqdm( 
                zip(file_paths_left, file_paths_right), total=len(file_paths_left), desc="Stereo Calibration") ):
        # Load image
        img_left = cv2.imread(file_path_left, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(file_path_right, cv2.IMREAD_GRAYSCALE)
        
        # Step-1. Detect board corners
        c_corners_left, c_ids_left, _, _ = detector.detectBoard(img_left)
        c_corners_right, c_ids_right, _, _ = detector.detectBoard(img_right)
        
        # Error check-1
        if c_corners_left is None or c_corners_right is None:
            print(f'ID = {i} : There are no corners')
            continue
        
        # ID matching check
        if use_id_matching and (len(c_corners_left) != num_corners or len(c_corners_right) != num_corners):
            # get common ids with left and right
            common_ids = np.intersect1d(c_ids_left.flatten(), c_ids_right.flatten())        
          
            # Error check-2
            if len(common_ids) < detect_thresh_points_num:
                print(f'ID = {i} : detected common_ids num is smaller than {detect_thresh_points_num}')
                print(f'    len(common_ids) = {len(common_ids)}')
                continue
            
            # get array position for common ids
            pos_common_left = [np.where(c_ids_left.flatten() == cid)[0][0] for cid in common_ids]
            pos_common_right = [np.where(c_ids_right.flatten() == cid)[0][0] for cid in common_ids]
            
            # get filtered points and ids
            c_corners_left = (c_corners_left[pos_common_left]).reshape(-1,1,2)  # for cv2 shape=(N, 1, 2)
            c_corners_right = (c_corners_right[pos_common_right]).reshape(-1,1,2)  # for cv2 shape=(N, 1, 2)
            c_ids_left = common_ids.reshape(-1,1) # for cv2 shape=(N, 1)
            c_ids_right = common_ids.reshape(-1,1) # for cv2 shape=(N, 1)
        else:
            if len(c_corners_left) != len(c_corners_right):
                print(f'ID = {i} : detected corners num is not the same')
                print(f'    len(c_corners_left) = {len(c_corners_left)}')
                print(f'    len(c_corners_right) = {len(c_corners_right)}')
                continue
        
        # Step-2. Match
        world_points_left, img_points_left = board.matchImagePoints(c_corners_left, c_ids_left)
        world_points_right, img_points_right = board.matchImagePoints(c_corners_right, c_ids_right)
    
        # Save: results of detector
        detected_corner_ids_left.append(c_ids_left)
        detected_corner_ids_right.append(c_ids_right)
        detected_corner_points_left.append(c_corners_left)
        detected_corner_points_right.append(c_corners_right)
        
        # Save: results of matcher
        matched_img_points_left.append(img_points_left)
        matched_img_points_right.append(img_points_right)
        matched_world_points_left.append(world_points_left)
        matched_world_points_right.append(world_points_right)
        
        matched_file_ids.append(i)

    print(f'Done (2/4) : Corner detection and matching')

    # =================== Step-3: Mono Vision Calibration =================== #
    img_size = (img_left.shape[1], img_left.shape[0]) # (width, height)
    # Left
    ret_mono_left, K_mono_left, D_mono_left, R_mono_left, T_mono_left = cv2.calibrateCamera(matched_world_points_left, matched_img_points_left, img_size, None, None)
    new_K_mono_left, roi_left = cv2.getOptimalNewCameraMatrix(K_mono_left, D_mono_left, img_size, 1, img_size)

    # Right
    ret_mono_right, K_mono_right, D_mono_right, R_mono_right, T_mono_right = cv2.calibrateCamera(matched_world_points_right, matched_img_points_right, img_size, None, None)
    new_K_mono_right, roi_right = cv2.getOptimalNewCameraMatrix(K_mono_right, D_mono_right, img_size, 1, img_size)
    
    print(f'Done (3/4): Mono vision calibration')
    
    # =================== Step-4: Stereo Vision Calibration =================== #
    # Set stereo calibration parameters
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0e-4)
    
    # Run stereo calibration !
    # 左右カメラの内部・外部パラメータをもとに、両カメラ間の相対位置関係（回転・並進）を推定する
    # 入力
    #   - matched_world_points_left : 左カメラ画像に対応するチャート上の3Dワールド座標群（List[ndarray(N,3)]）
    #   - matched_img_points_left  : 左カメラ画像上の対応する2D画像座標（List[ndarray(N,2)]）
    #   - matched_img_points_right : 右カメラ画像上の対応する2D画像座標（List[ndarray(N,2)]）
    #   - K_mono_left, D_mono_left : 左カメラの内部パラメータとレンズ歪み係数（事前に mono calibration 済）
    #   - K_mono_right, D_mono_right : 右カメラの内部パラメータとレンズ歪み係数（同上）
    #   - img_size : キャリブレーション画像サイズ（例：1226x370）
    #   - stereo_criteria : 最適化の収束条件（cv2.TERM_CRITERIA_XXX）
    #   - stereo_flags : 内部パラメータ固定などのフラグ（例：cv2.CALIB_FIX_INTRINSIC）
    # 出力：
    #   - ret_stereo         : 最小化された再投影誤差（平均ピクセル単位）
    #   - K_stereo_left      : 左カメラの内部パラメータ（通常はK_mono_leftと同じ）
    #   - D_stereo_left      : 左カメラの歪み係数
    #   - K_stereo_right     : 右カメラの内部パラメータ（通常はK_mono_rightと同じ）
    #   - D_stereo_right     : 右カメラの歪み係数
    #   - R_stereo           : 左カメラ座標系から見た右カメラの回転行列（3×3）
    #   - T_stereo           : 左カメラ座標系から見た右カメラの並進ベクトル（3×1）
    #   - E_stereo           : 基本行列（Essential matrix）：回転＋並進の情報のみ（内部パラメータなし）
    #   - F_stereo           : 基礎行列（Fundamental matrix）：内部パラメータも考慮した両画像間の幾何関係    
    ret_stereo, K_stereo_left, D_stereo_left, K_stereo_right, D_stereo_right, R_stereo, T_stereo, E_stereo, F_stereo = cv2.stereoCalibrate(
        matched_world_points_left,
        matched_img_points_left, matched_img_points_right,
        K_mono_left, D_mono_left, K_mono_right, D_mono_right,
        img_size, 
        stereo_criteria, stereo_flags
    )
    
    print(f'Done (4/4): Stereo vision calibration')
    
    print(f'Used image num for stereo calibration = {len(matched_file_ids)} / {len(file_paths_left)}')
    print(f'Used image ratio for stereo calibration = {len(matched_file_ids) / len(file_paths_left) * 100: .1f} %')
    
    # create output dict
    output = {}
    output['img_size'] = img_size
    output['matched_file_ids'] = matched_file_ids
    output['matched_world_points_left'] = matched_world_points_left
    output['matched_img_points_left'] = matched_img_points_left
    output['matched_world_points_right'] = matched_world_points_right
    output['matched_img_points_right'] = matched_img_points_right
    output['detected_corner_ids_left'] = detected_corner_ids_left
    output['detected_corner_points_left'] = detected_corner_points_left
    output['detected_corner_ids_right'] = detected_corner_ids_right
    output['detected_corner_points_right'] = detected_corner_points_right
    output['ret_mono_left'] = ret_mono_left
    output['K_mono_left'] = K_mono_left
    output['D_mono_left'] = D_mono_left
    output['R_mono_left'] = R_mono_left
    output['T_mono_left'] = T_mono_left
    output['new_K_mono_left'] = new_K_mono_left
    output['ret_mono_right'] = ret_mono_right
    output['K_mono_right'] = K_mono_right
    output['D_mono_right'] = D_mono_right
    output['R_mono_right'] = R_mono_right
    output['T_mono_right'] = T_mono_right
    output['new_K_mono_right'] = new_K_mono_right
    output['ret_stereo'] = ret_stereo
    output['K_stereo_left'] = K_stereo_left
    output['D_stereo_left'] = D_stereo_left
    output['K_stereo_right'] = K_stereo_right
    output['D_stereo_right'] = D_stereo_right
    output['R_stereo'] = R_stereo
    output['T_stereo'] = T_stereo
    output['E_stereo'] = E_stereo
    output['F_stereo'] = F_stereo

    # Return values
    return output


def cropImage(roiL, roiR, imgL, imgR):
    # Get common ROI
    # roi = (x, y, width, height)
    x_left = max(roiL[0], roiR[0])
    x_right = min(roiL[0] + roiL[2], roiR[0] + roiR[2])
    y_top = max(roiL[1], roiR[1])
    y_bottom = min(roiL[1] + roiL[3], roiR[1] + roiR[3])
    return imgL[y_top:y_bottom, x_left:x_right], imgR[y_top:y_bottom, x_left:x_right]


def getStereoRectifiedImage(file_path_img_left, file_path_img_right, img_size, 
                            K_mono_left, D_mono_left,
                            K_mono_right, D_mono_right,
                            R_stereo, T_stereo,
                            crop_image=True):

    # Stereo Rectification
    # 両カメラの画像を変換して、対応点が同じy座標上（水平線）に来るようにする
    # 入力
    #   - xx_mono --> 左右それぞれのカメラパラメータ
    #   - xx_stereo --> 左カメラから見た右カメラの回転,位置
    # 出力
    #   - R1, R2: 左右カメラ画像のrectification用回転行列（画像を整列するための3x3回転行列）
    #   - P1, P2: 左右カメラの新しい投影行列（整列後の仮想カメラの投影行列）
    #        - KITTI形式で言えば P_rect_02, P_rect_03 に相当
    #   - Q: 再投影行列（視差とピクセル位置から3D座標を計算するための4x4行列）
    #   - roiL, roiR: 整列後の有効画像領域（矩形領域：x, y, width, height）    
    R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(K_mono_left, D_mono_left,
                                                      K_mono_right, D_mono_right, img_size, 
                                                      R_stereo, T_stereo, 
                                                      flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    rectify_res = {}
    rectify_res['R_left'] = R1
    rectify_res['R_right'] = R2
    rectify_res['P_left'] = P1
    rectify_res['P_right'] = P2
    rectify_res['Q'] = Q
    rectify_res['roiL'] = roiL
    rectify_res['roiR'] = roiR
    
    # Compute the rectification maps
    undistort_map_left = cv2.initUndistortRectifyMap(K_mono_left, D_mono_left, R1, P1, img_size, cv2.CV_32F)
    undistort_map_right = cv2.initUndistortRectifyMap(K_mono_right, D_mono_right, R2, P2, img_size, cv2.CV_32F)

    # Load images
    img_left = cv2.imread(file_path_img_left, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(file_path_img_right, cv2.IMREAD_GRAYSCALE)

    # Get undistorted images
    rectified_img_left = cv2.remap(img_left, *undistort_map_left, cv2.INTER_LANCZOS4)
    rectified_img_right = cv2.remap(img_right, *undistort_map_right, cv2.INTER_LANCZOS4)  
    
    if crop_image:
        rectified_img_left, rectified_img_right = cropImage(roiL, roiR, rectified_img_left, rectified_img_right)
    
    return rectified_img_left, rectified_img_right, rectify_res


def drawEpipolarLines(rectified_img_1, rectified_img_2, F_stereo, top_k=100):
    
    # Detect poitns, 
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(rectified_img_1, None)  # mask = None
    kp2, des2 = sift.detectAndCompute(rectified_img_2, None) # mask = None

    # Use FLANN-based matcher to find matches. FLANN=Fast Library for Approximate Nearest Neighbors
    FLANN_INDEX_KDTREE = 1 # set algorithm id
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50) # checks = search repeat num

    #
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)  

    # Filter matches using the Lowe's ratio test
    good_matches = []
    pts1 = []
    pts2 = []

    # Lowe’s Ratio Test
    for m, n in matches:
        # m=best match, n=2nd best match
        if m.distance < 0.7 * n.distance: # 0.7=strict check
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    # Sort the matches by distance (quality of the match)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Choose the top n matches
    n = top_k  # Set this to the number of top matches you want
    pts1 = np.int32([kp1[m.queryIdx].pt for m in good_matches[:n]])
    pts2 = np.int32([kp2[m.trainIdx].pt for m in good_matches[:n]])

    # Compute epilines corresponding to points in the left image, map to the right image
    lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_stereo)
    lines1 = lines1.reshape(-1, 3)    
    
    
    ''' 
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines 
    '''
    width = rectified_img_1.shape[1]
    
    a_by_b = 0.0
    for line1, pt2 in zip(lines1, pts2):
        a, b, c = line1  # ax + by + c = 0
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -c / b])
        x1, y1 = map(int, [width, -(c + a * width) / b])
        rectified_img_1 = cv2.line(rectified_img_1, (x0, y0), (x1, y1), color, 5)
        rectified_img_2 = cv2.circle(rectified_img_2, (int(pt2[0]), int(pt2[1])), 15, color, -1)
        a_by_b += np.abs(a/b)
        
    a_by_b /= len(lines1)
    print(f'line: ax+by+c = 0 // average of -a/b = {-a/b:.5f}')
    
    plt.figure(figsize=(20, 10))

    # Plot the images with epipolar lines
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rectified_img_2, cv2.COLOR_BGR2RGB))
    plt.title('Points in left image')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(rectified_img_1, cv2.COLOR_BGR2RGB))
    plt.title('Corresponding Epipolar Lines in right image')
    plt.grid()    
    
    # return rectified_img_1, rectified_img_2, a_by_b 


def calcDisparity(rectified_img_1, rectified_img_2, matcher="stereo_sgbm", num_disparities=16*13, block_size=9, window_size=9):
    # Create Matcher
    if matcher == "stereo_bm":
        # num_disparities = 6*16
        stereo_bm_obj = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    elif matcher == "stereo_sgbm":
        '''
        Understand parameters: https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
        '''
        # Create the stereo matcher
        stereo_bm_obj = cv2.StereoSGBM_create(
            minDisparity=0,  # Consider setting this to a positive value if objects aren't very close
            numDisparities=num_disparities,  # Increase disparity range for outdoor scene, e.g., 192 or higher
            blockSize=block_size,  # Smaller block size to capture more detail on glossy surfaces
            P1=8*3*window_size**2,  # Adjust P1 and P2 based on blockSize
            P2=32*3*window_size**2,
            disp12MaxDiff=1,  # Keep low for better consistency
            uniquenessRatio=1,  # Reduce for low-texture objects like black glossy surfaces
            speckleWindowSize=50,  # Reduce to avoid filtering out too many useful points
            speckleRange=1,  # Keep small for low disparity variation
            preFilterCap=40,  # Adjust depending on lighting (keep around 30-63 for outdoor scenes)
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Keep 3-way mode for higher accuracy
            )

    # Calculate Disparity map
    disparity_map = stereo_bm_obj.compute(rectified_img_1, rectified_img_2).astype(np.float32)/16

    return disparity_map


def calcPointCloudRange(vertices):
    # Get extents of point cloud
    pcd_range = np.zeros((3, 2))
    pcd_range[:, 0] = np.min(vertices, axis=0)  # min of x,y,z
    pcd_range[:, 1] = np.max(vertices, axis=0)  # max of x,y,z
    return pcd_range


def calcPointCloud(disparity_map, rectfied_img_left_rgb, z_min=0.0, z_max=20.0,
                   file_path_ply='my_point_cloud.ply'):

    # Normalize the disparity map for visualization
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)
    
    # Reconstruct points to 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    
    # Mask out points with zero disparity (infinite depth)
    mask = disparity_map > disparity_map.min()
    output_points = points_3D[mask]
    output_colors = rectfied_img_left_rgb[mask]
    
    # Calculate point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output_points)
    pcd.colors = o3d.utility.Vector3dVector(output_colors.astype(float)/255.0)
    
    # Convert point cloud to numpy arrays for processing
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)    
    
    # Create a mask for filtering based on the z values
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    
    # Apply the mask to both points and colors
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    # Create a new point cloud from the filtered points and colors
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    # Save point cloud data to view point cloud with MeshLab
    o3d.io.write_point_cloud(file_path_ply, filtered_pcd)
       
    return filtered_pcd


def displayPointCloud(filtered_pcd):
    # Convert the point cloud to numpy arrays
    points = np.asarray(filtered_pcd.points)
    colors = np.asarray(filtered_pcd.colors)
    
    points = points.astype(np.float32)
    colors = colors.astype(np.float32)    
    
    # Calculate data extentes and center
    center = points.mean(axis=0)
    pcd_range = calcPointCloudRange(points)
    
    max_delta = np.max( pcd_range[:, 1] - pcd_range[:, 0] )
    camera_pos = [center[i] + 4*max_delta for i in range(3)]
    #lgiht_pos = [center[i] + (i+3)*max_delta for i in range(3)]
    
    # Set up a scene and render it:
    camera = p3js.PerspectiveCamera(position=camera_pos, fov=20)
    camera.up = (0,0,1)
    
    # Create a Three.js point cloud object
    geometry = p3js.BufferGeometry(
        attributes={
            'position': p3js.BufferAttribute(points, normalized=False),
            'color': p3js.BufferAttribute(colors, normalized=True)
        }
    )    
    
    material = p3js.PointsMaterial(vertexColors='VertexColors', size=0.03)
    points_cloud = p3js.Points(geometry=geometry, material=material) 
    
    # Set up the scene and camera
    # camera = p3js.PerspectiveCamera(position=[0, -10, 3], fov=75)
    scene = p3js.Scene(children=[points_cloud, camera, p3js.AmbientLight()])
    controller = p3js.OrbitControls(controlling=camera)

    # Set up the renderer
    renderer = p3js.Renderer(camera=camera, scene=scene, controls=[controller], width=800, height=600)

    # Display the point cloud
    display(renderer)


def readRawDataCalibFile(file_path):
    calib_params = {}

    with open(file_path, 'r') as f:
        for line in f:
            if not line or ':' not in line or 'calib_time' in line:
                continue

            key, value_str = line.split(':', 1)
            values = list(map(float, value_str.strip().split()))

            if key.startswith(('K', 'R')):
                # K = Intrinsic Matrix, R = Rotation Matrix
                calib_params[key] = np.array(values).reshape(3, 3)
            elif key.startswith(('P')):
                # P = Projection Matrix
                calib_params[key] = np.array(values).reshape(3, 4)
            elif key.startswith(('T')):
                # T = Translation Matrix
                calib_params[key] = np.array(values).reshape(3, 1)
            elif key.startswith(('S')):
                # S = Shape 
                calib_params[key] = np.array(values).reshape(1, 2) # (width, height)
            elif key.startswith(('D')):
                # D = Distortion Matrix
                calib_params[key] = np.array(values).reshape(1, 5)       
            else:
                calib_params[key] = np.array(values)

    return calib_params


def readObjectCalibFile(file_path):
    calib_params = {}

    with open(file_path, 'r') as f:
        for line in f:
            if not line or ':' not in line:
                continue

            key, value_str = line.split(':', 1)
            values = list(map(float, value_str.strip().split()))

            if key.startswith(('P', 'Tr')):
                # P = Projection Matrix
                # Tr = [R | t] Matrix
                calib_params[key] = np.array(values).reshape(3, 4)
            elif key.startswith(('R0_rect')):
                # R0_rect = Rectification Rotation Matrix
                calib_params[key] = np.array(values).reshape(3, 3)   
            else:
                calib_params[key] = np.array(values)

    return calib_params

