import cv2
import numpy as np
import open3d as o3d
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import json
import trimesh


def extract_matrix_data(element):
    matrix_data = {}
    matrix_data['rows'] = int(element.find('rows').text)
    matrix_data['cols'] = int(element.find('cols').text)
    matrix_data['dt'] = element.find('dt').text
    data_text = element.find('data').text.strip()
    matrix_data['data'] = [float(x) for x in data_text.split()]
    return matrix_data

# Extracting data from the XML
def extract_xml_data(root):
    data = {}
    for child in root:
        section_data = {}
        section_data['ImageSize'] = tuple(map(int, child.find('ImageSize').text.split()))
        
        intrinsic_element = child.find('Intrinsic')
        section_data['Intrinsic'] = extract_matrix_data(intrinsic_element)
        
        distortion_element = child.find('Distortion')
        section_data['Distortion'] = extract_matrix_data(distortion_element)
        
        data[child.tag] = section_data
    return data

def load_camera(dir):

    tree = ET.parse(dir)
    root = tree.getroot()
    # data = extract_xml_data(root)
    data = extract_new_xml_data(root)

    return data


def extract_new_xml_data(root):
    rms = root.find('RMS').text
    left_data = {
        'Idx': root.find('Left/Idx').text,
        'CamName': root.find('Left/CamName').text,
        'Intrinsic': [float(x) for x in root.find('Left/Intrinsic/data').text.split()],
        'Distortion': [float(x) for x in root.find('Left/Distotion/data').text.split()]
    }
    right_data = {
        'Idx': root.find('Right/Idx').text,
        'CamName': root.find('Right/CamName').text,
        'Intrinsic': [float(x) for x in root.find('Right/Intrinsic/data').text.split()],
        'Distortion': [float(x) for x in root.find('Right/Distotion/data').text.split()]
    }
    rotation = [float(x) for x in root.find('Rotation/data').text.split()]
    translation = [float(x) for x in root.find('Translation/data').text.split()]

    return left_data,right_data,rotation,translation

        

def undist_rgbd(coffs,intr,rgb,depth):

    w,h,_ = rgb.shape


    mapx_color, mapy_color = cv2.initUndistortRectifyMap(intr, coffs, None, None, (h,w), cv2.CV_32FC1)
    mapx_depth, mapy_depth = cv2.initUndistortRectifyMap(intr, coffs, None, None, (h,w), cv2.CV_32FC1)

    # Apply the rectification maps to the color image
    color_image_rect = cv2.remap(rgb, mapx_color, mapy_color, cv2.INTER_NEAREST)

    # Apply the rectification maps to the depth image
    depth_image_rect = cv2.remap(depth, mapx_depth, mapy_depth, cv2.INTER_NEAREST)

    # a2 = 0.5
    # a1 = 0.3
    # # result = cv2.addWeighted(rgb,a1,color_image_rect,a2,0)
    # result = merge_rgbd(color_image_rect,depth_image_rect)

    # cv2.imshow("debug",result)
    # cv2.waitKey(-1)
    return color_image_rect,depth_image_rect




def normalize_depth(depth_img):
    """Normalize depth image to 0-255 and convert to uint8."""
    depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    return depth_normalized.astype(np.uint8)

def combine_rgb_depth(rgb_img, depth_img):
    """Combine RGB and depth images into one."""
    # Normalize the depth image
    depth_img_normalized = normalize_depth(depth_img)
    
    # Convert the depth image to a 3-channel grayscale image
    depth_img_3ch = cv2.cvtColor(depth_img_normalized, cv2.COLOR_GRAY2BGR)
    
    # Combine the RGB image with the depth image using weighted sum
    combined_img = cv2.addWeighted(rgb_img, 0.3, depth_img_3ch, 0.7, 0)
    
    return combined_img


def detect_corners(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chessboard_size = (8, 11)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Refine the corner locations
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
        # for i, corner in enumerate(corners_refined):
        #     corner_pos = (int(corner[0][0]), int(corner[0][1]))  # Ensure the corner position is a tuple of integers
        #     cv2.putText(img, str(i), corner_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
        # Draw the corners on the image
        # cv2.imwrite("chessboard.png",img)

        return corners_refined

    else:
        return None

# def load_images(rgb_image_path, depth_image_path):
#     rgb = cv2.imread(rgb_image_path)
#     depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
#     return rgb, depth
def rgbd2pcd(color,depth,intr,extr):
    source_color = o3d.io.read_image(color)
    source_depth = o3d.io.read_image(depth)
    pinhole_cam_intr = intr
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    source_color, source_depth,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,pinhole_cam_intr,extr)
    return pcd


def create_point_cloud(rgb, depth, intrinsic,extr):

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), 
        o3d.geometry.Image(depth),
        depth_scale=50000.0, 
        depth_trunc=1.2, 
        convert_rgb_to_intensity=False)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic,
        extr)

    # pcd.transform(extr)
    
    return pcd

def save_point_cloud(pcd, output_file):
    o3d.io.write_point_cloud(output_file, pcd)


def get_intr_dist(cam):

    color_cam = cam["Color"]
    intr = color_cam['Intrinsic']['data']
    dist =  color_cam['Distortion']['data']

    h = color_cam["ImageSize"][1]
    w = color_cam["ImageSize"][0]

    return h,w,intr,dist

def sort_func(ss):
    _ss = ss.split('.')[0]

    __ss = _ss.split("-")[-1]

    return int(__ss)


def split_data(data_folder):

    files = os.listdir(data_folder)
    files_dat_color = sorted([item for item in files if 'color' in item],key = sort_func )
    files_dat_depth = [item.replace("color","depth").replace("bmp","png") for item in files_dat_color]

    return files_dat_color,files_dat_depth


def ifvalid(d):

    if d>=0.2 and d<=1.5:
        return True
    else:
        return False


def get_depth(uv,depth_img):

    uv = uv.astype(np.int64)

    # 
    size=3 // 2
    count = 0
    sum_d = 0
    for x in range(uv[0]-size,uv[0]+size,1):
        for y in range(uv[1]-size,uv[1]+size,1):
            dv = depth_img[int(y),int(x)]
            sum_d +=dv
            count +=1

    mean_d = sum_d / count *1e-3

    if ifvalid(mean_d):
        return mean_d
    else:
        return -1.0
    
def img2cam(intr,uvs,depth):

    # uvs = np.array(uvs).reshape(-1,2)

    ds = []
    for i in range(uvs.shape[0]):

        d = get_depth(uvs[i],depth)

        ds.append(d)
    ds = np.array(ds).reshape(-1,1)
    uvn = np.ones([uvs.shape[0],3])
    uvn[:,:2] = uvs
    inv_k = np.linalg.inv(intr)
    tmp_cam = inv_k @ uvn.T # [3,n]
    cam_xyz = tmp_cam.T * ds
    return cam_xyz


    



def load_extrs(dir):

    with open(dir,'r') as fp:
        data = json.load(fp)
        fp.close()

    for key in data.keys():
        data[key] = np.arary(data[key]).reshape(4,4)

    return data


def img2pcd(img,depth,intr,extr,output_file):


    # rgb, depth = load_images(img, depth)

    # h,w,_ = rgb.shape

    fx = intr[0,0]
    fy = intr[1,1]
    cx = intr[0,2]
    cy = intr[1,2]
    
    # Assuming the camera intrinsic parameters (for a typical RGB-D camera)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1920,
        height=1080,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy)
    
    inv_extr = np.linalg.inv(extr)
    
    pcd = rgbd2pcd(img, depth, intrinsic,inv_extr)

    save_point_cloud(pcd, output_file)

def cvtcamid(ss):
    return int(ss[3:])

def load_multi_cams(cam_dir):
    all_cams = os.listdir(cam_dir)
    cam_ids=[]
    cams={}
    for item in all_cams:
        if '.xml' in item:
            s = item.split('-')
            cam0 = s[0].split('(')[0]
            cam1 = s[1].split('(')[0]
            cam_ids.append(cam0)
            cam_ids.append(cam1)
            # left,right,rot,t  = load_camera(f"{cam_dir}/{item}")
            cams[f"{cam0}_{cam1}"] = load_camera(f"{cam_dir}/{item}")
    # 
    cam_ids = sorted(list(set(cam_ids)),key=cvtcamid)
    num_cam = len(cam_ids)

    cam_intr={}
    cam_extr={}
    cam_extr["Cam0"]=np.eye(4)
    cam_extr["Cam1"]=np.eye(4)
 
    # 0->5
    for i in range(2,6):
        cam0 = f"Cam{i-1}"
        cam1 = f"Cam{i}"
        left,right,rot,t = cams[f"{cam0}_{cam1}"]

        t = np.array(t).reshape(-1)
        T = np.eye(4)
        T[:3,:3] = np.array(rot).reshape(3,3)
        T[:3,3] = t*1e-3
        cam_extr[f"{cam1}"] = T @ cam_extr[f"{cam0}"]

        if cam0 not in cam_intr.keys():
            cam_intr[cam0] = {
                    'intr' : left["Intrinsic"],
                    "dist": left["Distortion"]
                }
        if cam1 not in cam_intr.keys():

            cam_intr[cam1] = {
                    'intr' : right["Intrinsic"],
                    "dist": right["Distortion"]
                }

    # 
    cam_intr["Cam0"] = cam_intr["Cam1"]

    # 0->6
    left,right,rot,t = cams[f"Cam0_Cam6"]

    t = np.array(t).reshape(-1)
    T = np.eye(4)
    T[:3,:3] = np.array(rot).reshape(3,3)
    T[:3,3] = t*1e-3
    cam_extr["Cam6"] = T

    for i in range(7,12):
        cam0 = f"Cam{i-1}"
        cam1 = f"Cam{i}"
        left,right,rot,t = cams[f"{cam0}_{cam1}"]

        t = np.array(t).reshape(-1)
        T = np.eye(4)
        T[:3,:3] = np.array(rot).reshape(3,3)
        T[:3,3] = t*1e-3
        cam_extr[f"{cam1}"] = T @ cam_extr[f"{cam0}"]

        if cam0 not in cam_intr.keys():
            cam_intr[cam0] = {
                    'intr' : left["Intrinsic"],
                    "dist": left["Distortion"]
                }
        if cam1 not in cam_intr.keys():

            cam_intr[cam1] = {
                    'intr' : right["Intrinsic"],
                    "dist": right["Distortion"]
                }


    left,right,rot,t = cams[f"Cam0_Cam12"]

    t = np.array(t).reshape(-1)
    T = np.eye(4)
    T[:3,:3] = np.array(rot).reshape(3,3)
    T[:3,3] = t*1e-3
    cam_extr["Cam12"] = T

    cam_intr["Cam12"] = {
        'intr' : right["Intrinsic"],
        "dist": right["Distortion"]
    }

    for i in range(13,14):
        cam0 = f"Cam{i-1}"
        cam1 = f"Cam{i}"
        left,right,rot,t = cams[f"{cam0}_{cam1}"]

        t = np.array(t).reshape(-1)
        T = np.eye(4)
        T[:3,:3] = np.array(rot).reshape(3,3)
        T[:3,3] = t*1e-3
        cam_extr[f"{cam1}"] = T @ cam_extr[f"{cam0}"]

        if cam0 not in cam_intr.keys():
            cam_intr[cam0] = {
                    'intr' : left["Intrinsic"],
                    "dist": left["Distortion"]
                }
        if cam1 not in cam_intr.keys():
            cam_intr[cam1] = {
                    'intr' : right["Intrinsic"],
                    "dist": right["Distortion"]
                }


    return cam_intr,cam_extr



def align2key(ss):

    return ss.replace('-','')

def get_common_frame(dir_,ignore_list=[]):
    sub_dirs = os.listdir(dir_)

    same_frames=[]

    data=[]
    for item in sub_dirs:
        if item in ignore_list:continue
        tmp_d = os.listdir(f"{dir_}/{item}")
        data.append(tmp_d)

    while len(same_frames)<=0:

        min_list = min(data,key=len)
        for i in range(len(min_list)):
            value = min_list[i]
            num_valid=0
            for item in data:
                if value in item:
                    num_valid+=1
            if num_valid == len(data):
                same_frames.append(value)
        data.remove(min_list)

    return same_frames

def export_pcd(out_path,data):
    point_cloud = trimesh.PointCloud(data)
    point_cloud.export(out_path)
    

def compute_error_2_cam(xyz0,xyz1):

    delta_d = np.abs(xyz0-xyz1)

    n_dd = np.where(delta_d<=0.05,delta_d,0.0)

    diff_01 = n_dd.mean(1).mean()

    return  diff_01