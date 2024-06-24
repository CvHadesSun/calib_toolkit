import numpy as np
from itertools import combinations
import os
import cv2
import xml.etree.ElementTree as ET
import trimesh
import json


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
        section_data['ImageSize'] = tuple(
            map(int, child.find('ImageSize').text.split()))

        intrinsic_element = child.find('Intrinsic')
        section_data['Intrinsic'] = extract_matrix_data(intrinsic_element)

        distortion_element = child.find('Distortion')
        section_data['Distortion'] = extract_matrix_data(distortion_element)

        data[child.tag] = section_data
    return data


def load_camera(dir):

    tree = ET.parse(dir)
    root = tree.getroot()
    data = extract_xml_data(root)
    # data = extract_new_xml_data(root)
    data = data['Color']
    intr = np.array(data['Intrinsic']['data']).reshape(3, 3)
    dist = np.array(data["Distortion"]['data'])

    return intr, dist


def get_camera_intr(dir):

    cam_files = os.listdir(dir)
    cam_intrs = {}

    for cam in cam_files:
        if 'xml' in cam:
            cam_name = cam.split('-')[0]
            intr, dist = load_camera(f"{dir}/{cam}")
            cam_intrs[cam_name] = {
                'intr': intr,
                'dist': dist
            }
    return cam_intrs


def cvtid(cam):
    return int(cam[3:])


def invercvtid(cam_id):
    return f"Cam{cam_id}"


def get_final_extrs(json_dir, cam_num):

    with open(json_dir, 'r') as fp:
        data = json.load(fp)
        fp.close()

    result = compute_extrs(data, cam_num)

    return result, data


def get_ori_extrs(json_dir):

    with open(json_dir, 'r') as fp:
        data = json.load(fp)
        fp.close()

    return data


def compute_extrs_handle(data):

    cams_extrs = {}

    cams_extrs["Cam0"] = np.eye(4)

    for i in range(1, 6):
        cams_extrs[f"Cam{i}"] = cams_extrs[f"Cam{i-1}"] @ np.array(
            data[f"Cam{i-1}_Cam{i}"])

    cams_extrs["Cam6"] = cams_extrs["Cam0"]  @ np.array(data["Cam0_Cam6"])

    for i in range(7, 12):
        cams_extrs[f"Cam{i}"] = cams_extrs[f"Cam{i-1}"] @ np.array(
            data[f"Cam{i-1}_Cam{i}"])

    cams_extrs[f"Cam{12}"] = np.array(data[f"Cam{0}_Cam{12}"])

    for i in range(13, 16):
        cams_extrs[f"Cam{i}"] = cams_extrs[f"Cam{i-1}"] @ np.array(
            data[f"Cam{i-1}_Cam{i}"])

    return cams_extrs


def compute_extrs(combinations, cam_num):

    cam0_cam1 = {}
    cam1_cam0 = {}
    center_id = 0
    cam_num += 1

    for key in combinations.keys():  # extr : right->left.
        cam0 = key.split('_')[0]
        cam1 = key.split('_')[1]

        cam0_cam1[cam0] = cam1
        cam1_cam0[cam1] = cam0

    final_extrs = {}
    center_cam = invercvtid(center_id)
    final_extrs[center_cam] = np.eye(4).astype(np.float32)
    not_combined_id = []

    for i in range(1, cam_num-1):
        cam = invercvtid(i)
        if cam in cam1_cam0:
            cam1 = cam
            cam0 = cam1_cam0[cam1]
            if cam1 not in final_extrs:
                d_key = f"{cam0}_{cam1}"
                if d_key in combinations.keys():
                    pose = combinations[d_key]

                    if cam0 in final_extrs:
                        final_extrs[cam1] = pose @ final_extrs[cam0]
                    else:
                        not_combined_id.append(cvtid(cam1))

    while 1:
        tmp_not_comb = []
        for i in not_combined_id:
            cam = invercvtid(i)
            if cam in cam1_cam0:
                cam1 = cam
                cam0 = cam1_cam0[cam1]
                if cam1 not in final_extrs:
                    d_key = f"{cam0}_{cam1}"
                    if d_key in combinations.keys():
                        pose = combinations[d_key]

                        if cam0 in final_extrs:
                            final_extrs[cam1] = pose @ final_extrs[cam0]
                        else:
                            tmp_not_comb.append(cvtid(cam1))

        if len(tmp_not_comb) == 0:
            break
        else:
            not_combined_id = tmp_not_comb

    return final_extrs


def combine_cameras(cam_num):

    l = [x for x in range(cam_num)]

    results = list(combinations(l, 2))

    return results


def imgname2tm(imgname):

    tm = int(imgname.split('-')[-1].split('.')[0])
    return tm


def split_color_depth(folder):

    files = os.listdir(folder)

    rgbs = []
    depths = []

    for item in files:

        if 'color' in item:
            rgbs.append(item)
        else:
            depths.append(item)
    rgbs = sorted(rgbs, key=imgname2tm)
    depths = sorted(depths, key=imgname2tm)

    new_rgbs = get_full_path(folder, rgbs)
    new_depths = get_full_path(folder, depths)

    return new_rgbs, new_depths


def get_full_path(folder, files):
    new_list = []

    for item in files:
        new_list.append(f"{folder}/{item}")

    return new_list


def get_batch_tm(files):

    tms = []

    for item in files:
        name = os.path.basename(item)
        tms.append(imgname2tm(name))

    return min(tms), max(tms)


def undist_rgbd(coffs, intr, rgb, depth):

    w, h, _ = rgb.shape

    mapx_color, mapy_color = cv2.initUndistortRectifyMap(
        intr, coffs, None, None, (h, w), cv2.CV_32FC1)
    mapx_depth, mapy_depth = cv2.initUndistortRectifyMap(
        intr, coffs, None, None, (h, w), cv2.CV_32FC1)

    # Apply the rectification maps to the color image
    color_image_rect = cv2.remap(
        rgb, mapx_color, mapy_color, cv2.INTER_NEAREST)

    # Apply the rectification maps to the depth image
    depth_image_rect = cv2.remap(
        depth, mapx_depth, mapy_depth, cv2.INTER_NEAREST)

    # a2 = 0.5
    # a1 = 0.3
    # # result = cv2.addWeighted(rgb,a1,color_image_rect,a2,0)
    # result = merge_rgbd(color_image_rect,depth_image_rect)

    # cv2.imshow("debug",result)
    # cv2.waitKey(-1)
    return color_image_rect, depth_image_rect


def detect_corners(img, size):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chessboard_size = size

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Refine the corner locations
        criteria = (cv2.TermCriteria_EPS +
                    cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)

        # cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
        # for i, corner in enumerate(corners_refined):
        #     corner_pos = (int(corner[0][0]), int(corner[0][1]))  # Ensure the corner position is a tuple of integers
        #     cv2.putText(img, str(i), corner_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw the corners on the image
        # cv2.imwrite("chessboard.png",img)

        return corners_refined

    else:
        return None


def ifvalid(d):

    if d >= 0.2 and d <= 5.0:
        return True
    else:
        return False


def get_depth(uv, depth_img):

    uv = uv.astype(np.int64)
    h, w = depth_img.shape

    #
    size = 3 // 2
    count = 0
    sum_d = 0
    # zero_num = 0
    for x in range(uv[0]-size, uv[0]+size, 1):
        for y in range(uv[1]-size, uv[1]+size, 1):
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            dv = depth_img[int(y), int(x)]
            # if dv <= 200:
            #     zero_num += 1
            sum_d += dv
            count += 1

    mean_d = sum_d / count * 1e-3

    if ifvalid(mean_d):
        return mean_d
    else:
        return 1000.0


def img2cam(intr, uvs, depth):

    # uvs = np.array(uvs).reshape(-1,2)

    ds = []
    for i in range(uvs.shape[0]):

        d = get_depth(uvs[i], depth)

        ds.append(d)
    ds = np.array(ds).reshape(-1, 1)
    uvn = np.ones([uvs.shape[0], 3])
    uvn[:, :2] = uvs
    inv_k = np.linalg.inv(intr)
    tmp_cam = inv_k @ uvn.T  # [3,n]
    cam_xyz = tmp_cam.T * ds
    return cam_xyz


def export_pcd(out_path, data):
    point_cloud = trimesh.PointCloud(data)
    point_cloud.export(out_path)


def write_extr(extrs, out_dir):
    if len(extrs) < 1:
        return

    if os.path.exists(out_dir):

        with open(out_dir, 'r') as fp1:
            ori_data = json.load(fp1)
            fp1.close()
    else:
        ori_data = {}

    for key in extrs.keys():
        extrs[key] = extrs[key].tolist()

    for key in ori_data.keys():
        if key not in extrs.keys():
            extrs[key] = ori_data[key]

    with open(out_dir, 'w') as fp:
        json.dump(extrs, fp, indent=4)
        fp.close()


def write_refined_extrs(out_path, data):

    if len(data) < 1:
        return

    for key in data.keys():
        data[key] = data[key].tolist()

    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=4)
        fp.close()


def compute_error_mean(errors):

    error_sum = 0
    count = 0

    for key in errors.keys():
        count += 1
        error_sum += errors[key]

    return error_sum / count


def write_failed_calib(failed_list, out_dir):

    with open(out_dir, 'w') as fp:
        for item in failed_list:
            fp.write(f"{item[0]}-{item[1]}\n")

        fp.close()


def read_failed_calib(file_dir):

    with open(file_dir, 'r') as fp:
        data = fp.readlines()

        fp.close()

    # process

    failed_list = []

    for line in data:
        cam0, cam1 = line.rsplit()[0].split('-')

        c0 = int(cam0[3:])
        c1 = int(cam1[3:])
        failed_list.append([c0, c1])

    return failed_list


def get_all_calibed(extrs):

    calibed = []

    for key in extrs.keys():
        calibed.append(key)
    return calibed


def output_extrs(extrs, out_dir):

    for key in extrs.keys():
        extrs[key] = extrs[key].tolist()

    with open(out_dir, 'w') as fp:
        json.dump(extrs, fp, indent=4)
        fp.close()


def output_intrs(intrs, out_dir):

    for key in intrs.keys():
        intrs[key]['intr'] = intrs[key]['intr'].tolist()
        intrs[key]['dist'] = intrs[key]['dist'].tolist()

    with open(out_dir, 'w') as fp:
        json.dump(intrs, fp, indent=4)
        fp.close()


def write_calib_error(err, out_dir):

    with open(out_dir, 'w') as fp:

        json.dump(err, fp, indent=4)
        fp.close()
