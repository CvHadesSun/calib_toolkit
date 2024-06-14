import cv2
import os
from data_type import Camera, CamType

from utils import *
from log_utils import *
import numpy as np


def sort_func(ss):

    return int(ss.split('-')[0][3:])


class Dataset:
    def __init__(self, data_path, cam_num) -> None:

        self.root_dir = data_path

        self.cam_num = cam_num

        self.cam_intrs = {}

        self.cam_data = {}

        self.cam_pts = {}

        self.get_all_data()
        self.load_cam_intr()
        # self.get_3d_pts()

    def get_all_data(self):
        print_info_green("read all camera data.")
        sub_dirs = os.listdir(self.root_dir)

        sub_dirs_ = sorted(
            [x for x in sub_dirs if 'xml' not in x and 'Cam' in x], key=sort_func)

        for i in range(self.cam_num):
            dir_name = f"{self.root_dir}/{sub_dirs_[i]}"

            rgbs, depths = self.get_one_cam(dir_name)
            self.cam_data[f"Cam{i}"] = {
                "rgbs": rgbs,
                "depths": depths
            }

    def load_cam_intr(self):
        print_info_green("read obc camera params from file.")

        cam_params_dir = f"{self.root_dir}"
        self.cam_intrs = get_camera_intr(cam_params_dir)

    def get_one_cam(self, dir):

        rgbs, depths = split_color_depth(dir)

        return rgbs, depths

    def get_cam_data(self, cam_id):

        return self.cam_data[cam_id]

    def compute_delta(self, cam_id1, cam_id2):

        min_0, max_0 = get_batch_tm(self.cam_data[cam_id1]['rgbs'])
        min_1, max_1 = get_batch_tm(self.cam_data[cam_id2]['rgbs'])

        return abs(min_0-min_1), abs(max_0 - max_1), \
            abs(len(self.cam_data[cam_id1]['rgbs']) -
                len(self.cam_data[cam_id2]['rgbs']))

    def get_cam_pts(self, cam_id, frame_ids, chessboard_size, undistort=True):

        print_info_green(
            f"get 3d pts from rgbd for {cam_id}---frame:{frame_ids}")
        rgbs = self.cam_data[cam_id]['rgbs']
        depths = self.cam_data[cam_id]['depths']
        pts = {}
        if len(depths) > 0 and len(depths) == len(rgbs):
            intr = self.cam_intrs[cam_id]['intr']
            dist = self.cam_intrs[cam_id]['dist']

            for i in frame_ids:
                img = cv2.imread(rgbs[i])
                depth = cv2.imread(depths[i], -1)
                if undistort:
                    undist_img, undist_depth = undist_rgbd(
                        dist, intr, img, depth)
                else:
                    undist_img = img
                    undist_depth = depth
                corners = detect_corners(undist_img, chessboard_size)

                if corners is not None:
                    uvs = np.array(corners).reshape(-1, 2).astype(np.float32)
                    cam_xyz = img2cam(intr, uvs, undist_depth)
                    pts[i] = cam_xyz

        return pts

    def get_3d_pts(self):
        print_info_green("get 3d pts from rgbd.")
        for key in self.cam_data.keys():
            rgbs = self.cam_data[key]['rgbs']
            depths = self.cam_data[key]['depths']
            pts = {}
            if len(depths) > 0 and len(depths) == len(rgbs):

                intr = self.cam_intrs[key]['intr']
                dist = self.cam_intrs[key]['dist']
                for i in range(len(depths)):
                    img = cv2.imread(rgbs[i])
                    depth = cv2.imread(depths[i], -1)
                    if 1:
                        undist_img, undist_depth = undist_rgbd(
                            dist, intr, img, depth)
                    else:
                        undist_img = img
                        undist_depth = depth
                    corners = detect_corners(undist_img)

                    if corners is not None:
                        uvs = np.array(corners).reshape(-1,
                                                        2).astype(np.float32)
                        cam_xyz = img2cam(intr, uvs, undist_depth)
                        pts[i] = cam_xyz
            else:
                print_info_red(f"{key} no valid depth imgs to use.")

            self.cam_pts[key] = pts

        # for log output
        cams = []
        frames = []
        for key in self.cam_pts.keys():
            pts = self.cam_pts[key]
            fids = []
            for fid in pts.keys():
                fids.append(fid)
            cams.append(key)
            frames.append(fids)
        print_pts_frame(cams, frames)

    def vis_cam_undist_result(self, cam_id, frame_id):

        rgb_pth = self.cam_data[cam_id]['rgbs'][frame_id]
        depth_pth = self.cam_data[cam_id]['depths'][frame_id]
        img = cv2.imread(rgb_pth)
        depth = cv2.imread(depth_pth, -1)

        intr = self.cam_intrs[cam_id]['intr']
        dist = self.cam_intrs[cam_id]['dist']

        undist_img, undist_depth = undist_rgbd(dist, intr, img, depth)

        return undist_img, undist_depth
