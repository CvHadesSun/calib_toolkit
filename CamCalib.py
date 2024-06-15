import numpy as np
import cv2
from dataset import Dataset
from log_utils import *
from optimizer import NonLinearOptimizer
from itertools import combinations
import threading
import time
import concurrent.futures
from bundle_opt import BundleAdjust
import _global


class Caliber:

    def __init__(self, cfg) -> None:
        super(Caliber, self).__init__()

        self.cam_nums = cfg.cam_num
        self.chessboard_size = cfg.chessboard_size
        self.square_size = cfg.square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.dataset = Dataset(cfg.root_dir, cfg.cam_num)

        objp = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        self.objp = objp

        self.cam_extrs_2 = {}
        self.rms_threshold = 2.0

        self.atuo_calib_combinations()

        self.cam_intrs = self.dataset.cam_intrs

        self.lock = threading.Lock()
        self.refine_global_extr = None

        self.refine_errors = {}

        self.calib_errors = {}
        self.calib_errors_rms = {}

    def calib_stereo(self, cam0, cam1, only_pair=False):

        print_info_red(f"Calib {cam0} and {cam1} with color imgs...")

        min_v, max_v, delta_len = self.dataset.compute_delta(cam0, cam1)
        print_stereo_camera_data_info(
            f"{cam0}->{cam1}", min_v, max_v, delta_len)

        cam0_data = self.dataset.cam_data[cam0]
        cam1_data = self.dataset.cam_data[cam1]

        objpoints_stereo = []
        imgpoints_left = []
        imgpoints_right = []

        pair_ids = []

        for i in range(len(cam0_data['rgbs'])):
            img_left = cv2.imread(cam0_data["rgbs"][i])
            img_right = cv2.imread(cam1_data["rgbs"][i])
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(
                gray_left, self.chessboard_size, None)

            if not ret_left:
                continue

            ret_right, corners_right = cv2.findChessboardCorners(
                gray_right, self.chessboard_size, None)

            if ret_left and ret_right:
                objpoints_stereo.append(self.objp)
                corners2_left = cv2.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
                corners2_right = cv2.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
                imgpoints_left.append(corners2_left)
                imgpoints_right.append(corners2_right)
                pair_ids.append(i)

            if len(pair_ids) > 20:
                break

        # stereo calib.

        if only_pair:
            return pair_ids

        undist_flag = _global.get_value('undistort')
        if undist_flag:
            cam0_dist = np.array(self.cam_intrs[cam0]["dist"])
            cam1_dist = np.array(self.cam_intrs[cam1]["dist"])
        else:
            cam0_dist = np.array([0.0]*5)
            cam1_dist = np.array([0.0]*5)

        cam0_intr = np.array(self.cam_intrs[cam0]['intr'])

        cam1_intr = np.array(self.cam_intrs[cam1]['intr'])

        if len(objpoints_stereo) < 3:
            print_info_red(f"{cam0} -- {cam1} no valid color img to calib.")
            return 10.0, [], np.eye(4)

        # ret, cam0_intr, cam0_dist, rvecs_l, tvecs_l = cv2.calibrateCamera(
        #     objpoints_stereo, imgpoints_left, gray_left.shape[::-1], None, None)
        # ret, cam1_intr, cam1_dist, rvecs_r, tvecs_r = cv2.calibrateCamera(
        #     objpoints_stereo, imgpoints_right, gray_right.shape[::-1], None, None)

        # self.update_intrs(cam0, cam1, cam0_intr,
        #                   cam0_dist, cam1_intr, cam1_dist)

        rms, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints_stereo, imgpoints_left, imgpoints_right,
            cam0_intr, cam0_dist,
            cam1_intr, cam1_dist,
            gray_left.shape[::-1], criteria=self.criteria)

        pose = np.eye(4).astype(np.float32)

        pose[:3, :3] = R
        pose[:3, 3] = T.reshape(3)*1e-3  # mm to m

        print_info_red(f"the stereo calib result rms: {rms}")

        inv_pose = np.linalg.inv(pose)

        return rms, pair_ids, inv_pose  # rms error

    def update_intrs(self, cam0, cam1, cam0_intr, cam0_dist, cam1_intr, cam1_dist):
        self.cam_intrs[cam0]['intr'] = cam0_intr
        self.cam_intrs[cam1]['intr'] = cam1_intr

        self.cam_intrs[cam1]['dist'] = cam0_dist
        self.cam_intrs[cam1]['dist'] = cam1_dist

        self.dataset.cam_intrs[cam0] = self.cam_intrs[cam0]
        self.dataset.cam_intrs[cam1] = self.cam_intrs[cam1]

    def refine_stereo(self, cam0, cam1, pairs, init_pose):

        print_info_green(
            f"Refine {cam0} and {cam1} with 3d points and depth imgs...")

        # TODO: get 3d pts to refine.

        # init_pose = self.cam_extrs_2[f"{cam0}_{cam1}"]

        undist_flag = _global.get_value('undistort')

        pts0 = self.dataset.get_cam_pts(
            cam0, pairs, self.chessboard_size, undistort=undist_flag)
        pts1 = self.dataset.get_cam_pts(
            cam1, pairs, self.chessboard_size, undistort=undist_flag)

        if len(pts0) < 1 or len(pts1) < 1:
            print_info_red(f"{cam0}--{cam1} no crosspondence and ignore.")
            return np.eye(4), 100

        # refine with 3d pts several iterations.
        opt_handle = NonLinearOptimizer(init_pose, pts0, pts1, [cam0, cam1])
        errors = opt_handle.run()
        print_pts_refine_error(f"{cam0}<-{cam1}", errors)

        refined_pose = opt_handle.opt_params

        R = np.array(refined_pose[:9]).reshape(3, 3)
        t = np.array(refined_pose[9:]).reshape(-1)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        # self.cam_extrs_2[f"{cam0}_{cam1}"] = T

        return T, errors[-1]

        # TODO: use depth icp refine?

        # print error info before icp

        # get undistorted depth img.

        # icp refine

        # print error after icp

        # write result into self.cam_extrs_2

    def get_calib_failed(self):

        failed_list = []

        for pair in self.calib_cams:
            cam0, cam1 = pair

            key_name = f"{cam0}_{cam1}"

            if key_name not in self.cam_extrs_2.keys():
                failed_list.append([cam0, cam1])

        return failed_list

    def atuo_calib_combinations(self,):
        cam_num = self.cam_nums
        cam_lists = []
        for i in range(cam_num):
            cam_lists.append(f"Cam{i}")

        all_combination = list(combinations(cam_lists, 2))

        self.calib_cams = all_combination

    def set_calib_combinations(self, comb_inp):

        self.calib_cams = []

        for item in comb_inp:
            cam0, cam1 = item
            tmp = [f"Cam{cam0}", f"Cam{cam1}"]
            self.calib_cams.append(tmp)

    def process_two_cam(self, cam_pair):
        cam0, cam1 = cam_pair
        rms, corr_frames, pose = self.calib_stereo(cam0, cam1)

        refine_flag = _global.get_value('refine')

        with self.lock:
            self.calib_errors_rms[f"{cam0}_{cam1}"] = {
                "rms": rms
            }

        if rms > 2.0:
            return

        if not refine_flag:
            with self.lock:
                if rms < 2.0:
                    # [right->left]
                    self.cam_extrs_2[f"{cam0}_{cam1}"] = pose
        else:
            refined_T, error = self.refine_stereo(
                cam0, cam1, corr_frames, pose)

            with self.lock:
                self.calib_errors[f"{cam0}_{cam1}"] = {
                    "err": error
                }

            with self.lock:
                self.cam_extrs_2[f"{cam0}_{cam1}"] = refined_T

    def run_multi(self, num_tasks=5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
            future_to_task = {executor.submit(
                self.process_two_cam, cam_pair): cam_pair for cam_pair in self.calib_cams}

            # Process the results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Task {task_id} generated an exception: {exc}")

    def set_global_extrs(self, extrs):

        self.refine_global_extr = extrs

    def bundle_adjust(self, cam_pairs):
        cam0, cam1 = cam_pairs

        # detect chessboard.
        frame_pairs = self.calib_stereo(cam0, cam1, only_pair=True)

        with self.lock:
            pose0 = self.refine_global_extr[cam0]
            pose1 = self.refine_global_extr[cam1]

        print_info_green(
            f"Refine {cam0} and {cam1} with 3d points and depth imgs...")

        # TODO: get 3d pts to refine.

        # init_pose = self.cam_extrs_2[f"{cam0}_{cam1}"]

        pts0 = self.dataset.get_cam_pts(cam0, frame_pairs,self.chessboard_size, undistort=False)
        pts1 = self.dataset.get_cam_pts(cam1, frame_pairs,self.chessboard_size, undistort=False)

        if len(pts0) < 1 or len(pts1) < 1:
            print_info_red(f"{cam0}--{cam1} no crosspondence and ignore.")
            return np.eye(4), 100

        # refine with 3d pts several iterations.
        opt_handle = BundleAdjust(pose0, pose1, cam0, cam1, pts0, pts1)
        errors = opt_handle.run()
        print_pts_refine_error(f"{cam0}<-{cam1}", errors)

        refined_pose0, refined_pose1 = opt_handle.get_refined_poses()

        with self.lock:
            self.refine_errors[f'{cam0}_{cam1}'] = errors[-1]

            # if cam0 not in self.refine_errors:
            #     self.refine_global_extr[cam0] = refined_pose0
            #     self.refine_errors[cam0] = errors[-1]
            # else:
            #     ori_error = self.refine_errors[cam0]
            #     if ori_error > errors[-1]:
            #         self.refine_errors[cam0] = errors[-1]
            #         self.refine_global_extr[cam0] = refined_pose0

            # if cam1 not in self.refine_errors:
            #     self.refine_global_extr[cam1] = refined_pose1
            #     self.refine_errors[cam1] = errors[-1]
            # else:
            #     ori_error = self.refine_errors[cam1]
            #     if ori_error > errors[-1]:
            #         self.refine_errors[cam1] = errors[-1]
            #         self.refine_global_extr[cam1] = refined_pose1

            # done.

    def compute_error_multi(self, num_tasks=5):

        # before run ,need to run auto set calib combinations.

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
            future_to_task = {executor.submit(
                self.bundle_adjust, cam_pair): cam_pair for cam_pair in self.calib_cams}

            # Process the results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"Task {task_id} generated an exception: {exc}")
