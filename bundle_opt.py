
from dataset import Dataset
from scipy.optimize import least_squares, minimize
import numpy as np
from log_utils import *
from utils import export_pcd
import os
from optimizer import combine_pts


class BundleAdjust:

    def __init__(self, pose0, pose1, cam0, cam1, pts0, pts1, iteration=10) -> None:

        self.pose0 = pose0
        self.pose1 = pose1

        self.cam0 = cam0
        self.cam1 = cam1

        self.iter_num = iteration

        self.pts0, self.pts1 = combine_pts(pts0, pts1)

        self.opt_params = np.zeros(12)  # 9+9+3+3

        self.opt_params[:9] = self.pose1[:3, :3].reshape(-1).tolist()

        self.opt_params[9:12] = self.pose1[:3, 3].reshape(-1).tolist()

        # self.opt_params[12:21] = self.pose1[:3, :3].reshape(-1).tolist()

        # self.opt_params[21:] = self.pose1[:3, 3].reshape(-1).tolist()

        self.data_size = self.pts0.shape[0]

    # @staticmethod
    def loss_func(self, params, pts_0, pts_1):

        parms1 = params[:12]
        # parms1 = params[12:]

        R0 = self.pose0[:3, :3]
        t0 = self.pose0[:3, 3].reshape(3, 1)

        R1 = np.array(parms1[:9]).reshape(3, 3)
        t1 = np.array(parms1[9:]).reshape(3, 1)

        pw0 = R0 @ pts_0.T + t0
        pw1 = R1 @ pts_1.T + t1

        dist = np.sqrt(np.sum((pw0.T-pw1.T)**2, axis=1))  # m
        # filterd_error = np.where(dist < 20*1e-3, dist, 0.0)
        error = dist.mean()
        return error

    def error_valid(self, params, pts_0, pts_1, export_flag=False):
        parms1 = params[:12]
        # parms1 = params[12:]

        R0 = self.pose0[:3, :3]
        t0 = self.pose0[:3, 3].reshape(3, 1)

        R1 = np.array(parms1[:9]).reshape(3, 3)
        t1 = np.array(parms1[9:]).reshape(3, 1)

        pw0 = R0 @ pts_0.T + t0
        pw1 = R1 @ pts_1.T + t1

        if export_flag:
            tmp_dir = f"./output/"
            os.makedirs(tmp_dir, exist_ok=True)
            export_pcd(f"{tmp_dir}/{self.cam0}.ply", pw0.T)
            export_pcd(f"{tmp_dir}/{self.cam1}.ply", pw1.T)

        dist = np.sqrt(np.sum((pw0.T-pw1.T)**2, axis=1))  # m
        filterd_error = np.where(dist < 20*1e-3, dist, 0.0)
        error = filterd_error.mean()
        abs_error = dist.mean()
        return error, abs_error

    def run(self):

        if self.pts0.shape[0] < 1:
            print_info_red("no valid 3d pts for cams")
            return
        errors = []

        for iter in range(self.iter_num):

            random_index = np.random.randint(0, self.data_size-1, 200)

            pts0 = self.pts0[random_index]
            pts1 = self.pts1[random_index]
            error, ab_error = self.error_valid(
                self.opt_params, pts0, pts1, export_flag=True)
            errors.append(error*1e+3)
            break
            # if ab_error < 5*1e-3:
            if ab_error < 5*1e-3:
                break
            # result = least_squares(
            #     NonLinearOptimizer.loss_func, self.opt_params, args=(pts0, pts1))
            result = minimize(self.loss_func,
                              self.opt_params, args=(pts0, pts1), method='BFGS')
            optimized_params = result.x
            self.opt_params = optimized_params

            # break
            # if error < 5*1e-3:
            #     break
        return errors

    def get_refined_poses(self):

        params = self.opt_params
        parms1 = params[:12]
        # parms1 = params[12:]

        # R0 = np.array(parms0[:9]).reshape(3, 3)
        # t0 = np.array(parms0[9:])

        R1 = np.array(parms1[:9]).reshape(3, 3)
        t1 = np.array(parms1[9:])

        refined_T0 = self.pose0
        refined_T1 = np.eye(4)

        # refined_T0[:3, :3] = R0
        # refined_T0[:3, 3] = t0

        refined_T1[:3, :3] = R1
        refined_T1[:3, 3] = t1

        return refined_T0, refined_T1
