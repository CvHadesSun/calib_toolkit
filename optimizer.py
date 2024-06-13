# TODO : use non-linear to opt pose
from scipy.optimize import least_squares, minimize
import numpy as np
from log_utils import *
from utils import export_pcd
import os


class NonLinearOptimizer:
    def __init__(self, init_pose, pts0, pts1, cams, iteration=10) -> None:

        self.init_pose = init_pose
        self.pts0, self.pts1 = combine_pts(pts0, pts1)
        self.iter_num = iteration

        self.opt_params = np.zeros(12)

        self.opt_params[:9] = self.init_pose[:3, :3].reshape(-1).tolist()
        self.opt_params[9:] = self.init_pose[:3, 3].reshape(-1).tolist()

        self.cam0 = cams[0]
        self.cam1 = cams[1]

        self.data_size = self.pts0.shape[0]

    @staticmethod
    def loss_func(params, pts_0, pts_1):
        R_1_0 = np.array(params[:9]).reshape(3, 3)
        # print(R_0_1)
        t = np.array(params[9:])

        pts_0 = pts_0.T
        pts_10 = R_1_0 @ pts_1.T + t.reshape(3, 1)  # [3,N]

        dist = np.sqrt(np.sum((pts_10.T-pts_0.T)**2, axis=1))  # m
        # filterd_error = np.where(dist < 20*1e-3, dist, 0.0)
        error = dist.mean()
        return error

    def error_valid(self, params, pts_0, pts_1, export_flag=False):
        R_1_0 = np.array(params[:9]).reshape(3, 3)
        t = np.array(params[9:])
        pts_0 = pts_0.T
        pts_10 = R_1_0 @ pts_1.T + t.reshape(3, 1)  # [3,N]

        if export_flag:
            tmp_dir = f"./output/"
            os.makedirs(tmp_dir, exist_ok=True)
            export_pcd(f"{tmp_dir}/{self.cam0}.ply", pts_0.T)
            export_pcd(f"{tmp_dir}/{self.cam1}.ply", pts_10.T)

        dist = np.sqrt(np.sum((pts_10.T-pts_0.T)**2, axis=1))  # m
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
            errors.append(ab_error*1e+3)
            # break
            if ab_error < 5*1e-3:
                break
            # result = least_squares(
            #     NonLinearOptimizer.loss_func, self.opt_params, args=(pts0, pts1))
            result = minimize(NonLinearOptimizer.loss_func,
                              self.opt_params, args=(pts0, pts1), method='BFGS')
            optimized_params = result.x
            self.opt_params = optimized_params

        return errors


def combine_pts(pts0, pts1):

    key0 = pts0.keys()
    key1 = pts1.keys()

    min_list = min([key0, key1], key=len)

    pts_0 = []
    pts_1 = []

    for item in min_list:

        p0 = pts0[item]
        p1 = pts1[item]

        # var0 = max(np.var(p0,axis=0))
        # var1 = max(np.var(p1,axis=0))

        p0, p1 = get_valid_pts_v2(p0, p1)
        # if var0<1.0 and var1<1.0:
        pts_0.append(p0)
        pts_1.append(p1)
        # break

    # TODO: split detection pts to more split distrbution, or not just use one frame is fine.
    np_pts0 = np.concatenate(pts_0, axis=0)
    np_pts1 = np.concatenate(pts_1, axis=0)

    return np_pts0, np_pts1


def get_valid_pts(pts0, pts1):

    first_point0 = pts0[0]
    dist0 = np.linalg.norm(pts0 - first_point0, axis=1)

    first_point1 = pts1[0]
    dist1 = np.linalg.norm(pts1 - first_point1, axis=1)

    index0 = np.where(dist0 < 0.1)
    index1 = np.where(dist1 < 0.1)

    index = index0 and index1

    p0 = pts0[index]
    p1 = pts1[index]

    return p0, p1


def get_valid_pts_v2(pts0, pts1):

    index0 = np.where(pts0[:, -1] < 500)
    index1 = np.where(pts1[:, -1] < 500)

    index = index0 and index1

    p0 = pts0[index]
    p1 = pts1[index]

    return p0, p1
