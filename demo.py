# TODO:
"""
1. extract the points (features or chessboard)
2. matching
3. get transformed camera poses
4. refine poses using depth?

"""

from config import CONFIG as cfg

from config import config_args

import os

from log_utils import *

from CamCalib import Caliber

import _global

from utils import *

from tools import *
import time

_global._init()


def main(args):

    cfg.root_dir = args.src_path
    cfg.output_path = args.output_path
    cfg.cam_num = args.cam_num
    cfg.thread_num = args.thread_num

    os.makedirs("output", exist_ok=True)

    print_info_blue('x'*20+"  Config  "+'x'*20, output=True)

    for k in args.__dict__:
        print_info_green(k + ": " + str(args.__dict__[k]), output=True)

    print_info_blue('x'*50, output=True)

    # init caliber

    OP = args.operation

    if args.verbose:
        _global.set_value('verbose', True)
    else:
        _global.set_value('verbose', False)

    if args.with_refine:
        _global.set_value('refine', True)
    else:
        _global.set_value('refine', False)

    if args.undistort:
        _global.set_value('undistort', True)
    else:
        _global.set_value('undistort', False)

    calib_handle = Caliber(cfg)

    intr_dir = f"{cfg.output_path}/cam_intrs.json"

    # TODO: write intrs into file.
    output_intrs(calib_handle.cam_intrs, intr_dir)

    extr_dir = f"{cfg.output_path}/tmp_extr.json"

    #
    if OP == 0:
        print_info_blue('>> OP 0:Calib camera and refine...', output=True)

        failed_dir = f"{cfg.output_path}/tmp_calib.txt"

        if not os.path.exists(failed_dir):
            comb_manual = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                           [0, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]  # can modify .

        else:  # calib failed camera.
            comb_manual = read_failed_calib(failed_dir)

        calib_handle.set_calib_combinations(comb_manual)

        calib_handle.run_multi()
        # calib_handle.process_two_cam(['Cam4', 'Cam5'])

        write_extr(calib_handle.cam_extrs_2, extr_dir)

        failed_list = calib_handle.get_calib_failed()

        calib_error = calib_handle.calib_errors_rms

        out_name = str(int(time.time()*1e+3))+'.json'

        write_calib_error(calib_error, f"{args.output_path}/{out_name}")

        if len(failed_list):

            print_info_red(
                f"Failed calib camera pairs: {failed_list}", output=True)

            write_failed_calib(failed_list, failed_dir)

        else:
            if os.path.exists(failed_dir):
                os.remove(failed_dir)  # remove calib faield tmp file.
            # TODO: all camera calibed and compute extrs.
            ori_data = get_ori_extrs(extr_dir)
            final_extrs = compute_extrs_handle(ori_data)
            # TODO: write into file.
            extr_out = f"{cfg.output_path}/cam_extrs.json"
            output_extrs(final_extrs, extr_out)
    elif OP == 1:
        print_info_blue(
            '>> OP 1:compute the mean error for all rgbd cameras.', output=True)

        # TODO: if camera all calib sucess

        ori_data = get_ori_extrs(extr_dir)
        final_extrs = compute_extrs_handle(ori_data)
        calibed_cam = get_all_calibed(final_extrs)

        if len(calibed_cam) < cfg.cam_num:
            print_info_blue(
                f'check not calibed camera, already calibed are {calibed_cam} ', output=True)

        else:
            calib_handle.set_global_extrs(final_extrs)
            calib_handle.compute_error_multi(num_tasks=10)
            # calib_handle.bundle_adjust(['Cam0', "Cam1"])
            mean_error = compute_error_mean(calib_handle.refine_errors)

            if mean_error < 5.0:
                print_info_green(
                    f"the all cameras mean error: {mean_error}", output=True)

            else:
                print_info_red(
                    f"the all cameras mean error: {mean_error}", output=True)

        # compute error.

    elif OP == 2:
        print_info_blue('>> OP 2:combine pcd from all cameras.', output=True)
        ori_data = get_ori_extrs(extr_dir)

        # calibed_cam = get_all_calibed(final_extrs)

        cam_data = calib_handle.dataset.cam_data

        # if len(calibed_cam) < cfg.cam_num:
        if 0:
            print_info_blue(
                f'check not calibed camera, already calibed are {calibed_cam} ', output=True)

        else:
            frame_id = args.frame_id
            if len(args.cam_ids) == 0:
                final_extrs = compute_extrs_handle(ori_data)
            else:
                final_extrs = {}
                cam0 = args.cam_ids[0]
                cam1 = args.cam_ids[1]
                final_extrs[cam0] = np.eye(4)
                final_extrs[cam1] = np.array(ori_data[f"{cam0}_{cam1}"])

            for key in final_extrs.keys():
                img = cam_data[key]['rgbs'][frame_id]
                base_name = os.path.basename(img).replace('bmp', 'png')

                intr = np.array(calib_handle.dataset.cam_intrs[key]['intr'])
                dist = np.array(calib_handle.dataset.cam_intrs[key]['dist'])
                depth = cam_data[key]['depths'][frame_id]

                bmp_img = cv2.imread(img)
                depth_img = cv2.imread(depth, -1)

                if args.undistort:
                    undist_img, undist_depth = undist_rgbd(
                        dist, intr, bmp_img, depth_img)
                else:
                    undist_img = bmp_img
                    undist_depth = depth_img

                cv2.imwrite(f'{cfg.output_path}/{base_name}', undist_img)
                cv2.imwrite(
                    f"{cfg.output_path}/depth_{base_name}", undist_depth)

                new_img = f'{cfg.output_path}/{base_name}'
                new_depth = f"{cfg.output_path}/depth_{base_name}"

                extr = np.array(final_extrs[key])

                out_name = f"{cfg.output_path}/{key}.ply"

                img2pcd(new_img, new_depth, intr, extr, out_name)
                os.remove(new_depth)
                os.remove(new_img)

    else:
        print_info_red(
            f">> OP {OP} not support, please check:(0: calib), (1: error inference), (2: pcd vis)", output=True)


if __name__ == "__main__":

    args = config_args()

    main(args)
