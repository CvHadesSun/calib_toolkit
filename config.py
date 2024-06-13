import os
import argparse


class CONFIG:
    chessboard_size = (11, 8)
    square_size = 20  # mm

    root_dir = None

    output_path = None

    cam_num = 12

    thread_num = 5

    tmp_extr_json = rf"{output_path}\tmp_extr_new.json"

    os.makedirs("output", exist_ok=True)


def config_args():

    parser = argparse.ArgumentParser(description="calib toolkit box.")
    parser.add_argument('-s', '--src_path', type=str,
                        help='the calib camera data root dir', required=True)
    parser.add_argument('-o', '--output_path', type=str,
                        help='the calib camera data root dir', default="./output")

    parser.add_argument('-b', '--chessboard', nargs='+',
                        help='the chessboard size : (rows,cols)', default=(11, 8))

    parser.add_argument('-l', '--square_length', type=int,
                        help='the chessboard square size (mm)', default=20)

    parser.add_argument('-n', '--cam_num', type=int,
                        help='the number of needed to calib', default=12)

    parser.add_argument('-t', '--thread_num', type=int,
                        help='the multithred pool size', default=5)

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')

    parser.add_argument('--with_refine', action='store_true',
                        help='with 3d pts refine processing.')

    parser.add_argument('--undistort', action='store_true',
                        help='undistort the rgb and depth')

    parser.add_argument('--operation', type=int,
                        help='the calib operation: (0: calib), (1: error inference), (2: pcd vis)', default=0)

    parser.add_argument('--frame_id', type=int,
                        help='the pcd vis frame id', default=0)
    parser.add_argument('--cam_ids',  nargs='+',
                        help='vis which camera pcd', default=[])

    args = parser.parse_args()

    return args
