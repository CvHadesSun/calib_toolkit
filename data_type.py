import numpy as np


class CamType:
    OBC = 0
    OAK = 1
    Unknow = 2

class Camera:
    cam_type = CamType.OBC
    cam_name = None
    cam_intr = np.eye(3)
    cam_extr = np.eye(4)
    cam_dist = np.ones(4)