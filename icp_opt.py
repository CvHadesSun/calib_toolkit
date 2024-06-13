# TODO: open3d icp refine .?? 

import cv2
import open3d as o3d
import numpy as np


class ICPOptimizer:
    
    def __init__(self,init_pose,cam0_data,cam1_data,cam_intr0,cam_intr1) -> None:
        
        self.cam0_depths = cam0_data
        self.cam1_depths = cam1_data

        self.init_pose = init_pose

        self.cam_intr0 = cam_intr0['intr']
        self.cam_intr1 = cam_intr1['intr']

        self.cam0_dist = cam_intr0['dist']
        self.cam1_dist = cam_intr1['dist']

        self.cam0_pcds=[]
        self.cam1_pcds=[]


    @staticmethod
    def undistort_depth_image(depth_img,intr,dist):

        pass

    @staticmethod
    def depth2pcd(depth_image,intrinsic_matrix):
        """
        Convert a depth image to a point cloud using the camera's intrinsic parameters.
        
        :param depth_image: numpy array of shape (H, W) containing depth values.
        :param intrinsic_matrix: numpy array of shape (3, 3) containing camera intrinsic parameters.
        :return: Open3D point cloud.
        """
        height, width = depth_image.shape
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - cx) / fx
        y = (y - cy) / fy

        z = depth_image / 1000.0  # Assuming depth is in millimeters and converting to meters
        x = np.multiply(x, z)
        y = np.multiply(y, z)

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        
        return point_cloud
    

    def get_processed_data(self):

        for left_img,right_img in zip(self.cam0_depths,self.cam1_depths):
            undisted_left_img = ICPOptimizer.undistort_depth_image(left_img,self.cam_intr0,self.cam0_dist)
            undisted_right_img = ICPOptimizer.undistort_depth_image(right_img,self.cam_intr1,self.cam1_dist)

            pcd1 = ICPOptimizer.depth2pcd(undisted_left_img, self.cam_intr0)
            pcd2 = ICPOptimizer.depth2pcd(undisted_right_img, self.cam_intr1)
            self.cam0_pcds.append(pcd1)
            self.cam1_pcds.append(pcd2)


        self.num_data = len(self.cam1_pcds)


    def run(self):
        for i in range(self.num_data):
            pcd1 = self.cam0_pcds[i]
            pcd2 = self.cam1_pcds[i]
            initial_transformation = np.linalg.inv(self.init_pose)

            # Perform ICP
            threshold = 0.02  # distance threshold for ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                pcd2, pcd1, threshold, initial_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            refined_transformation = np.linalg.inv(icp_result.transformation)
            self.init_pose = refined_transformation
