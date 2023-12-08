from os.path import join as pjoin
import os
import cv2
import numpy as np
import os.path as osp
import re
import pickle
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import pandas as pd
from TEASER_SIM_Sync import TEASER_SimSync
import time
import matplotlib.pyplot as plt
import copy
import open3d
from utils import image_io
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from pose_optimization import PoseOptimizer

from utils.utils import getTranslationError, getAngularError
from collections import defaultdict

#######################
#### with keyframe ####
#######################
class PoseOptimizerTUM(PoseOptimizer):
    def __init__(self, gt_path="rgbd_dataset_freiburg2_xyz",\
                    pred_path ="dataset/rgbd_dataset_freiburg2_xyz_output/dpt_beit_large_512_60-260"):

        super().__init__(gt_path, pred_path)

        self.KeyFrame_file = 'KeyFrameTrajectory_xyz2_mono.txt'

        self.use_gt_flow = False # TUM does not have gt flow
        self.use_gt_depth = False
        self.pointCloudPairSeparateDistance = 1
        self.dataset_type = 'TUM'
        self.depth_min = []
        self.depth_max = []

        # Extract sequence name
        pattern = re.compile(r'freiburg(1|2)_(xyz|rpy|teddy|floor)', re.IGNORECASE)
        match = re.search(pattern, self.pred_path)
        if match:
            self.sequence_name = match.group(0)
        else:
            print("No match found.")

        # Extract the range
        match = re.search(r'(\d+-\d+)$', self.pred_path)
        range_value = match.group(1)
        self.start_frame_optimizer, self.end_frame_optimizer = map(int,range_value.split('-'))

        if self.sequence_name == 'freiburg1_xyz':
            self.whole_sequence_len = 798
        elif self.sequence_name == 'freiburg1_teddy':
            self.whole_sequence_len = 1419
        elif self.sequence_name == 'freiburg1_rpy':
            self.whole_sequence_len = 723
        elif self.sequence_name == 'freiburg2_xyz':
            self.whole_sequence_len = 3669
        elif self.sequence_name == 'freiburg1_floor':
            self.whole_sequence_len = 1242

        self.target_width, self.target_height = 640,480

        self.feat_path = 'rgbd_dataset_' + self.sequence_name + '_CAPS/'

        rgb = self.gt_path + '/rgb/'
        rgb_files = os.listdir(rgb)
        self.rgb_files = sorted(rgb_files)
        self.rgb_time_stamp = []
        for i in range(len(self.rgb_files)):

            pattern = r'(\d+\.\d+)'
            # Use regex to find the timestamp in the file name
            match = re.search(pattern, self.rgb_files[i])
            timestamp_str = match.group(1)
            timestamp_float = float(timestamp_str)
            self.rgb_time_stamp.append(timestamp_float)

    def read_imgs(self,i):
        rgb = self.gt_path + '/rgb/'
        rgb_file = rgb + self.rgb_files[i]
        img = cv2.imread(rgb_file)
        return img

    def read_feat(self,i):
        feat_file = self.feat_path + 'rgb-' + self.rgb_files[i] + '.caps'
        feat = np.load(feat_file)
        key_points = feat['keypoints']
        descriptors = feat['descriptors']

        return key_points, descriptors


    def get_intrinsics(self, i):
        if self.sequence_name == 'freiburg1_xyz' \
            or self.sequence_name == 'freiburg1_rpy'\
            or self.sequence_name == 'freiburg1_teddy'\
            or self.sequence_name == 'freiburg1_floor':
            # camera intrinsics
            fx = 517.3  # focal length x
            fy = 516.5  # focal length y
            cx = 318.6  # optical center x
            cy = 255.3  # optical center y
        elif self.sequence_name == 'freiburg2_xyz':
            # camera intrinsics
            fx = 520.9  # focal length x
            fy = 521.0  # focal length y
            cx = 325.1  # optical center x
            cy = 249.7  # optical center y
        return fx, fy, cx, cy


    def get_gt_depth_file_list(self):
        depth_gt = self.gt_path + '/depth/'
        rgb = self.gt_path + '/rgb/'
        depth_files = os.listdir(depth_gt)
        depth_files = sorted(depth_files)
        rgb_files = os.listdir(rgb)
        rgb_files = sorted(rgb_files)

        rgb_time_stamp = []
        for i in range(self.start_frame_optimizer, self.end_frame_optimizer):

            pattern = r'(\d+\.\d+)'
            # Use regex to find the timestamp in the file name
            match = re.search(pattern, rgb_files[i])
            timestamp_str = match.group(1)
            timestamp_float = float(timestamp_str)
            rgb_time_stamp.append(timestamp_float)

        depth_time_stamp = []
        for j in range(len(depth_files)):
            pattern = r'(\d+\.\d+)'
            # Use regex to find the timestamp in the file name
            match = re.search(pattern, depth_files[j])
            timestamp_str = match.group(1)
            timestamp_float = float(timestamp_str)
            depth_time_stamp.append(timestamp_float)

        return depth_files, rgb_time_stamp, depth_time_stamp

    def get_gt_depth(self, depth_filelist, rgb_time_stamp, depth_time_stamp, i):

        time_stamp = rgb_time_stamp[i]
        depth_gt = self.gt_path + '/depth/'

        # If number is smaller than smallest element, return -1 as it's not available
        if time_stamp < depth_time_stamp[0]:
            idx = -1
            depth_file = depth_filelist[idx]
            gt_file = depth_gt + depth_file
            gt_depth = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
            gt_depth = gt_depth.astype(np.float32) / 5000.0  # convert to meters
            gt_depth[gt_depth == 0] = np.nan

        # If number is greater than or equal to the greatest element, return the index of the greatest element
        elif time_stamp >= depth_time_stamp[-1]:
            idx = len(depth_time_stamp) - 1
            depth_file = depth_filelist[idx]
            gt_file = depth_gt + depth_file
            gt_depth = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
            gt_depth = gt_depth.astype(np.float32) / 5000.0  # convert to meters
            gt_depth[gt_depth == 0] = np.nan
        else:
            idx = np.searchsorted(depth_time_stamp, time_stamp) - 1
            depth_file1 = depth_filelist[idx]
            gt_file1 = depth_gt + depth_file1
            gt_depth1 = cv2.imread(gt_file1, cv2.IMREAD_UNCHANGED)
            gt_depth1 = gt_depth1.astype(np.float32) / 5000.0  # convert to meters
            gt_depth1[gt_depth1 == 0] = np.nan

            depth_file2 = depth_filelist[idx+1]
            gt_file2 = depth_gt + depth_file2
            gt_depth2 = cv2.imread(gt_file2, cv2.IMREAD_UNCHANGED)
            gt_depth2 = gt_depth2.astype(np.float32) / 5000.0  # convert to meters
            gt_depth2[gt_depth2 == 0] = np.nan

            time_stamp1 = depth_time_stamp[idx]
            time_stamp2 = depth_time_stamp[idx+1]

            assert time_stamp1 <= time_stamp <= time_stamp2

            factor = (time_stamp - time_stamp1) / (time_stamp2 - time_stamp1)
            # Interpolate the images
            gt_depth = (1 - factor) * gt_depth1 + factor * gt_depth2


        # remove points out of range in gt
        deep_copied_gt_depth = copy.deepcopy(gt_depth)
        deep_copied_gt_depth[np.isnan(deep_copied_gt_depth)] = 0
        threshold = np.percentile(gt_depth, self.percentile_threshold)
        gt_depth = np.where(gt_depth > threshold, np.nan, gt_depth)
        gt_depth = np.where(gt_depth <= 0, np.nan, gt_depth)
        predicted_depth = gt_depth

        return predicted_depth

    def get_gt_traj(self):

        gt_rgb_path = self.gt_path + '/rgb'
        file_names = sorted(os.listdir(gt_rgb_path))

        # Extract the desired time stamps from rgb files
        times = []
        for file_name in file_names:
            # Assuming the time stamp is in the format "timestamp.jpg"
            time_parts = file_name.split(".")
            time_stamp = float(".".join(time_parts[:-1]))
            times.append(time_stamp)

        depth_dir = self.pred_path + '/depth_dpt_beit_large_512/depth/'
        predict_files = sorted(os.listdir(depth_dir))
        number_files = int(len(predict_files)/2)
        self.N = number_files
        desired_times = np.array(times)[self.start_frame_optimizer:self.end_frame_optimizer]

        # Extract gt translation and rotation
        gt_traj_path = self.gt_path + '/groundtruth.txt'
        data = pd.read_csv(gt_traj_path, delimiter=' ', header=None)
        time = data.iloc[:, 0].astype(float).values
        translation = data.iloc[:, 1:4].astype(float).values
        rotation = data.iloc[:, 4:].astype(float).values

        # interpolate rotation
        rotations = R.from_quat(rotation)
        # Add a small offset to duplicate times
        unique_times = []
        epsilon = 1e-6
        for single_time in time:
            while single_time in unique_times:
                single_time += epsilon
            unique_times.append(single_time)
        time = np.array(unique_times)
        slerp = Slerp(time, rotations)
        interpolated_rotation = slerp(desired_times)

        # interpolate translation
        interpolate_translation = interp1d(time, translation, axis=0, kind='linear')
        interpolated_translation = interpolate_translation(desired_times)

        # Anchor the first timestamp
        rotation_matrix = interpolated_rotation.as_matrix()
        rotation_matrix_reshaped = rotation_matrix.reshape(-1, 3)
        first_rotation = rotation_matrix_reshaped[:3, :]
        inverse_first_rotation = first_rotation.T
        self.R_gt = np.zeros((3*len(desired_times), 3))
        for i in range(len(desired_times)):
            self.R_gt[3*i:3*(i+1), :] = inverse_first_rotation @ rotation_matrix_reshaped[3*i:3*(i+1), :]


        self.t_gt = interpolated_translation - interpolated_translation[0]
        self.t_gt = (inverse_first_rotation @ self.t_gt.T).T

    def visFilteredPointclouds(self, weights):

        start_idx = 0
        weights_dict = {}
        for edge in self.scaled_cloud_camera_frame_dict.keys():
            frame1 = edge[0]
            num_points = self.scaled_cloud_camera_frame_dict[edge]["frame{}".format(frame1)].shape[0]

            weights_dict[edge] = weights[start_idx: start_idx+num_points]
            start_idx += num_points

        for edge in self.scaled_cloud_camera_frame_dict.keys():
            frame1 = edge[0]
            frame2 = edge[1]
            # get original points
            original_point_cloud_frame1 = copy.deepcopy(self.original_point_cloud["frame{}".format(frame1)] * self.scale_factor)
            original_point_cloud_frame2 = copy.deepcopy(self.original_point_cloud["frame{}".format(frame2)] * self.scale_factor)
            original_point_cloud_frame2[:,2] = original_point_cloud_frame2[:,2] + self.pointCloudPairSeparateDistance
            original_point_cloud_frame2[:,1] = original_point_cloud_frame2[:,1] + self.pointCloudPairSeparateDistance

            original_point_cloud_rgb_frame1 = self.original_point_cloud_rgb["frame{}".format(frame1)]
            original_point_cloud_rgb_frame2 = self.original_point_cloud_rgb["frame{}".format(frame2)]
            original_pcd1_o3d = open3d.geometry.PointCloud()
            original_pcd2_o3d = open3d.geometry.PointCloud()
            original_pcd1_o3d.points = open3d.utility.Vector3dVector(original_point_cloud_frame1)
            original_pcd1_o3d.colors = open3d.utility.Vector3dVector(original_point_cloud_rgb_frame1)
            original_pcd2_o3d.points = open3d.utility.Vector3dVector(original_point_cloud_frame2)
            original_pcd2_o3d.colors = open3d.utility.Vector3dVector(original_point_cloud_rgb_frame2)


            point_frame1 = self.scaled_cloud_camera_frame_dict[edge]["frame{}".format(frame1)]

            print("Frame pair {} -> {} has {} correspondences before filtering.".format(edge[0],edge[1], len(point_frame1)))


            ######## visualize points that are filtered ########
            left_point_frame1 = self.scaled_cloud_camera_frame_dict[edge]["frame{}".format(frame1)][weights_dict[edge].astype(int)==0,:]
            left_point_frame2 = self.scaled_cloud_camera_frame_dict[edge]["frame{}".format(frame2)][weights_dict[edge].astype(int)==0,:]


            # Create line correspondences
            pcd1_np = copy.deepcopy(left_point_frame1)
            pcd2_np = copy.deepcopy(left_point_frame2)
            pcd2_np[:,2] = pcd2_np[:,2] + self.pointCloudPairSeparateDistance # to seperate pcd1 and pcd2
            pcd2_np[:,1] = pcd2_np[:,1] + self.pointCloudPairSeparateDistance # to seperate pcd1 and pcd2
            num_points = max(len(pcd1_np), len(pcd2_np))
            colors = np.zeros((num_points, 3))
            colors[:, 0] = 1  # Red channel
            colors_o3d = open3d.utility.Vector3dVector(colors)
            ## Create LineSet object
            line_set = open3d.geometry.LineSet()
            line_set.points = open3d.utility.Vector3dVector(np.concatenate((pcd1_np, pcd2_np), axis=0))
            line_set.lines = open3d.utility.Vector2iVector(np.array([[i, i + len(pcd1_np)] for i in range(len(pcd1_np))]))
            line_set.colors = colors_o3d
            ## Create Open3D point cloud objects with colors
            pcd1_o3d = open3d.geometry.PointCloud()
            pcd1_o3d.points = open3d.utility.Vector3dVector(pcd1_np)
            pcd1_o3d.paint_uniform_color([0, 0, 1]) ## blue
            pcd2_o3d = open3d.geometry.PointCloud()
            pcd2_o3d.points = open3d.utility.Vector3dVector(pcd2_np)
            pcd2_o3d.paint_uniform_color([0, 0, 1]) ## blue

            ######## Visualize points that are clean ########
            filtered_point_frame1 = self.scaled_cloud_camera_frame_dict[edge]["frame{}".format(frame1)][weights_dict[edge].astype(int)==1,:]
            filtered_point_frame2 = self.scaled_cloud_camera_frame_dict[edge]["frame{}".format(frame2)][weights_dict[edge].astype(int)==1,:]

            print("Frame pair {} -> {} has {} correspondences after filtering.".format(edge[0],edge[1], len(filtered_point_frame1)))

            # Create line correspondences
            filtered_pcd1_np = copy.deepcopy(filtered_point_frame1)
            filtered_pcd2_np = copy.deepcopy(filtered_point_frame2)
            filtered_pcd2_np[:,2] = filtered_pcd2_np[:,2] + self.pointCloudPairSeparateDistance # to seperate pcd1 and pcd2
            filtered_pcd2_np[:,1] = filtered_pcd2_np[:,1] + self.pointCloudPairSeparateDistance # to seperate pcd1 and pcd2

            num_points = max(len(filtered_pcd1_np), len(filtered_pcd2_np))
            colors = np.zeros((num_points, 3))
            colors[:, 1] = 1  # Green channel
            colors_o3d = open3d.utility.Vector3dVector(colors)
            ## Create LineSet object
            filtered_line_set = open3d.geometry.LineSet()
            filtered_line_set.points = open3d.utility.Vector3dVector(np.concatenate((filtered_pcd1_np, filtered_pcd2_np), axis=0))
            filtered_line_set.lines = open3d.utility.Vector2iVector(np.array([[i, i + len(filtered_pcd1_np)] for i in range(len(filtered_pcd1_np))]))
            filtered_line_set.colors = colors_o3d
            ## Create Open3D point cloud objects with colors
            filtered_pcd1_o3d = open3d.geometry.PointCloud()
            filtered_pcd1_o3d.points = open3d.utility.Vector3dVector(filtered_pcd1_np)
            filtered_pcd1_o3d.paint_uniform_color([0, 0, 1]) ## blue
            filtered_pcd2_o3d = open3d.geometry.PointCloud()
            filtered_pcd2_o3d.points = open3d.utility.Vector3dVector(filtered_pcd2_np)
            filtered_pcd2_o3d.paint_uniform_color([0, 0, 1]) ## blue


            # Visualize the point clouds
            visualizer = open3d.visualization.Visualizer()
            visualizer.create_window()

            # Add the point clouds to the visualizer
            visualizer.add_geometry(pcd1_o3d)
            visualizer.add_geometry(pcd2_o3d)
            visualizer.add_geometry(filtered_pcd1_o3d)
            visualizer.add_geometry(filtered_pcd2_o3d)
            visualizer.add_geometry(original_pcd1_o3d)
            visualizer.add_geometry(original_pcd2_o3d)
            visualizer.add_geometry(line_set)
            visualizer.add_geometry(filtered_line_set)

            axes_scale = 0.1
            visualizer.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame\
                                                            (size=axes_scale, origin=[0, 0, 0]))
            visualizer.get_render_option().point_size = 3

            # Run the visualizer
            visualizer.run()
            visualizer.destroy_window()

    def Reconstruction(self, solution_path):


        with open(solution_path, 'rb') as file:
            solution = pickle.load(file)

        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()
        pcd = open3d.geometry.PointCloud()
        # init rgb and depth file
        depth_dir = self.pred_path + '/depth_dpt_beit_large_512/depth/'
        predict_files = sorted(os.listdir(depth_dir))
        rgb_dir = self.pred_path + '/color_full/'
        rgb_files = sorted(os.listdir(rgb_dir))
        depth_filelist, rgb_time_stamp, depth_time_stamp = self.get_gt_depth_file_list()

        coordinate_frames = []
        for frame in range(self.N):

            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[frame*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape
            # remove points out of range in prediction
            # threshold = np.percentile(predicted_depth, self.percentile_threshold)
            # predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)
            # read intrinsics
            fx, fy, cx, cy = self.get_intrinsics(frame)

            predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, frame)
            if self.use_gt_depth == True:
                # Load the gt depth map
                predicted_depth = predicted_depth_gt

            y, x = np.meshgrid(np.arange(self.target_height), np.arange(self.target_width), indexing="ij")
            points = np.stack((x, y, np.ones((self.target_height, self.target_width))), axis=2)
            points = np.reshape(points,(self.target_height*self.target_width,3))
            predicted_depth = np.reshape(predicted_depth,(self.target_height*self.target_width,1))
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (points[:,0] - cx) * points[:,2] / fx
            points[:,1] = (points[:,1] - cy) * points[:,2] / fy

            if self.use_gt_depth == False:
                points = points * self.scale_factor

            # estimated camera pose
            position = solution['t_est'][:, frame]
            orientation = solution["R_est"][3*frame:3*(frame+1), :]
            scale = solution["s_est"][frame]
            points = (orientation @ points.T * scale + position.reshape(3,1)).T

            # ground truth camera pose
            # position = self.t_gt[frame, :]
            # orientation = self.R_gt[3*(frame):3*(frame+1), :]
            # points = (orientation @ points.T + position.reshape(3,1)).T

            # save original point clouds for visualization in match_frame_pair_points function
            rgb_file = osp.join(rgb_dir, rgb_files[frame])
            rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            colors = np.reshape(rgb_image, (h*w, 3))
            colors = colors[~np.isnan(points).any(axis=1)] # remove points with NaN values
            points_plot = points[~np.isnan(points).any(axis=1)] # remove points with NaN values
            assert len(points_plot) == len(colors)
            colors = colors / 255.0

            pcd.points = open3d.utility.Vector3dVector(points_plot)
            pcd.colors = open3d.utility.Vector3dVector(colors)

            if frame == 0:
                # Initialization for the first frame
                visualizer.add_geometry(pcd)
                axes_scale = 0.15
                # if self.use_gt_depth == False:
                #     coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale, origin=[position[0], position[1], position[2]])
                #     visualizer.add_geometry(coordinate_frame)
                # else:
                #     coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale, origin=[position[0], position[1], position[2]])
                #     visualizer.add_geometry(coordinate_frame)
                # coordinate_frames.append(coordinate_frame)
                visualizer.get_render_option().point_size = 3
                ctr = visualizer.get_view_control()
            # else:
            #     # Remove the coordinate frame from the previous frame
            #     print(visualizer.get_geometry_list())
            #     visualizer.remove_geometry(visualizer.get_geometry_list()[-1])

            # Add the coordinate frame for the current frame

            # if frame > 0:
            #     visualizer.remove_geometry(coordinate_frames[frame-1])

            if self.use_gt_depth == False:
                # if frame > 0:
                #     visualizer.remove_geometry(coordinate_frame)
                coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale, origin=[position[0], position[1], position[2]])
                visualizer.add_geometry(coordinate_frame)
                # Set the camera pose to look from the camera origin towards the z-axis
                # xyz 1
                # ctr.set_lookat([position[0]+1, position[1]+0.3, position[2]+7])
                # xyz 2
                # ctr.set_lookat([position[0]-1, position[1], position[2]+106])
                ctr.set_lookat([position[0], position[1], position[2]+488])

            else:
                # if frame > 0:
                #     visualizer.remove_geometry(coordinate_frame)
                coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale, origin=[position[0], position[1], position[2]])
                visualizer.add_geometry(coordinate_frame)
                # Set the camera pose to look from the camera origin towards the z-axis
                # xyz 1
                # ctr.set_lookat([position[0]+1, position[1]+0.3, position[2]+3])
                # xyz 2
                ctr.set_lookat([position[0]-1, position[1], position[2]+106])

            coordinate_frames.append(coordinate_frame)

            # xyz 1
            # ctr.set_front([0, 0, -1])
            # xyz 2
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0]) # Up is positive Y, but since we want upside down, this is inverted
            visualizer.update_geometry(pcd)
            visualizer.poll_events()
            visualizer.update_renderer()
            visualizer.run()
        visualizer.destroy_window()


    def FullReconstruction(self, solution_path):
        with open(solution_path, 'rb') as file:
            solution = pickle.load(file)


        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()
        # init rgb and depth file
        depth_dir = self.pred_path + '/depth_dpt_beit_large_512/depth/'
        predict_files = sorted(os.listdir(depth_dir))
        rgb_dir = self.pred_path + '/color_full/'
        rgb_files = sorted(os.listdir(rgb_dir))
        depth_filelist, rgb_time_stamp, depth_time_stamp = self.get_gt_depth_file_list()

        pcd_list = []
        for frame in range(self.N):

            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[frame*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape
            # remove points out of range in prediction
            threshold = np.percentile(predicted_depth, self.percentile_threshold)
            predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)
            # read intrinsics
            fx, fy, cx, cy = self.get_intrinsics(frame)


            predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, frame)
            if self.use_gt_depth == True:
                # Load the gt depth map
                predicted_depth = predicted_depth_gt

            y, x = np.meshgrid(np.arange(self.target_height), np.arange(self.target_width), indexing="ij")
            points = np.stack((x, y, np.ones((self.target_height, self.target_width))), axis=2)
            points = np.reshape(points,(self.target_height*self.target_width,3))
            predicted_depth = np.reshape(predicted_depth,(self.target_height*self.target_width,1))
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (points[:,0] - cx) * points[:,2] / fx
            points[:,1] = (points[:,1] - cy) * points[:,2] / fy

            if self.use_gt_depth == False:
                points = points * self.scale_factor

            # estimated camera pose
            # position = solution['t_est'][:, frame]
            # orientation = solution["R_est"][3*frame:3*(frame+1), :]
            # scale = solution["s_est"][frame]
            # points = (orientation @ points.T * scale + position.reshape(3,1)).T


            # ground truth camera pose
            position = self.t_gt[frame, :]
            orientation = self.R_gt[3*(frame):3*(frame+1), :]
            points = (orientation @ points.T + position.reshape(3,1)).T


            # save original point clouds for visualization in match_frame_pair_points function
            rgb_file = osp.join(rgb_dir, rgb_files[frame])
            rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            colors = np.reshape(rgb_image, (h*w, 3))
            colors = colors[~np.isnan(points).any(axis=1)] # remove points with NaN values
            points_plot = points[~np.isnan(points).any(axis=1)] # remove points with NaN values
            assert len(points_plot) == len(colors)
            colors = colors / 255.0

            #################
            ### subsample ###
            #################
            # indices = np.arange(points_plot.shape[0])
            # np.random.shuffle(indices)
            # selected_indices = indices[:10000]
            # points_plot = points_plot[selected_indices]
            # colors = colors[selected_indices]


            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points_plot)
            pcd.colors = open3d.utility.Vector3dVector(colors)

            # if frame>0:
            #     # local refinement using ICP
            #     # open3d.visualization.draw_geometries([pcd,pcd_list[-1]])
            #     icp_sol = open3d.pipelines.registration.registration_icp(
            #         pcd, pcd_list[-1], 0.01, np.eye(4),
            #         open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            #         open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
            #     T_icp = icp_sol.transformation
            #     print(T_icp)
            #     pcd_icp = copy.deepcopy(pcd).transform(T_icp)
            #     pcd_list.append(pcd_icp)
            #     # open3d.visualization.draw_geometries([pcd_icp,pcd_list[-1]])
            # else:
            #     pcd_list.append(pcd)

            pcd_list.append(pcd)

        ###########################
        ########## icp reg ########
        ###########################
        for frame in range(self.N-1):
            for i in range(frame+1):
                pcd_A = copy.deepcopy(pcd_list[i])
                pcd_B = copy.deepcopy(pcd_list[frame+1])
                # open3d.visualization.draw_geometries([pcd_A,pcd_B])
                icp_sol = open3d.pipelines.registration.registration_icp(
                    pcd_A, pcd_B, 0.01, np.eye(4),
                    open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
                T_icp = icp_sol.transformation
                print(T_icp)
                pcd_A_icp = copy.deepcopy(pcd_A).transform(T_icp)
                pcd_list[i] = pcd_A_icp
                # open3d.visualization.draw_geometries([pcd_A_icp,pcd_B])

        pcd = pcd_list[0]
        for i in range(1, len(pcd_list)):
            pcd += pcd_list[i]
        visualizer.add_geometry(pcd)

        visualizer.get_render_option().point_size = 3
        ctr = visualizer.get_view_control()
        # if self.use_gt_depth == False:
        #     ctr.set_lookat([position[0]-2, position[1], position[2]])
        # else:
        #     ctr.set_lookat([position[0]-2, position[1], position[2]])

        # xyz 2
        if self.use_gt_depth == False:
            ctr.set_lookat([position[0], position[1], position[2]])
        else:
            ctr.set_lookat([position[0], position[1], position[2]])
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0]) # Up is positive Y, but since we want upside down, this is inverted

        visualizer.run()

        from matplotlib.backends.backend_pdf import PdfPages
        # Capture the screen image and save as a png
        image = visualizer.capture_screen_float_buffer(True)
        open3d.io.write_image("output.png", image)

        # Convert the image file to PDF using matplotlib
        img = plt.imread('output.png')
        plt.imshow(img)
        pdf_pages = PdfPages('output.pdf')
        pdf_pages.savefig(bbox_inches='tight')
        pdf_pages.close()

        visualizer.destroy_window()

    def printErr(self, solution):

        norm_list_est = np.linalg.norm(solution['t_est'].T, axis=1)
        norm_list_gt = np.linalg.norm(self.t_gt, axis=1)
        scale_list = norm_list_gt / norm_list_est

        scale = np.median(scale_list[~np.isnan(scale_list)])

        solution['t_est'] = solution['t_est'] * scale

        stats = defaultdict()

        R_err = 0
        for i in range(self.N):
            R_err += getAngularError( self.R_gt[3*i:3*(i+1), :], solution['R_est'][3*i:3*(i+1), :])

        avg_R_err = R_err / self.N

        t_err = 0
        for i in range(self.N):
            t_err += getTranslationError( self.t_gt[i, :], solution['t_est'][:, i])
        avg_t_err = t_err / self.N

        print(f"avg_R_err = {avg_R_err}[deg], avg_t_err = {avg_t_err}.")
        stats['avg_R_err'] = avg_R_err
        stats['avg_t_err'] = avg_t_err

        poses_ground_truth = []
        poses = []
        for i in range(self.N):
            one_pose = np.zeros((4,4))
            one_pose[0:3,0:3] = self.R_gt[3*i:3*(i+1), :]
            one_pose[0:3,3] = self.t_gt[i,:]
            one_pose[3,3] = 1
            poses_ground_truth.append(one_pose)

            one_pose = np.zeros((4,4))
            one_pose[0:3,0:3] = solution['R_est'][3*i:3*(i+1), :]
            one_pose[0:3,3] = solution['t_est'][:, i]
            one_pose[3,3] = 1
            poses.append(one_pose)

        def compute_rpe(gt_poses, pred_poses, delta):
            "Compute relative pose error (RPE) in terms of translation and rotation"
            # Compute the number of poses and the number of steps between them
            n_poses = len(gt_poses)
            n_steps = n_poses - delta
            # Initialize arrays for storing the errors
            trans_errors = np.zeros(n_steps)
            rot_errors = np.zeros(n_steps)
            # Compute the errors for each step
            for i in range(n_steps):
                # Extract the corresponding poses from the sequences
                gt_pose1 = gt_poses[i]
                gt_pose2 = gt_poses[i+delta]
                pred_pose1 = pred_poses[i]
                pred_pose2 = pred_poses[i+delta]
                # Compute the relative pose between the poses in the ground truth sequence
                rel_gt_pose = np.linalg.inv(gt_pose1) @ gt_pose2
                # Compute the relative pose between the corresponding predicted poses
                rel_pred_pose = np.linalg.inv(pred_pose1) @ pred_pose2
                # Compute the translation error
                trans_error = np.linalg.norm(rel_pred_pose[:3, 3] - rel_gt_pose[:3, 3])
                trans_errors[i] = trans_error
                # Compute the rotation error
                rel_gt_rot = R.from_matrix(rel_gt_pose[:3, :3])
                rel_pred_rot = R.from_matrix(rel_pred_pose[:3, :3])
                # Compute the relative rotation between the poses
                rel_gt_rot_obj = rel_gt_rot.inv() * rel_pred_rot
                # Compute the angle of the relative rotation
                angle = rel_gt_rot_obj.magnitude()
                # Convert the angle to degrees
                rot_errors[i] = np.rad2deg(angle)
            # Compute the mean errors
            mean_trans_error = np.mean(trans_errors[1:])
            mean_rot_error = np.mean(rot_errors[1:])
            return mean_trans_error, mean_rot_error


        mean_trans_error, mean_rot_error = compute_rpe(poses_ground_truth, poses, 1)
        print("RPE-T: {:.4f}".format(mean_trans_error))
        print("RPE-R: {:.4f}".format(mean_rot_error))

        stats['RPE-T'] = mean_trans_error
        stats['RPE-R'] = mean_rot_error

        return stats

    def get_EssentialMap(self):
        ##################################################
        ################## pose extractor ################
        ##################################################

        _, rgb_timestamp, _ = self.get_gt_depth_file_list()

        # Read the text file
        with open(self.KeyFrame_file, "r") as file:
            lines = file.readlines()

        translations = []
        timestamps = []
        rotations_quat = []

        for line in lines:
            parts = line.split()
            timestamp = float(parts[0])
            translation = np.array([float(parts[i]) for i in range(1, 4)])
            quaternion = np.array([float(parts[i]) for i in range(4, 8)])  # Assuming x, y, z, w order
            # rotation = R.from_quat(quaternion).as_matrix()  # Convert quaternion to rotation matrix

            timestamps.append(timestamp)
            translations.append(translation)
            rotations_quat.append(quaternion)

        timestamps = np.array(timestamps)
        translations = np.array(translations)
        rotations_quat = np.array(rotations_quat)

        print("Timestamps:", timestamps)
        print("Translations:", translations)
        print("Rotations:", rotations_quat)

        EssentialMap = []
        timestamp_EssentialMap = []
        for i in range(len(rgb_timestamp)):
            if rgb_timestamp[i] in timestamps:
                EssentialMap.append(i)
                timestamp_EssentialMap.append(rgb_timestamp[i])

        if len(rgb_timestamp)-1 not in EssentialMap:
            EssentialMap.append(len(rgb_timestamp)-1)
            timestamp_EssentialMap.append(rgb_timestamp[-1])
        if 0 not in EssentialMap:
            EssentialMap.append(0)
            timestamp_EssentialMap.append(rgb_timestamp[0])

        # for i in range(0, len(rgb_timestamp), 40):
        #     if i not in EssentialMap:
        #         EssentialMap.append(i)
        #         timestamp_EssentialMap.append(rgb_timestamp[i])

        EssentialMap = sorted(EssentialMap)
        timestamp_EssentialMap = sorted(timestamp_EssentialMap)

        return EssentialMap, rgb_timestamp, timestamp_EssentialMap

if __name__ == "__main__":

    avg_R_err = []
    avg_t_err = []
    RPE_T = []
    RPE_R = []
    subopt = []
    Rounds = 1

    for round in range(Rounds):
        pose_optimizer = PoseOptimizerTUM()
        # image_pair_correspondence: (frame1, frame2) -> (h,w,2)

        EssentialMap, desired_timestamp, timestamps = pose_optimizer.get_EssentialMap()


        print("Get flow constraints:")
        image_pair_correspondence = pose_optimizer.get_matching_features_CAPS_EssentialMap(EssentialMap)
        print("Get depth maps:")
        scaled_cloud_camera_frame = pose_optimizer.get_depth_map_CAPS(image_pair_correspondence)
        print("Match points in frame pairs:")
        pose_optimizer.match_frame_pair_points(scaled_cloud_camera_frame)

        # N = pose_optimizer.N

        N = len(EssentialMap)
        edges = list(pose_optimizer.scaled_cloud_camera_frame_dict.keys())
        for j in range(len(edges)):
            for i in range(len(EssentialMap)):
                if edges[j][0] == EssentialMap[i]:
                    edges[j] = (i,edges[j][1])
                if edges[j][1] == EssentialMap[i]:
                    edges[j] = (edges[j][0],i)
        pointclouds = list(pose_optimizer.scaled_cloud_camera_frame_dict.values())
        pointclouds = []
        for pointclouds_pair in list(pose_optimizer.scaled_cloud_camera_frame_dict.values()):
            pointclouds_pair_list = list(pointclouds_pair.values())
            Pi = pointclouds_pair_list[0].T
            Pj = pointclouds_pair_list[1].T
            combined_array = np.vstack((Pi, Pj))
            pointclouds.append(combined_array)

        print("########################################")
        print("########## TEASER + SimSync ############")
        print("########################################")

        start = time.time()
        scale_gt = np.ones((N, 1))
        solution, weights = TEASER_SimSync(N, edges, pointclouds, reg_lambda=0)
        end = time.time()


        ################################
        ######## GTSAM preproc #########
        ################################

        edges_original = list(pose_optimizer.scaled_cloud_camera_frame_dict.keys())
        edges_corresponding = list(pose_optimizer.scaled_cloud_camera_frame_dict.keys())
        for j in range(len(edges_corresponding)):
            for i in range(len(EssentialMap)):
                if edges_corresponding[j][0] == EssentialMap[i]:
                    edges_corresponding[j] = (i,edges_corresponding[j][1])
                if edges_corresponding[j][1] == EssentialMap[i]:
                    edges_corresponding[j] = (edges_corresponding[j][0],i)

        camera_pair_mapping = defaultdict()
        for j in range(len(edges_corresponding)):
            camera_pair_mapping[edges_corresponding[j]] = edges_original[j]


        correspondences_points = defaultdict()


        index_point_counter = 0
        for i in range(N):
            correspondences_points['camera{}'.format(i)] = defaultdict()
            for j in range(i, N):
                correspondences_points['camera{}'.format(i)]['camera{}'.format(j)] = defaultdict()
                if (i,j) in camera_pair_mapping:
                    camera_pair = camera_pair_mapping[(i,j)]
                    corresponding_points = image_pair_correspondence[camera_pair]
                    correspondences_points['camera{}'.format(i)]['camera{}'.format(j)]['point_index'] = np.zeros((corresponding_points.shape[0], ))
                    for k in range(corresponding_points.shape[0]):
                        correspondences_points['camera{}'.format(i)]['camera{}'.format(j)]['point_index'][k] = index_point_counter
                        index_point_counter += 1
                    correspondences_points['camera{}'.format(i)]['camera{}'.format(j)]['pixel_coord_camera{}'.format(i)] = corresponding_points[:,0:2]

        for i in range(N):
            for j in range(0, i):
                correspondences_points['camera{}'.format(i)]['camera{}'.format(j)] = defaultdict()
                if (j, i) in camera_pair_mapping:
                    camera_pair = camera_pair_mapping[(j,i)]
                    corresponding_points = image_pair_correspondence[camera_pair]
                    correspondences_points['camera{}'.format(i)]['camera{}'.format(j)]['point_index'] = np.zeros((corresponding_points.shape[0], ))
                    for k in range(corresponding_points.shape[0]):
                        correspondences_points['camera{}'.format(i)]['camera{}'.format(j)]['point_index'][k] = correspondences_points['camera{}'.format(j)]['camera{}'.format(i)]['point_index'][k]
                    correspondences_points['camera{}'.format(i)]['camera{}'.format(j)]['pixel_coord_camera{}'.format(i)] = corresponding_points[:,2:4]

        solution['num_points'] = index_point_counter
        if pose_optimizer.sequence_name == 'freiburg2_xyz':
            with open("correspondences_points_GTSAM_xyz2.pickle", "wb") as file:
                pickle.dump(correspondences_points, file)
            with open('solution_GTSAM_xyz2.pickle', 'wb') as file:
                pickle.dump(solution, file)
        elif pose_optimizer.sequence_name == 'freiburg1_xyz':
            with open("correspondences_points_GTSAM_xyz1.pickle", "wb") as file:
                pickle.dump(correspondences_points, file)
            with open('solution_GTSAM_xyz1.pickle', 'wb') as file:
                pickle.dump(solution, file)


        # with open("solution_GTSAMresults.pickle", "rb") as file:
        #     solution = pickle.load(file)

        ################################
        ##### interpolate solution #####
        ################################

        desired_times = desired_timestamp
        translations = []
        rotations_quat = []
        scales = []
        for i in range(N):
            rotation = R.from_matrix(solution['R_est'][3*i:3*(i+1), :])
            quaternion = rotation.as_quat()
            translations.append(solution['t_est'][:, i])
            rotations_quat.append(quaternion)
            scales.append(solution['s_est'][i])

        translations = np.array(translations)
        rotations_quat = np.array(rotations_quat)
        scales = np.array(scales)
        desired_times = np.array(desired_times)

        print("Timestamps:", timestamps)
        print("Translations:", translations)
        print("Rotations:", rotations_quat)
        print("Scales:", scales)
        # interpolate rotation
        rotations = R.from_quat(rotations_quat)
        slerp = Slerp(timestamps, rotations)
        interpolated_rotation = slerp(desired_times)
        # interpolate translation
        interpolate_translation = interp1d(timestamps, translations, axis=0, kind='linear')
        interpolated_translation = interpolate_translation(desired_times)
        interpolate_scales = interp1d(timestamps, scales, axis=0, kind='linear')
        interpolated_scales = interpolate_scales(desired_times)
        solution['t_est'] = interpolated_translation.T
        reshaped_rotation = interpolated_rotation.as_matrix()
        solution['R_est'] = reshaped_rotation.reshape(desired_times.shape[0]*3,3)
        solution['s_est'] = interpolated_scales


        print("Get ground truth trajectory:")
        pose_optimizer.get_gt_traj()
        # Save the defaultdict to a file
        with open('solution.pkl', 'wb') as file:
            pickle.dump(solution, file)
        print("Visualize camera trajectory:")
        # pose_optimizer.visCameraTraj(solution_path = 'solution.pkl')
        stats = pose_optimizer.printErr(solution)
        # Save the defaultdict to a file
        with open('solution.pkl', 'wb') as file:
            pickle.dump(solution, file)
        print("Visualize camera trajectory:")
        # pose_optimizer.visCameraTraj(solution_path = 'solution.pkl')

        avg_R_err.append(stats['avg_R_err'])
        avg_t_err.append(stats['avg_t_err'])
        RPE_T.append(stats['RPE-T'])
        RPE_R.append(stats['RPE-R'])
        subopt.append(solution['relDualityGap'])

        pose_optimizer.Reconstruction(solution_path = 'solution.pkl') # video
        # pose_optimizer.visFilteredPointclouds(weights)

        # pose_optimizer.FullReconstruction(solution_path = 'solution.pkl') # full reconstruct

        # position = solution['t_est'][:, 0]
        # orientation = solution["R_est"][3*0:3*(0+1), :]
        # scale = solution["s_est"][0]

        # # Create an array
        # y = pose_optimizer.depth_min

        # # Create corresponding x values
        # x = np.arange(len(y))

        # # Plot the array
        # plt.plot(x, y)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Array Plot')
        # plt.grid(True)
        # plt.show()


        # # Create an array
        # y = pose_optimizer.depth_max

        # # Create corresponding x values
        # x = np.arange(len(y))

        # # Plot the array
        # plt.plot(x, y)
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('Array Plot')
        # plt.grid(True)
        # plt.show()

    avg_R_err = np.array(avg_R_err)
    avg_t_err = np.array(avg_t_err)
    RPE_T = np.array(RPE_T)
    RPE_R = np.array(RPE_R)
    subopt = np.array(subopt)


    mean_R_err = np.mean(avg_R_err)
    mean_t_err = np.mean(avg_t_err)
    mean_RPE_T = np.mean(RPE_T)
    mean_RPE_R = np.mean(RPE_R)
    mean_subopt = np.mean(subopt)

    tmp = 1