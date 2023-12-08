# Xihang Yu
# 07/26/2023

from os.path import join as pjoin
import os
import cv2
import numpy as np
import os.path as osp
import open3d
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle
import copy
plt.rcParams['backend'] = 'TkAgg'

from utils import image_io

class PoseOptimizer():
    def __init__(self, gt_path, pred_path):
        """
        pred_path: path to predicted optical flow and depth priors
        gt_path: ground truth of camera calibrations
        vis_correspondence_mode: binary variable to control correspondence visualization
        vis_depth_mode: binary variable to control depth visualization
        vis_final_matching_mode: binary variable to control final point match visualization
        sequence_name: a short name for the sequence
        percentile_threshold: filtering threshold for depth prior to filter out erroneous depth prediction (Not used for now)
        num_pairs_keep: number of corresponding points in each image pair
        start_frame_optimizer: start frame in a sequence to optimize
        end_frame_optimizer: end frame in a sequence to optimize
        distance_threshold: filtering threshold for large distant correpsondences
        point_scaling_factor: scale the point position (camera frame) to make SDP optimization stable (Not used for now)
        lower_scale_bound: lower scale bound for SDP
        upper_scale_bound: upper scale bound for SDP
        tighterInequalityConstraint: binary variable to specify sqrt(3)*s <= |Rs| <= sqrt(3)*s
        tightestInequalityConstraint: binary variable to specify s <= |c_i*s| <= s
        target_width,target_height: size of alignedplt.rcParams['backend'] = 'TkAgg'se ground truth depth (Modify in subclass)
        pointCloudPairSeparateDistance: distance to separate point clouds between two frames. Just for visualization
        start_frame_optimizer: start frame for optimization
        end_frame_optimizer: end frame for optimization
        whole_sequence_len: total frames in a whole sequence. end_frame_optimizer-start_frame_optimizer <= whole_sequence_len
        """

        self.pred_path = pred_path
        self.gt_path = gt_path
        self.vis_correspondence_mode = False
        self.vis_depth_mode = False
        self.vis_final_matching_mode = False
        
        self.percentile_threshold = 90
        # self.num_pairs_keep = 400 # best performance
        self.num_pairs_keep = 400
        self.distance_threshold = 100  # percentage

        self.scale_factor = None

        # scale_bound
        self.lower_scale_bound = 0.9
        self.upper_scale_bound = 1.1
        self.tighterInequalityConstraint = False
        self.tightestInequalityConstraint = False

    def subsample_point_clouds(self, point_cloud_1, point_cloud_2, num_pairs):
        # Generate random indices
        indices = np.arange(point_cloud_1.shape[0])
        np.random.shuffle(indices)
        
        # Select the first num_pairs pairs
        selected_indices = indices[:num_pairs]
        
        # Extract the corresponding points
        selected_points_1 = point_cloud_1[selected_indices]
        selected_points_2 = point_cloud_2[selected_indices]
        return selected_points_1, selected_points_2

    def remove_nan_pairs(self,point_cloud_1, point_cloud_2):
        # Find indices of NaN values in either point cloud
        nan_indices_1 = np.isnan(point_cloud_1).any(axis=1)
        nan_indices_2 = np.isnan(point_cloud_2).any(axis=1)
        
        # Find indices of pairs to be removed
        remove_indices = np.logical_or(nan_indices_1, nan_indices_2)
        
        # Remove pairs from both point clouds
        filtered_point_cloud_1 = point_cloud_1[~remove_indices]
        filtered_point_cloud_2 = point_cloud_2[~remove_indices]

        return filtered_point_cloud_1, filtered_point_cloud_2

    def get_indices(self, name):
        strs = os.path.splitext(name)[0].split("_")[1:]
        return [int(s) for s in strs]

    def get_flow_constraints(self):
        
        # init flow constraints
        flow_fmt = pjoin(self.pred_path, "flow", "flow_{:06d}_{:06d}.raw")
        flow_names = sorted(os.listdir(os.path.dirname(flow_fmt)))
        image_pair_correspondence = {}
        for flow_name in flow_names:
            indices = self.get_indices(flow_name)
            indices_pair = [indices, indices[::-1]]
            if tuple(indices_pair[0]) in image_pair_correspondence \
                 or tuple(indices_pair[1]) in image_pair_correspondence:
                continue
            flow_fns = [flow_fmt.format(*idxs) for idxs in indices_pair]
            flows = [image_io.load_raw_float32_image(fn) for fn in flow_fns]

            # Compute pixel-wise correspondences
            # Create a 2D matrix of pixel positions
            x_coords = np.arange(self.target_width)
            y_coords = np.arange(self.target_height)
            x_matrix, y_matrix = np.meshgrid(x_coords, y_coords)
            pixel_matrix_image1 = np.stack((y_matrix, x_matrix), axis=2)
            mapped_pixel_matrix_image1 = pixel_matrix_image1 + flows[0]

            # mask out the corresponding pixel that has distance less than delta pixels
            # h = np.arange(self.target_height)[:, np.newaxis]
            # w = np.arange(self.target_width)
            # distance = np.sqrt((mapped_pixel_matrix_image1[h, w, 0] - h) ** 2 + (mapped_pixel_matrix_image1[h, w, 1] - w) ** 2)
            # mask_delta = distance < 0
            # mapped_pixel_matrix_image1[mask_delta, 0] = -1
            # mapped_pixel_matrix_image1[mask_delta, 1] = -1

            diff_pixels_xy = abs(flows[0] + flows[1]) # width, height, 2
            mask_match_y = diff_pixels_xy[:,:,0] < 0.1
            mask_match_x = diff_pixels_xy[:,:,1] < 0.1
            mask_match_combined = np.logical_and(mask_match_y, mask_match_x)
            # only keep those pixels that are in range of size of image2
            mask_lower_bound_y = mapped_pixel_matrix_image1[:,:,0] >= 0
            mask_upper_bound_y = mapped_pixel_matrix_image1[:,:,0] < self.target_height
            mask_lower_bound_x = mapped_pixel_matrix_image1[:,:,1] >= 0
            mask_upper_bound_x = mapped_pixel_matrix_image1[:,:,1] < self.target_width
            mask_range_y = np.logical_and(mask_lower_bound_y, mask_upper_bound_y)
            mask_range_x = np.logical_and(mask_lower_bound_x, mask_upper_bound_x)
            mask_range = np.logical_and(mask_range_y,mask_range_x)
            # mask out the dynamic objects
            mask_fmt = pjoin(self.pred_path, "dynamic_mask", "frame_{:06d}.png")
            dynamic_mask_file = mask_fmt.format(indices[0])
            mask_image_dynamic_objects = cv2.imread(dynamic_mask_file, cv2.IMREAD_GRAYSCALE)
            mask_dynamic_objects = mask_image_dynamic_objects == 255
            mapped_pixel_matrix_image1[mask_match_combined==False] = np.array((-1,-1))
            mapped_pixel_matrix_image1[mask_range==False] = np.array((-1,-1))
            mapped_pixel_matrix_image1[mask_dynamic_objects==False] = np.array((-1,-1))

            # (y,x) entry in filtered matrix is (y, x) entry in image 1 
            # It has value (map(y), map(x)) which is the entry in image 2
            filtered_mapped_pixel_matrix_image1 = mapped_pixel_matrix_image1 
            image_pair_correspondence[tuple(indices)] = filtered_mapped_pixel_matrix_image1

        if self.use_gt_flow == True:
            image_pair_correspondence = self.get_gt_flow_constraints()

        if self.vis_correspondence_mode == True:
            rgb_dir = self.pred_path + '/color_full/'
            rgb_files = sorted(os.listdir(rgb_dir))            
            for key, value in image_pair_correspondence.items():
                frame1 = key[0]
                frame2 = key[1]

                print("Visualize frame {} and frame {} correspondances.".format(frame1, frame2))
                
                # Create original images
                rgb_file_frame1 = osp.join(rgb_dir, rgb_files[frame1])
                rgb_file_frame2 = osp.join(rgb_dir, rgb_files[frame2])
                rgb_image_frame1 = cv2.imread(rgb_file_frame1, cv2.IMREAD_COLOR)
                rgb_image_frame2 = cv2.imread(rgb_file_frame2, cv2.IMREAD_COLOR)
                ## reshape RGB image to a 1D array of RGB values
                colors_frame1 = np.reshape(rgb_image_frame1, (self.target_height*self.target_width, 3)) 
                colors_frame2 = np.reshape(rgb_image_frame2, (self.target_height*self.target_width, 3))
                colors_frame1 = colors_frame1 / 255.0
                colors_frame2 = colors_frame2 / 255.0
                y, x = np.meshgrid(np.arange(self.target_height), np.arange(self.target_width), indexing="ij")
                points_frame1 = np.stack((x, y, np.zeros((self.target_height, self.target_width))), axis=2)
                points_frame2 = np.stack((x, y, np.zeros((self.target_height, self.target_width))), axis=2)

                # Extract the corresponding points from points_frame1 and points_frame2
                valid_indices = np.where((value[:,:,0]!=-1)&(value[:,:,1]!=-1))
                y_indices = valid_indices[0]
                x_indices = valid_indices[1]
                y_frame2 = value[y_indices, x_indices, 0]
                x_frame2 = value[y_indices, x_indices, 1]
                corresponding_points_frame1 = points_frame1[y_indices, x_indices]
                corresponding_points_frame2 = points_frame2[y_frame2, x_frame2]

                # subsample points
                corresponding_points_frame1, corresponding_points_frame2 = self.subsample_point_clouds(corresponding_points_frame1, \
                                                                                corresponding_points_frame2, self.num_pairs_keep)

                # Separate frame 1 and frame 2, image and correspondings
                points_frame1 = np.reshape(points_frame1,(self.target_height*self.target_width,3))
                points_frame1[:,2] = points_frame1[:,2] + 500
                corresponding_points_frame1[:,2] = corresponding_points_frame1[:,2] + 500
                corresponding_points_frame1[:,2] = corresponding_points_frame1[:,2] - 1
                corresponding_points_frame2[:,2] = corresponding_points_frame2[:,2] - 1
                points_frame2 = np.reshape(points_frame2,(self.target_height*self.target_width,3))

                # Create image object
                original_img1_o3d = open3d.geometry.PointCloud()
                original_img2_o3d = open3d.geometry.PointCloud()
                original_img1_o3d.points = open3d.utility.Vector3dVector(points_frame1)
                original_img1_o3d.colors = open3d.utility.Vector3dVector(colors_frame1)
                original_img2_o3d.points = open3d.utility.Vector3dVector(points_frame2)
                original_img2_o3d.colors = open3d.utility.Vector3dVector(colors_frame2)

                # Create LineSet object
                num_corresponding_points = len(corresponding_points_frame1)
                assert len(corresponding_points_frame1) == len(corresponding_points_frame2)
                colors = np.zeros((num_corresponding_points, 3))
                colors[:, 0] = np.linspace(0, 1, num_corresponding_points)  # Red channel
                colors_o3d = open3d.utility.Vector3dVector(colors)
                line_set = open3d.geometry.LineSet()
                line_set.points = open3d.utility.Vector3dVector(np.concatenate((corresponding_points_frame1, \
                                                                                    corresponding_points_frame2), axis=0))
                line_set.lines = open3d.utility.Vector2iVector(np.array([[i, i + len(corresponding_points_frame1)] \
                                                                            for i in range(len(corresponding_points_frame1))]))
                line_set.colors = colors_o3d
                pcd1_o3d = open3d.geometry.PointCloud()
                pcd1_o3d.points = open3d.utility.Vector3dVector(corresponding_points_frame1)
                pcd1_o3d.colors = colors_o3d
                pcd2_o3d = open3d.geometry.PointCloud()
                pcd2_o3d.points = open3d.utility.Vector3dVector(corresponding_points_frame2)
                pcd2_o3d.colors = colors_o3d

                # Visualize the point clouds
                visualizer = open3d.visualization.Visualizer()
                visualizer.create_window()

                # Add the point clouds to the visualizer
                visualizer.add_geometry(pcd1_o3d)
                visualizer.add_geometry(pcd2_o3d)
                visualizer.add_geometry(original_img1_o3d)
                visualizer.add_geometry(original_img2_o3d)
                visualizer.add_geometry(line_set)
                axes_scale = 100
                visualizer.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale, origin=[0, 0, 0]))
                visualizer.get_render_option().point_size = 3

                # Run the visualizer
                visualizer.run()
                visualizer.destroy_window()       

        return image_pair_correspondence


    def get_matching_features_CAPS(self):
        flow_fmt = pjoin(self.pred_path, "flow", "flow_{:06d}_{:06d}.raw")
        flow_names = sorted(os.listdir(os.path.dirname(flow_fmt)))
        image_pair_correspondence = {}
        for flow_name in flow_names:
            indices = np.array(self.get_indices(flow_name))
            if (indices[1], indices[0]) in image_pair_correspondence:
                continue            
            image_pair_correspondence[(indices[0], indices[1])] = []

            img1 = self.read_imgs(indices[0])
            l,h = img1.shape[0],img1.shape[1]
            img2 = self.read_imgs(indices[1])

            coords_1, descriptor1 = self.read_feat(indices[0])
            coords_2, descriptor2 = self.read_feat(indices[1])

            bf = cv2.BFMatcher()
            matches=bf.match(descriptor1,descriptor2)
            matches = sorted(matches, key = lambda x:x.distance)


            for i in range(self.num_pairs_keep):
                image_pair_correspondence[(indices[0], indices[1])].append(np.concatenate((coords_1[matches[i].queryIdx], coords_2[matches[i].trainIdx])))

            image_pair_correspondence[(indices[0], indices[1])] = np.array(image_pair_correspondence[(indices[0], indices[1])])

            tmp = 1
            # visualize
            # sift = cv2.SIFT_create()
            # coords_11, _ = sift.detectAndCompute(img1,None)
            # coords_22, _ = sift.detectAndCompute(img2,None)
            # matching_result = cv2.drawMatches(img1, coords_11, img2, coords_22, matches[:100], None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)
            # cv2.imshow('Matching Result', matching_result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        return image_pair_correspondence

    def get_matching_features_CAPS_EssentialMap(self, EssentialMap):
        from itertools import combinations
        KeyFramesPairs = list(combinations(EssentialMap, 2))

        image_pair_correspondence = {}
        for indices in KeyFramesPairs:
            if (indices[1], indices[0]) in image_pair_correspondence:
                continue            
            image_pair_correspondence[(indices[0], indices[1])] = []

            img1 = self.read_imgs(indices[0])
            l,h = img1.shape[0],img1.shape[1]
            img2 = self.read_imgs(indices[1])

            coords_1, descriptor1 = self.read_feat(indices[0])
            coords_2, descriptor2 = self.read_feat(indices[1])

            bf = cv2.BFMatcher()
            matches=bf.match(descriptor1,descriptor2)
            matches = sorted(matches, key = lambda x:x.distance)


            for i in range(self.num_pairs_keep):
                image_pair_correspondence[(indices[0], indices[1])].append(np.concatenate((coords_1[matches[i].queryIdx], coords_2[matches[i].trainIdx])))

            image_pair_correspondence[(indices[0], indices[1])] = np.array(image_pair_correspondence[(indices[0], indices[1])])

            tmp = 1
            # visualize
            # sift = cv2.SIFT_create()
            # coords_11, _ = sift.detectAndCompute(img1,None)
            # coords_22, _ = sift.detectAndCompute(img2,None)
            # matching_result = cv2.drawMatches(img1, coords_11, img2, coords_22, matches[:100], None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)
            # cv2.imshow('Matching Result', matching_result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
        return image_pair_correspondence

    def get_depth_map_CAPS(self, image_pair_correspondence):

        # for sintel
        if self.dataset_type == 'Sintel':
            depth_gt, gt_depth_file_list = self.get_gt_depth_file_list()
        
        if self.dataset_type == 'TUM':
            depth_filelist, rgb_time_stamp, depth_time_stamp = self.get_gt_depth_file_list()

        # init rgb and depth file
        depth_dir = self.pred_path + '/depth_dpt_beit_large_512/depth/'
        predict_files = sorted(os.listdir(depth_dir))
        rgb_dir = self.pred_path + '/color_full/'
        rgb_files = sorted(os.listdir(rgb_dir))
        number_files = int(len(predict_files)/2)
        self.N = number_files

        scaled_cloud_camera_frame = defaultdict()

        for key, value in image_pair_correspondence.items():

            scaled_cloud_camera_frame[key] = defaultdict()
            
            frame1, frame2 = key
            # init indices
            # valid_indices = np.where((value[:,:,0]!=-1)&(value[:,:,1]!=-1))
            # y_indices = valid_indices[0]
            # x_indices = valid_indices[1]
            # y_frame2 = value[y_indices, x_indices, 0]
            # x_frame2 = value[y_indices, x_indices, 1]


            x_frame1 = value[:, 0]
            y_frame1 = value[:, 1]

            x_frame2 = value[:, 2]
            y_frame2 = value[:, 3]

            #################################
            ##### frame1 correspondences ####
            #################################
            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[frame1*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape
            # remove points out of range in prediction
            threshold = np.percentile(predicted_depth, self.percentile_threshold)
            predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)
            # sintel
            if self.dataset_type == 'Sintel':
                predicted_depth_gt = self.get_gt_depth(depth_gt, gt_depth_file_list, frame1)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
            if self.dataset_type == 'TUM':
                predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, frame1)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean

            # read intrinsics
            fx_frame1, fy_frame1, cx_frame1, cy_frame1 = self.get_intrinsics(frame1)

            # construct points
            # points = np.zeros((y_frame2.shape[0], 3))
            # predicted_depth = predicted_depth[y_indices, x_indices]            
            # points[:,2] = predicted_depth.reshape(-1,)
            # points[:,0] = (x_indices - cx_frame1) * predicted_depth / fx_frame1
            # points[:,1] = (y_indices - cy_frame1) * predicted_depth / fy_frame1
            # scaled_cloud_camera_frame[key][frame1] = points



            points = np.zeros((y_frame1.shape[0], 3))
            y_frame1_ceil = np.ceil(y_frame1).astype(np.int32)
            y_frame1_floor = np.floor(y_frame1).astype(np.int32)
            x_frame1_ceil = np.ceil(x_frame1).astype(np.int32)
            x_frame1_floor = np.floor(x_frame1).astype(np.int32)
            y_frame1_ceil[y_frame1_ceil == self.target_height] = self.target_height - 1
            x_frame1_ceil[x_frame1_ceil == self.target_width] = self.target_width - 1
            depth1 = predicted_depth[y_frame1_ceil, x_frame1_ceil]
            depth2 = predicted_depth[y_frame1_ceil, x_frame1_floor]
            depth3 = predicted_depth[y_frame1_floor, x_frame1_ceil]
            depth4 = predicted_depth[y_frame1_floor, x_frame1_floor]
            predicted_depth = (depth1+depth2+depth3+depth4)/4
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (x_frame1 - cx_frame1) * predicted_depth / fx_frame1
            points[:,1] = (y_frame1 - cy_frame1) * predicted_depth / fy_frame1

            scaled_cloud_camera_frame[key][frame1] = points

            #################################
            ##### frame2 correspondences ####
            #################################
            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[frame2*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape
            # remove points out of range in prediction
            threshold = np.percentile(predicted_depth, self.percentile_threshold)
            predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)

            if self.dataset_type == 'Sintel':
                predicted_depth_gt = self.get_gt_depth(depth_gt, gt_depth_file_list, frame2)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
            if self.dataset_type == 'TUM':
                predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, frame1)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
            fx_frame2, fy_frame2, cx_frame2, cy_frame2 = self.get_intrinsics(frame2)            

            # construct points
            points = np.zeros((y_frame2.shape[0], 3))
            y_frame2_ceil = np.ceil(y_frame2).astype(np.int32)
            y_frame2_floor = np.floor(y_frame2).astype(np.int32)
            x_frame2_ceil = np.ceil(x_frame2).astype(np.int32)
            x_frame2_floor = np.floor(x_frame2).astype(np.int32)
            y_frame2_ceil[y_frame2_ceil == self.target_height] = self.target_height - 1
            x_frame2_ceil[x_frame2_ceil == self.target_width] = self.target_width - 1
            depth1 = predicted_depth[y_frame2_ceil, x_frame2_ceil]
            depth2 = predicted_depth[y_frame2_ceil, x_frame2_floor]
            depth3 = predicted_depth[y_frame2_floor, x_frame2_ceil]
            depth4 = predicted_depth[y_frame2_floor, x_frame2_floor]
            predicted_depth = (depth1+depth2+depth3+depth4)/4
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (x_frame2 - cx_frame2) * predicted_depth / fx_frame2
            points[:,1] = (y_frame2 - cy_frame2) * predicted_depth / fy_frame2

            scaled_cloud_camera_frame[key][frame2] = points


        ###########################
        #### For Visualization ####
        ###########################

        if self.dataset_type == 'Sintel':
            depth_gt, gt_depth_file_list = self.get_gt_depth_file_list()

        if self.dataset_type == 'TUM':
            depth_filelist, rgb_time_stamp, depth_time_stamp = self.get_gt_depth_file_list()

        # init rgb and depth file
        depth_dir = self.pred_path + '/depth_dpt_beit_large_512/depth/'
        predict_files = sorted(os.listdir(depth_dir))
        rgb_dir = self.pred_path + '/color_full/'
        rgb_files = sorted(os.listdir(rgb_dir))
        number_files = int(len(predict_files)/2)
        self.original_point_cloud = {}
        self.original_point_cloud_rgb = {}
        self.s_gt = []

        for i in range(number_files):

            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[i*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape

            # remove points out of range in prediction
            threshold = np.percentile(predicted_depth, self.percentile_threshold)
            predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)

            if self.dataset_type == 'Sintel':
                predicted_depth_gt = self.get_gt_depth(depth_gt, gt_depth_file_list, i)
                
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt                
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]

                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)

                    self.scale_factor = gt_depth_mean / pred_depth_mean
            if self.dataset_type == 'TUM':
                predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, i)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
                    tmp = 1

            # read intrinsics
            fx, fy, cx, cy = self.get_intrinsics(i)

            # construct points
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            points = np.stack((x, y, np.ones((h, w))), axis=2)
            points = np.reshape(points,(h*w,3))
            predicted_depth = np.reshape(predicted_depth,(h*w,1))
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (points[:,0] - cx) * points[:,2] / fx
            points[:,1] = (points[:,1] - cy) * points[:,2] / fy

            # save original point clouds for visualization in match_frame_pair_points function
            rgb_file = osp.join(rgb_dir, rgb_files[i])
            rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            colors = np.reshape(rgb_image, (h*w, 3))
            colors = colors[~np.isnan(points).any(axis=1)] # remove points with NaN values
            points_plot = points[~np.isnan(points).any(axis=1)] # remove points with NaN values
            assert len(points_plot) == len(colors)
            colors = colors / 255.0
            self.original_point_cloud["frame{}".format(i)] = points_plot
            self.original_point_cloud_rgb["frame{}".format(i)] = colors

            self.depth_min.append(min(points_plot[:,2]))
            self.depth_max.append(max(points_plot[:,2]))

            if self.vis_depth_mode == True: 

                print("depth of frame: ", i)

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(points_plot)
                pcd.colors = open3d.utility.Vector3dVector(colors)
                visualizer = open3d.visualization.Visualizer()
                visualizer.create_window()
                visualizer.add_geometry(pcd) 
                axes_scale = 1
                # visualizer.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale, origin=[0, 0, 0]))
                visualizer.get_render_option().point_size = 3
                visualizer.run()
                visualizer.destroy_window()

        return scaled_cloud_camera_frame 

    def get_depth_map(self, image_pair_correspondence):

        # for sintel
        if self.dataset_type == 'Sintel':
            depth_gt, gt_depth_file_list = self.get_gt_depth_file_list()
        
        if self.dataset_type == 'TUM':
            depth_filelist, rgb_time_stamp, depth_time_stamp = self.get_gt_depth_file_list()

        # init rgb and depth file
        depth_dir = self.pred_path + '/depth_dpt_beit_large_512/depth/'
        predict_files = sorted(os.listdir(depth_dir))
        rgb_dir = self.pred_path + '/color_full/'
        rgb_files = sorted(os.listdir(rgb_dir))
        number_files = int(len(predict_files)/2)
        self.N = number_files

        scaled_cloud_camera_frame = defaultdict()

        for key, value in image_pair_correspondence.items():

            scaled_cloud_camera_frame[key] = defaultdict()
            
            frame1, frame2 = key
            # init indices
            valid_indices = np.where((value[:,:,0]!=-1)&(value[:,:,1]!=-1))
            y_indices = valid_indices[0]
            x_indices = valid_indices[1]
            y_frame2 = value[y_indices, x_indices, 0]
            x_frame2 = value[y_indices, x_indices, 1]


            #################################
            ##### frame1 correspondences ####
            #################################
            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[frame1*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape
            # remove points out of range in prediction
            threshold = np.percentile(predicted_depth, self.percentile_threshold)
            predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)
            # sintel
            if self.dataset_type == 'Sintel':
                predicted_depth_gt = self.get_gt_depth(depth_gt, gt_depth_file_list, frame1)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
            if self.dataset_type == 'TUM':
                predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, frame1)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean

            # read intrinsics
            fx_frame1, fy_frame1, cx_frame1, cy_frame1 = self.get_intrinsics(frame1)

            # construct points
            points = np.zeros((y_frame2.shape[0], 3))
            predicted_depth = predicted_depth[y_indices, x_indices]            
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (x_indices - cx_frame1) * predicted_depth / fx_frame1
            points[:,1] = (y_indices - cy_frame1) * predicted_depth / fy_frame1
            scaled_cloud_camera_frame[key][frame1] = points

            #################################
            ##### frame2 correspondences ####
            #################################
            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[frame2*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape
            # remove points out of range in prediction
            threshold = np.percentile(predicted_depth, self.percentile_threshold)
            predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)

            if self.dataset_type == 'Sintel':
                predicted_depth_gt = self.get_gt_depth(depth_gt, gt_depth_file_list, frame2)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
            if self.dataset_type == 'TUM':
                predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, frame1)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
            fx_frame2, fy_frame2, cx_frame2, cy_frame2 = self.get_intrinsics(frame2)            

            # construct points
            points = np.zeros((y_frame2.shape[0], 3))
            y_frame2_ceil = np.ceil(y_frame2).astype(np.int32)
            y_frame2_floor = np.floor(y_frame2).astype(np.int32)
            x_frame2_ceil = np.ceil(x_frame2).astype(np.int32)
            x_frame2_floor = np.floor(x_frame2).astype(np.int32)
            y_frame2_ceil[y_frame2_ceil == self.target_height] = self.target_height - 1
            x_frame2_ceil[x_frame2_ceil == self.target_width] = self.target_width - 1
            depth1 = predicted_depth[y_frame2_ceil, x_frame2_ceil]
            depth2 = predicted_depth[y_frame2_ceil, x_frame2_floor]
            depth3 = predicted_depth[y_frame2_floor, x_frame2_ceil]
            depth4 = predicted_depth[y_frame2_floor, x_frame2_floor]
            predicted_depth = (depth1+depth2+depth3+depth4)/4
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (x_frame2 - cx_frame2) * predicted_depth / fx_frame2
            points[:,1] = (y_frame2 - cy_frame2) * predicted_depth / fy_frame2

            scaled_cloud_camera_frame[key][frame2] = points


        ###########################
        #### For Visualization ####
        ###########################

        if self.dataset_type == 'Sintel':
            depth_gt, gt_depth_file_list = self.get_gt_depth_file_list()

        if self.dataset_type == 'TUM':
            depth_filelist, rgb_time_stamp, depth_time_stamp = self.get_gt_depth_file_list()

        # init rgb and depth file
        depth_dir = self.pred_path + '/depth_dpt_beit_large_512/depth/'
        predict_files = sorted(os.listdir(depth_dir))
        rgb_dir = self.pred_path + '/color_full/'
        rgb_files = sorted(os.listdir(rgb_dir))
        number_files = int(len(predict_files)/2)
        self.original_point_cloud = {}
        self.original_point_cloud_rgb = {}
        self.s_gt = []

        for i in range(number_files):

            # Load the depth map
            predict_file = osp.join(depth_dir,predict_files[i*2+1])
            predicted_depth_inverse = image_io.load_raw_float32_image(predict_file)
            epsilon = 0.00001
            predicted_depth = 1.0 / (epsilon + predicted_depth_inverse)  # left predicted_depth is depth in real
            h, w = predicted_depth.shape

            # remove points out of range in prediction
            threshold = np.percentile(predicted_depth, self.percentile_threshold)
            predicted_depth = np.where(predicted_depth > threshold, np.nan, predicted_depth)
            predicted_depth = np.where(predicted_depth <= 0, np.nan, predicted_depth)

            if self.dataset_type == 'Sintel':
                predicted_depth_gt = self.get_gt_depth(depth_gt, gt_depth_file_list, i)
                
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt                
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]

                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)

                    self.scale_factor = gt_depth_mean / pred_depth_mean
            if self.dataset_type == 'TUM':
                predicted_depth_gt = self.get_gt_depth(depth_filelist, rgb_time_stamp, depth_time_stamp, i)
                if self.use_gt_depth == True:
                    # Load the gt depth map
                    predicted_depth = predicted_depth_gt
                if self.scale_factor == None:
                    reshaped_pred_depth = predicted_depth.reshape(-1)
                    min_indices = np.argpartition(reshaped_pred_depth, 10000)[:10000]
                    min_values_pred = reshaped_pred_depth[min_indices]
                    reshaped_predicted_depth_gt = predicted_depth_gt.reshape(-1)
                    min_indices = np.argpartition(reshaped_predicted_depth_gt, 10000)[:10000]
                    min_values_gt = reshaped_predicted_depth_gt[min_indices]
                    pred_depth_mean = np.mean(min_values_pred)
                    gt_depth_mean = np.mean(min_values_gt)
                    self.scale_factor = gt_depth_mean / pred_depth_mean
                    tmp = 1

            # read intrinsics
            fx, fy, cx, cy = self.get_intrinsics(i)

            # construct points
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            points = np.stack((x, y, np.ones((h, w))), axis=2)
            points = np.reshape(points,(h*w,3))
            predicted_depth = np.reshape(predicted_depth,(h*w,1))
            points[:,2] = predicted_depth.reshape(-1,)
            points[:,0] = (points[:,0] - cx) * points[:,2] / fx
            points[:,1] = (points[:,1] - cy) * points[:,2] / fy

            # save original point clouds for visualization in match_frame_pair_points function
            rgb_file = osp.join(rgb_dir, rgb_files[i])
            rgb_image = cv2.imread(rgb_file, cv2.IMREAD_COLOR)
            colors = np.reshape(rgb_image, (h*w, 3))
            colors = colors[~np.isnan(points).any(axis=1)] # remove points with NaN values
            points_plot = points[~np.isnan(points).any(axis=1)] # remove points with NaN values
            assert len(points_plot) == len(colors)
            colors = colors / 255.0
            self.original_point_cloud["frame{}".format(i)] = points_plot
            self.original_point_cloud_rgb["frame{}".format(i)] = colors

            self.depth_min.append(min(points_plot[:,2]))
            self.depth_max.append(max(points_plot[:,2]))

            if self.vis_depth_mode == True: 

                print("depth of frame: ", i)

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(points_plot)
                pcd.colors = open3d.utility.Vector3dVector(colors)
                visualizer = open3d.visualization.Visualizer()
                visualizer.create_window()
                visualizer.add_geometry(pcd) 
                axes_scale = 1
                # visualizer.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale, origin=[0, 0, 0]))
                visualizer.get_render_option().point_size = 3
                visualizer.run()
                visualizer.destroy_window()

        return scaled_cloud_camera_frame 


    def scale_points(self, point_frame1, point_frame2):

        scaled_set1 = point_frame1 * self.scale_factor
        scaled_set2 = point_frame2 * self.scale_factor
        
        return scaled_set1, scaled_set2

    def set_scale_factor(self, point_frame):
        median = np.median(point_frame[:,2])
        self.scale_factor = 1 / median

    def filter_correspondence_points(self, set1, set2):
        distances = np.linalg.norm(set1 - set2, axis=1)

        sorted_distances = np.sort(distances)

        # frequencies = np.arange(len(sorted_distances)) / len(sorted_distances)
        # plt.plot(sorted_distances, frequencies)
        # plt.xlabel('Distance')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Distances')
        # plt.show()

        cutoff_percentile = np.percentile(sorted_distances, self.distance_threshold)
        print("Distance threshold: ", cutoff_percentile)

        filtered_distances_index = distances <= cutoff_percentile
        
        filtered_set1 = set1[filtered_distances_index]
        filtered_set2 = set2[filtered_distances_index]

        return filtered_set1, filtered_set2
    
    def filter_correspondence_points_based_on_gt(self, set1, set2, frame1, frame2):
        
        
        rel_pose = self.get_relative_pose_given_frames(frame1, frame2)

        diff = rel_pose[0:3,0:3] @ set1.T + rel_pose[0:3,3].reshape(3,1) - set2.T

        abs_arr = np.abs(diff)
        max_abs_index = np.argmax(abs_arr, axis=0)
        ind = np.bincount(max_abs_index, minlength=3).reshape(-1, 1)

        distances = np.linalg.norm(diff, axis=0)

        cutoff_percentile = np.percentile(distances, self.distance_threshold)
        filtered_distances_index = distances <= cutoff_percentile

        filtered_distances = distances[filtered_distances_index]
        filtered_diff = diff[:,filtered_distances_index]
        
        filtered_set1 = set1[filtered_distances_index]
        filtered_set2 = set2[filtered_distances_index]

        return filtered_set1, filtered_set2


    def match_frame_pair_points(self, scaled_cloud_camera_frame):

        # init param
        self.scaled_cloud_camera_frame_dict = {}
        x = [0]
        y = [0]
        z = [0]
        for key, value in scaled_cloud_camera_frame.items():

            frame1, frame2 = key
            point_frame1 = scaled_cloud_camera_frame[key][frame1]
            point_frame2 = scaled_cloud_camera_frame[key][frame2]
            scaled_cloud_camera_frame_one_pair = {}      
            frame1_key = 'frame{}'.format(frame1)
            frame2_key = 'frame{}'.format(frame2)

            point_frame1, point_frame2 = self.remove_nan_pairs(point_frame1, point_frame2)
            
            if self.dataset_type == 'TUM':
                if self.scale_factor == None:
                    if frame1 == 0:
                        self.set_scale_factor(point_frame1)
                    elif frame2 == 0:
                        self.set_scale_factor(point_frame2)

            # scale points to make optimization stable. Median of two sets of points are both 1 after scaling.
            assert self.scale_factor != None 

            if self.use_gt_depth == False:
                point_frame1, point_frame2 = self.scale_points(point_frame1, point_frame2)

            ################
            ### not used ###
            ################
            # point_frame1, point_frame2 = self.filter_correspondence_points_based_on_gt(point_frame1, point_frame2, frame1, frame2)

            
            # Not append if zero correspondence
            if len(point_frame1) < 10:
                print("Frame pair {} -> {} has {} correspondences. So discard this pair.".format(key[0],key[1], len(point_frame1), self.num_pairs_keep))
                continue            

            # Subsample to accelerate optimization
            print("Frame pair {} -> {} has {} correspondences before subsampling.".format(key[0],key[1], len(point_frame1)))
            assert len(point_frame1) == len(point_frame2)
            point_frame1, point_frame2 = self.subsample_point_clouds(point_frame1, point_frame2, self.num_pairs_keep)            
            print("Frame pair {} -> {} has {} correspondences.".format(key[0],key[1], len(point_frame1)))

            # Append the points to the corresponding keys in the dictionary
            scaled_cloud_camera_frame_one_pair[frame1_key] = point_frame1
            scaled_cloud_camera_frame_one_pair[frame2_key] = point_frame2
            self.scaled_cloud_camera_frame_dict[key] = scaled_cloud_camera_frame_one_pair


            # # visualize the motion of point clouds
            # if frame2 -frame1 == 1:
            #     x_axis_mean = np.mean(point_frame2[:,0]-point_frame1[:,0])
            #     y_axis_mean = np.mean(point_frame2[:,1]-point_frame1[:,1])
            #     z_axis_mean = np.mean(point_frame2[:,2]-point_frame1[:,2])

            #     x.append(x[-1] + x_axis_mean)
            #     y.append(y[-1] + y_axis_mean)
            #     z.append(z[-1] + z_axis_mean)

            # if len(x) == 40:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            #         ax.scatter(xi, yi, zi)
            #         ax.text(xi, yi, zi, str(i), color='red')

            #     translationDistance = 1
            #     ax.set_xlim(-translationDistance, translationDistance)
            #     ax.set_ylim(-translationDistance, translationDistance)
            #     ax.set_zlim(-translationDistance, translationDistance)
            #     ax.set_xlabel('X')
            #     ax.set_ylabel('Y')
            #     ax.set_zlabel('Z')
            #     ax.set_title('3D scatter plot for average motion of point clouds')
            #     plt.grid(True)
            #     ax.set_box_aspect([1, 1, 2])
            #     plt.show()


            if self.vis_final_matching_mode == True:

                # get original points
                original_point_cloud_frame1 = copy.deepcopy(self.original_point_cloud["frame{}".format(frame1)] * self.scale_factor)
                original_point_cloud_frame2 = copy.deepcopy(self.original_point_cloud["frame{}".format(frame2)] * self.scale_factor)
                original_point_cloud_frame2[:,2] = original_point_cloud_frame2[:,2] + self.pointCloudPairSeparateDistance
                original_point_cloud_rgb_frame1 = self.original_point_cloud_rgb["frame{}".format(frame1)]
                original_point_cloud_rgb_frame2 = self.original_point_cloud_rgb["frame{}".format(frame2)]
                original_pcd1_o3d = open3d.geometry.PointCloud()
                original_pcd2_o3d = open3d.geometry.PointCloud()
                original_pcd1_o3d.points = open3d.utility.Vector3dVector(original_point_cloud_frame1)
                original_pcd1_o3d.colors = open3d.utility.Vector3dVector(original_point_cloud_rgb_frame1)
                original_pcd2_o3d.points = open3d.utility.Vector3dVector(original_point_cloud_frame2)
                original_pcd2_o3d.colors = open3d.utility.Vector3dVector(original_point_cloud_rgb_frame2)

                # Create line correspondences
                pcd1_np = copy.deepcopy(point_frame1)
                pcd2_np = copy.deepcopy(point_frame2)
                pcd2_np[:,2] = pcd2_np[:,2] + self.pointCloudPairSeparateDistance # to seperate pcd1 and pcd2
                num_points = max(len(pcd1_np), len(pcd2_np))
                colors = np.zeros((num_points, 3))
                colors[:, 0] = np.linspace(0, 1, num_points)  # Red channel
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
                pcd2_o3d.paint_uniform_color([0, 1, 0]) ## green

                # Visualize the point clouds
                visualizer = open3d.visualization.Visualizer()
                visualizer.create_window()

                # Add the point clouds to the visualizer
                visualizer.add_geometry(pcd1_o3d)
                visualizer.add_geometry(pcd2_o3d)
                visualizer.add_geometry(original_pcd1_o3d)
                visualizer.add_geometry(original_pcd2_o3d)
                visualizer.add_geometry(line_set)
                axes_scale = 0.1          
                visualizer.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame\
                                                                (size=axes_scale, origin=[0, 0, 0]))
                visualizer.get_render_option().point_size = 3

                # Run the visualizer
                visualizer.run()
                visualizer.destroy_window()

    def visCameraTraj(self, solution_path):
        # Visualize the camera trajectory

        with open(solution_path, 'rb') as file:
            solution = pickle.load(file)

        if hasattr(self, 't_gt'):
            translationDistance = max(np.max(solution['t_est']), np.max(self.t_gt))
        else:
            translationDistance = np.max(solution['t_est'])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot range
        start_frame = self.start_frame_optimizer
        end_frame = self.end_frame_optimizer
        assert (start_frame >= self.start_frame_optimizer and end_frame <= self.end_frame_optimizer)
        assert (self.end_frame_optimizer <= self.whole_sequence_len)
        # set color
        cmap = plt.cm.get_cmap('cool')
        color_index = np.arange(end_frame-start_frame)
        colors = cmap(color_index/np.max(color_index))
        # Loop through camera poses
        for i in range(start_frame,end_frame):

            if hasattr(self, 't_gt'):
                # Extract the camera pose components
                position = self.t_gt[i-self.start_frame_optimizer, :]
                orientation = self.R_gt[3*(i-self.start_frame_optimizer):3*(i-self.start_frame_optimizer+1), :]

                for j in range(3):
                    ax.quiver(position[0], position[1], position[2], orientation[0, j], orientation[1, j], orientation[2, j],
                                color='b', length=translationDistance/10, normalize=True)
            
            # estimated camera pose
            position = solution['t_est'][:, i-start_frame]
            orientation = solution["R_est"][3*(i-start_frame):3*(i-start_frame+1), :]

            for j in range(3):
                ax.quiver(position[0], position[1], position[2], orientation[0, j], orientation[1, j], 
                            orientation[2, j], color=colors[i-start_frame],length = translationDistance/10, normalize=True)

        # Set axis limits
        ax.set_xlim(-translationDistance*6, translationDistance*6)
        ax.set_ylim(-translationDistance*6, translationDistance*6)
        ax.set_zlim(0, translationDistance*2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Set the title with colored words
        title = 'Camera Poses Visualization'
        blue_patch = Patch(color='blue', label='GT')
        red_patch = Patch(color='red', label='Est')
        ax.set_title(title, loc='center', fontweight='bold')

        # Add legend with colored patches
        ax.legend(handles=[blue_patch, red_patch], loc='upper right')
        plt.grid(True)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

