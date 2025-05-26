from PIL import Image, ImageOps, ImageFilter

import torch

#from .augmenter import Augmenter
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision
import random
from torchvision import transforms
import numpy as np
from glob import glob
import cv2
from matplotlib import pyplot as plt
# from skimage import io
# from skimage.feature import ORB, match_descriptors
# from skimage.transform import EuclideanTransform, warp
# from skimage.measure import ransac


class VRGazeDatasetUnsupervised(Dataset):
    def __init__(self, args, data_dir_root, labels=None, stage=None):
        super().__init__()
        self.args = args
        self.root_dir = data_dir_root
        self.labels = labels

        self.groups = dict(tuple(labels.groupby(['person_id'])))
        self.group_keys = list(self.groups.keys())
        self.group_keys = [key for key in self.group_keys if len(self.groups[key]) >= 2]
        self.rng = random.Random(41)
        self.to_tensor = torchvision.transforms.ToTensor()

        #self.normalize = torchvision.transforms.Normalize(0.25,0.15)
        #self.normalize = torchvision.transforms.Normalize(0.25, 0.15)
        self.gaussian_blur = torchvision.transforms.GaussianBlur(5, sigma=(2.0, 2.0))
        self.normalize = torchvision.transforms.Normalize(0.5, 0.5)
        self.random_crop = torchvision.transforms.RandomCrop((400, 400), 50, True, padding_mode='constant')
        self.resize = torchvision.transforms.Resize((args.input_height, args.input_width))
        #self.random_crop = torchvision.transforms.RandomCrop((480, 640), 100, True)
        self.random_scale_crop = torchvision.transforms.RandomResizedCrop((400, 400),scale=(1.5, 2.0), ratio=(1,1))
        #self.random_auto_contrast = torchvision.transforms.RandomAutocontrast(1.0)
        self.random_photometric_distort = torchvision.transforms.ColorJitter(brightness=(0.2,1.1),contrast=(0.6, 1), saturation=(0.8,1))
        self.random_erasing = torchvision.transforms.RandomErasing(p=0.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        #self.random_erasing = torchvision.transforms.RandomErasing(p=0.5, scale=(0.1, 0.8), ratio=(0.3, 3.3), value=0,  inplace=False)
        self.to_pil = torchvision.transforms.ToPILImage()
        self.channels = 1
        self.batch_per_person = args.batch_per_person
        self.stage = stage
        # 'L', 'R' or 'LR'
        self.binocular_mode = 'LR'
        if self.stage == 'predict':
            self.image_files_left = sorted(glob(data_dir_root + '/*L*.png'))
            self.image_files_right = sorted(glob(data_dir_root + '/*R*.png'))
            assert len(self.image_files_left) == len(self.image_files_right)

        #torchvision.transforms.Grayscale(num_output_channels=3)
        #gil = test no random crop
        #is_train = False
        self.is_train = (stage == 'train')
        if self.is_train:
            self.transform = self.transform_train
        else:
            self.transform = self.transform_test

    def estimate_rigid_transformation(self, src_pts, dst_pts):
        """
        Estimate a rigid transformation (rotation and translation) that aligns src_pts to dst_pts.
        Both src_pts and dst_pts must be of shape (N, 2), where N is the number of points.
        """
        # Ensure the points are numpy arrays
        # src_pts = np.array(src_pts, dtype=np.float64)
        # dst_pts = np.array(dst_pts, dtype=np.float64)

        # Subtract centroids
        src_center = np.mean(src_pts, axis=0)
        dst_center = np.mean(dst_pts, axis=0)
        src_pts_centered = src_pts - src_center
        dst_pts_centered = dst_pts - dst_center

        # Compute the covariance matrix
        H = src_pts_centered.T @ dst_pts_centered

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation vector
        t = dst_center - R @ src_center

        # Create the homogeneous transformation matrix
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t

        return T

    def warp_LR2(self, left_image, right_image):

        # Load the images
        fixed_image = np.array(left_image)
        moving_image = np.array(right_image)

        # Initialize ORB detector
        orb = ORB(n_keypoints=500, fast_threshold=0.05)

        # Detect and extract features from both images
        orb.detect_and_extract(fixed_image)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors

        orb.detect_and_extract(moving_image)
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors

        # Match descriptors between images
        matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

        # Extract the matched keypoints
        src = keypoints2[matches[:, 1]][:, ::-1]
        dst = keypoints1[matches[:, 0]][:, ::-1]

        # Estimate the transformation model using RANSAC to robustly estimate the model
        model_robust, inliers = ransac((src, dst), EuclideanTransform, min_samples=3,
                                       residual_threshold=2, max_trials=1000)

        # Warp the moving image using the estimated transformation model
        registered_image = warp(moving_image, model_robust.inverse, output_shape=fixed_image.shape)


        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        ax[0, 0].imshow(fixed_image, cmap='gray')
        ax[0, 0].set_title('Fixed Image')

        ax[0, 1].imshow(moving_image, cmap='gray')
        ax[0, 1].set_title('Moving Image')

        ax[1, 0].imshow(registered_image, cmap='gray')
        ax[1, 0].set_title('Registered Image')

        # Plotting the inliers
        ax[1, 1].imshow(fixed_image, cmap='gray')
        ax[1, 1].autoscale(False)
        ax[1, 1].plot(src[inliers, 0], src[inliers, 1], '.r')
        ax[1, 1].plot(dst[inliers, 0], dst[inliers, 1], '.b')
        ax[1, 1].set_title('Inliers')
        plt.show()
        registered_image = Image.fromarray(registered_image)
        return registered_image

    def warp_LR(self, left_image, right_image):

        left_image = np.array(left_image)
        right_image = np.array(right_image)
        # Load the fixed and moving images
        fixed_image = left_image
        moving_image = right_image

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find keypoints and descriptors with ORB
        keypoints1, descriptors1 = orb.detectAndCompute(fixed_image, None)
        keypoints2, descriptors2 = orb.detectAndCompute(moving_image, None)

        # Create BFMatcher object and match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of matched keypoints
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Estimate affine matrix
        #M, inliers = cv2.estimateAffinePartial2D(points2, points1)
        M = self.estimate_rigid_transformation(points2, points1)
        # Apply transformation
        height, width = fixed_image.shape[:2]
        registered_image = cv2.warpAffine(moving_image, M, (width, height))

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(fixed_image, cmap='gray')
        axs[0].set_title('Fixed Image')
        axs[0].axis('off')

        axs[1].imshow(moving_image, cmap='gray')
        axs[1].set_title('Moving Image')
        axs[1].axis('off')

        axs[2].imshow(registered_image, cmap='gray')
        axs[2].set_title('Registered Image')
        axs[2].axis('off')
        # Display results
        registered_image = Image.fromarray(registered_image)
        return registered_image

    def transform_train(self, image):
        #angle = random.randint(-10, 10)
        #im = self.random_scale_crop(image)
        #im = torchvision.transforms.functional.rotate(im, angle)

        #im = image.filter(ImageFilter.GaussianBlur(radius=2))
        #im = self.gaussian_blur(image)
        #im = self.random_photometric_distort(image)
        #im = self.random_crop(im)
        im = self.resize(image)
        im = self.to_tensor(im)
        im = self.normalize(im)
        return im

    def transform_test(self, image):
        #im = self.gaussian_blur(image)
        im = self.resize(image)
        im = self.to_tensor(im)
        #random_erasing_im = self.random_erasing(im)
        #random_erasing_im = self.normalize(random_erasing_im)

        im = self.normalize(im)
        return im

    def __len__(self):
        if self.stage == 'predict':
            return len(self.image_files_left)
        else:
            return len(self.labels)


    def from_yaw_pitch_to_3D(self, xy_angles):
        pitch = xy_angles[1]
        yaw = xy_angles[0]

        y = torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        x = z * torch.tan(yaw)

        result = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],dim=0)
        return result

    def normalize_vector(self, v):
        """Normalize a vector."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def get_id(self, file_name):
        import re

        # Regular expression to extract the number 'N' at the beginning of the file name
        match = re.match(r"(\d+)_.*\.png", file_name)

        # Extract 'N' if the pattern matches
        if match:
            N = match.group(1)
        else:
            N = None

        return N

    def __getitem__(self, idx):

        # Randomly select a group
        group_key = random.choice(self.group_keys)
        group_df = self.groups[group_key]

        # Sample 2 distinct rows from the group
        samples = group_df.sample(n=2, replace=False)
        sample_1, sample_2 = samples.iloc[0], samples.iloc[1]

        image_l_1 = os.path.join(self.root_dir, 'train',sample_1['image_l'])
        image_r_1 = os.path.join(self.root_dir, 'train',sample_1['image_r'])
        image_l_2 = os.path.join(self.root_dir, 'train',sample_2['image_l'])
        image_r_2 = os.path.join(self.root_dir, 'train',sample_2['image_r'])

        pil_image_l_1 = self.transform(ImageOps.grayscale(Image.open(image_l_1)))
        pil_image_r_1 = self.transform(ImageOps.grayscale(Image.open(image_r_1)))
        pil_image_l_2 = self.transform(ImageOps.grayscale(Image.open(image_l_2)))
        pil_image_r_2 = self.transform(ImageOps.grayscale(Image.open(image_r_2)))
        #img_path = torch.tensor(img_path)

        label = self.labels.iloc[idx]
        gt = torch.tensor(0)
        dict_label = label.to_dict()

        return gt, pil_image_l_1, pil_image_r_1, pil_image_l_2, pil_image_r_2, dict_label


    def __getitem__orig(self, idx):


        if self.stage == 'predict':
            img_path_left = self.image_files_left[idx]
            pil_image_left = Image.open(img_path_left)
            pil_image_left = ImageOps.grayscale(pil_image_left)
            left_image = self.transform(pil_image_left)

            img_path_right = self.image_files_right[idx]
            pil_image_right = Image.open(img_path_right)
            pil_image_right = ImageOps.grayscale(pil_image_right)
            right_image = self.transform(pil_image_right)

            frame_id = self.get_id(os.path.basename(img_path_left))
            return left_image, right_image, frame_id


        label = self.labels.iloc[idx]
        gt = torch.tensor(0)
        if hasattr(label,'vec_3d_x'):
            gt = [label['vec_3d_x'], label['vec_3d_y'], label['vec_3d_z']]
            gt = torch.tensor(gt)
            gt = gt.to(torch.float32)

        L_file = os.path.join(self.root_dir, self.stage,  label['image_l'])
        R_file = os.path.join(self.root_dir, self.stage, label['image_r'])

        left_image = Image.open(L_file)
        right_image = Image.open(R_file)

        left_image = ImageOps.grayscale(left_image)
        right_image = ImageOps.grayscale(right_image)
        left_image = self.transform(left_image)
        right_image = self.transform(right_image)
        dict_label = label.to_dict()

        return gt, left_image, right_image, dict_label

        ######################################
        # if self.stage == 'predict':
        #     img_path_left = self.image_files_left[idx]
        #     pil_image_left = Image.open(img_path_left)
        #     pil_image_left = ImageOps.grayscale(pil_image_left)
        #     left_image = self.transform(pil_image_left)
        #
        #     img_path_right = self.image_files_right[idx]
        #     pil_image_right = Image.open(img_path_right)
        #     pil_image_right = ImageOps.grayscale(pil_image_right)
        #     right_image = self.transform(pil_image_right)
        #
        #     frame_id = self.get_id(os.path.basename(img_path_left))
        #     return left_image, right_image, frame_id
        #
        #
        # label = self.labels.iloc[idx]
        # gt = torch.tensor(0)
        # if hasattr(label,'vec_3d_x'):
        #     gt = [label['vec_3d_x'], label['vec_3d_y'], label['vec_3d_z']]
        #     gt = torch.tensor(gt)
        #     gt = gt.to(torch.float32)
        #
        # ADD_NOISE = False
        # if ADD_NOISE and self.is_train:
        #     angle = random.random() * 1.0
        #     angle_rad = np.radians(angle)
        #     random_axis = self.normalize_vector(np.random.rand(3))
        #     # Create the rotation matrix using the Rodrigues' rotation formula
        #     rotation_matrix = np.eye(3) + np.sin(angle_rad) * np.cross(np.eye(3), random_axis) + (1 - np.cos(angle_rad))\
        #                       * np.outer(random_axis, random_axis)
        #     gt = np.dot(rotation_matrix, gt)
        #     gt = self.normalize_vector(gt)
        #
        #
        # L_file = label['frame_id_L']
        # R_file = label['frame_id_R']
        #
        # left_image = Image.open(L_file)
        # right_image = Image.open(R_file)
        # person_id = label['id']
        #
        # left_image = ImageOps.grayscale(left_image)
        # right_image = ImageOps.grayscale(right_image)
        #
        # #warped_right = self.warp_LR2(left_image, right_image)
        #
        # left_image = self.transform(left_image)
        # right_image = self.transform(right_image)
        # #right_image = self.transform(right_image)
        # dict_label = label.to_dict()
        # return gt, left_image, right_image, dict_label



