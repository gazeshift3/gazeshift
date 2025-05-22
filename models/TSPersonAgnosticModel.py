import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
import torch
import numpy as np
from models.singleEye3DGazeTimm import singleEye3DGazeTimmModel
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image, ImageOps
import random
import torchvision
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import normalize
from models.mbnv2 import MobileNet_v2
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from sklearn.tree import DecisionTreeRegressor
from scipy.interpolate import Rbf
from scipy.special import softmax
from scipy.optimize import curve_fit
from scipy.special import comb
import ast

import wandb
plt.ioff()
def deactivate_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()



class SmallHead(LightningModule):
    def __init__(self, hparams):
        super(SmallHead, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        args = self.hparams['hparams']
        # width_mult_to_depth = {
        #     1.0: 32,
        #     0.75: 24,
        #     2.0: 128,
        # }
        width_mult_to_depth = {
            1.0: 64,
            0.75: 24,
            2.0: 128,
            3.0: 192,
        }

        self.mono_feature_depth = width_mult_to_depth[args.width_multiplier]
        self.output_feature_depth = 3
        self.conv1 = torch.nn.Conv2d(self.mono_feature_depth,  self.mono_feature_depth // 2 ,1,1)
        self.conv2 = torch.nn.Conv2d(self.mono_feature_depth // 2, 3, 1, 1)

    def forward(self, features):

        features = self.conv1(features)
        features = self.conv2(features).squeeze(-1).squeeze(-1)

        por_estimation = torch.tensor([0,0,1.0], device=features.device) + features
        por_estimation = torch.nn.functional.normalize(por_estimation)

        return por_estimation



class Head(LightningModule):
    def __init__(self, hparams):
        super(Head, self).__init__()
        self.save_hyperparameters()  # sets self.hparams
        args = self.hparams['hparams']
        # width_mult_to_depth = {
        #     1.0: 32,
        #     0.75: 24,
        #     2.0: 128,
        # }
        width_mult_to_depth = {
            1.0: 64,
            0.75: 24,
            2.0: 128,
            3.0: 192,
        }

        #self.input_feature_depth = 96
        #self.mono_feature_depth = 32
        #self.mono_feature_depth = 24
        #self.mono_feature_depth = 48
        self.mono_feature_depth = width_mult_to_depth[args.width_multiplier]
        #self.input_feature_depth = 32
        self.output_feature_depth = 3
        #self.conv1_att = torch.nn.Conv2d(self.input_feature_depth, 1, 1)
        self.conv1_bino = torch.nn.Conv2d(self.mono_feature_depth * 2, self.mono_feature_depth ,1,1)
        self.conv2 = torch.nn.Conv2d(self.mono_feature_depth, 3, 1, 1)

        # for param in self.conv1_bino.parameters():
        #     param.requires_grad = False

        #self.conv3 = torch.nn.Conv2d(self.output_feature_depth // 2, 3, 1, 1)

    def forward_mono(self, backbone_features):

        #concat_features = torch.cat(1(left_backbone_features, right_backbone_features), dim=1)
        #features = self.conv1_bino(backbone_features)
        features = self.conv2(backbone_features).squeeze(-1).squeeze(-1)

        #features = torch.tanh(features)

        por_estimation = torch.tensor([0,0,1.0], device=features.device) + features
        por_estimation = torch.nn.functional.normalize(por_estimation)

        return por_estimation, features

    def forward_bino(self, left_backbone_features, right_backbone_features):

        concat_features = torch.cat((left_backbone_features, right_backbone_features), dim=1)
        features = self.conv1_bino(concat_features)
        features = self.conv2(features).squeeze(-1).squeeze(-1)

        por_estimation = torch.tensor([0,0,1.0], device=features.device) + features
        por_estimation = torch.nn.functional.normalize(por_estimation)

        return por_estimation, concat_features



    def forward(self, left_backbone_features, right_backbone_features=None):
        # if right_backbone_features is None:
        #     return self.forward_mono(left_backbone_features)
        # else:
        return self.forward_bino(left_backbone_features, right_backbone_features)


#############################################################################################






class TSPersonAgnosticModel(LightningModule):
    def __init__(self, hparams, channels=1):
        super(TSPersonAgnosticModel, self).__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()  # sets self.hparams
        self.params = self.hparams['hparams']
        args = self.hparams['hparams']

        # bottleneckLayerDetails = [
        #     # (expansion, out_dimension, number_of_times, stride)
        #     (1, 4, 1, 2),
        #     (3, 8, 1, 2),
        #     (3, 16, 1, 2),
        #     (3, 32, 1, 2),
        #     (1, 64, 1, 2),
        #     (1, 64, 1, 2),
        # ]
        bottleneckLayerDetails = [
            # (expansion, out_dimension, number_of_times, stride)
            (1, 4, 1, 2),
            (6, 8, 1, 2),
            (6, 16, 1, 2),
            (6, 32, 1, 2),
            (6, 48, 1, 2),
            (6, 64, 1, 2),
        ]

        # bottleneckLayerDetails = [
        #     # (expansion, out_dimension, number_of_times, stride)
        #     (1, 4, 1, 2),
        #     (3, 8, 1, 2),
        #     (3, 16, 1, 2),
        #     (3, 32, 1, 2),
        #     (1, 48, 1, 2),
        #     (1, 64, 1, 2),
        # ]

        self.error_list = []
        self.backbone = MobileNet_v2(bottleneckLayerDetails, width_multiplier=self.params.width_multiplier, in_fts=1)

        # self.backbone.load_state_dict(
        # torch.load(hparams.ckpt_path, map_location=self.device)['state_dict'])
        self.avg_pool = torch.nn.AvgPool2d((3, 4))
        self.max_pool = torch.nn.MaxPool2d((3, 4))
        self.training_estimation = []
        self.training_labels = []
        self.loss_val = torch.nn.Parameter(torch.tensor(0.0))
        self.backbone_feaure_size = self.params.backbone_feature_size
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(0.25,0.15)
        self.random_crop = torchvision.transforms.RandomCrop((480, 640), 100, True, padding_mode='edge')
        self.resize = torchvision.transforms.Resize((args.input_height, args.input_width))
        self.channels = 1
        self.validation_it = 0
        self.head = Head(args)
        self.small_head_l = SmallHead(args)
        self.small_head_r = SmallHead(args)
        #self.load_from_checkpoint(hparams.ckpt_path, strict=False)
        if hparams.ckpt_path != '':
            self.load_state_dict(torch.load(hparams.ckpt_path, map_location=self.device)['state_dict'], strict=False)
        pass
        # #Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for param in self.parameters():
        #     param.requires_grad = False

    def freeze_batch_norm(self):
        for module in self.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_parameters()
                module.eval()
                with torch.no_grad():
                    module.weight.fill_(1.0)
                    module.bias.zero_()

                # module.eval()
                # # print(layer)
                # for param in module.parameters():
                #     param.requires_grad = False


    def get_current_lr(self):
        for param_group in self.optimizers().param_groups:
            return param_group['lr']


    def set_mode(self, mode):
        self.mode = mode



    def forward(self, images_left, images_right):
        features_left = self.backbone(images_left)
        features_right = self.backbone(images_right)
        por_estimation, out_features = self.head(features_left, features_right)

        return por_estimation, out_features

    def forward_bino(self, images_left, images_right):
        features_left = self.backbone(images_left)
        features_right = self.backbone(images_right)
        por_estimation_left = self.small_head_l(features_left)
        por_estimation_right = self.small_head_r(features_right)
        por_estimation, out_features = self.head(features_left, features_right)
        features_left = torch.nn.functional.normalize(features_left)
        features_right = torch.nn.functional.normalize(features_right)
        return por_estimation, out_features, features_left, features_right, por_estimation_left, por_estimation_right


    def forward_mono(self, images):
        features = self.backbone(images)

        por_estimation, out_features = self.head(features)

        return por_estimation, out_features



    def gaze_vector_to_yaw_pitch_np(self, gaze_labels):
        x = gaze_labels[:,0]
        y = gaze_labels[:,1]
        z = gaze_labels[:,2]

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)

        pitch = np.arcsin(y)
        yaw = np.arctan2(x, z)

        concat = np.concatenate([yaw, pitch], axis=1)
        return concat


    def from_yaw_pitch_to_3D_np(self, xy_angles):
        pitch = xy_angles[:,1]
        yaw = xy_angles[:,0]

        y = np.sin(pitch)
        z = np.cos(pitch) * np.cos(yaw)
        x = z * np.tan(yaw)

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)

        result = np.concatenate([x,y,z],axis=1)
        return result


    def from_3D_to_yaw_pitch_np(self, gaze_labels):
        x = gaze_labels[:,0]
        y = gaze_labels[:,1]
        z = gaze_labels[:,2]

        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)


        pitch = np.arcsin(y)
        yaw = np.arctan2(x, z)

        concat = np.concatenate([yaw, pitch], axis=1)
        return concat


    def from_3D_to_yaw_pitch(self, vec_3d):
        x = vec_3d[:,0]
        y = vec_3d[:, 1]
        z = vec_3d[:, 2]
        pitch = torch.asin(y).unsqueeze(dim=1)
        yaw = torch.atan2(x, z).unsqueeze(dim=1)

        final = torch.cat((pitch,yaw),dim=1)
        return final

    def from_yaw_pitch_to_3D(self, xy_angles):
        pitch = xy_angles[:,1]
        yaw = xy_angles[:,0]

        y = torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        x = z * torch.tan(yaw)

        result = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],dim=1)
        return result




    def transform_train(self, image):
        #im = self.random_crop(image)
        im = self.resize(image)
        im = self.to_tensor(im)
        im = self.normalize(im)
        if self.channels == 3:
            im = im.expand(3,-1,-1)

        return im

    def transform_val(self, image):
        im = self.resize(image)
        im = self.to_tensor(im)
        im = self.normalize(im)
        if self.channels == 3:
            im = im.expand(3,-1,-1)

        return im


    def prepare_batch_val(self, selected_batch):
        gt = selected_batch[0]
        images_paths = selected_batch[1]
        subject_ids = selected_batch[2]
        images = []
        for image_path in images_paths:
            pil_image = Image.open(image_path)
            if self.channels == 1:
                pil_image = ImageOps.grayscale(pil_image)

            image = self.transform_val(pil_image)
            images.append(image.unsqueeze(0))

        images = torch.cat(images)
        return images, (gt, images_paths, subject_ids)



    def prepare_batch_train(self, selected_batch):
        gt = selected_batch[0][0]
        images_paths = selected_batch[0][1]
        subject_ids = selected_batch[0][2]
        images = []
        for image_path in images_paths:
            pil_image = Image.open(image_path)
            if self.channels == 1:
                pil_image = ImageOps.grayscale(pil_image)

            image = self.transform_train(pil_image)
            images.append(image.unsqueeze(0))

        images = torch.cat(images)
        return images, (gt, images_paths, subject_ids)

    def generate_polynomial_features(self, input, degree):
        # Create an empty list to store the polynomial features
        features = []

        for d in range(1, degree + 1):
            # Compute the polynomial features of degree d
            features_d = input ** d
            features.append(features_d)

        # Concatenate the features along the last dimension to get the final polynomial features
        return torch.cat(features, dim=-1)


    def compute_loss_no_calib(self,estimated_por, gaze_labels_3d, error_weight):
        #estimated_gaze_after_calib_3d = self.from_yaw_pitch_to_3D(estimated_por)
        #gaze_labels_3d = self.from_yaw_pitch_to_3D(gaze_labels)

        average_error, avg_dist, median_err = self.calc_metrics(estimated_por, gaze_labels_3d, error_weight)
        return average_error, avg_dist, median_err


    def compute_loss_diff(self, estimated_por, gaze_labels_3d, error_weight):

        estimated_por_2d = self.from_3D_to_yaw_pitch(estimated_por)
        gaze_labels_2d =  self.from_3D_to_yaw_pitch(gaze_labels_3d)
        estimated_por_extended = self.generate_polynomial_features(estimated_por_2d, self.params.poly_calib_degree_train)
        #estimated_por_extended = estimated_por
        try:
            solution = torch.linalg.lstsq(estimated_por_extended, gaze_labels_2d)
            estimated_por_after_calib_2d = estimated_por_extended @ solution.solution
        except:
            print('Fitting failed')

        estimated_por_after_calib_3d = self.from_yaw_pitch_to_3D(estimated_por_after_calib_2d)
        gaze_labels_3d = self.from_yaw_pitch_to_3D(gaze_labels_2d)
        #estimated_por_no_calib = self.from_yaw_pitch_to_3D(estimated_por_no_calib)
        # estimated_gaze_after_calib_3d = self.from_yaw_pitch_to_3D(estimated_por_after_calib_2d)
        # gaze_labels_3d = self.from_yaw_pitch_to_3D(gaze_labels_2d)

        average_error, avg_dist, median_error = self.calc_metrics(estimated_por_after_calib_3d, gaze_labels_3d, error_weight)
        #average_no_error, avg_dist = self.calc_metrics(estimated_por_no_calib, gaze_labels_3d)
        return average_error, avg_dist, median_error


    def compute_loss(self, estimated_por, gaze_labels):

        estimated_por_np = estimated_por.squeeze(-1).squeeze(-1).detach().cpu().numpy()
        gaze_labels_np = gaze_labels.cpu().numpy()

        poly = PolynomialFeatures(self.poly_adapter_degree)
        estimated_por_extended = poly.fit_transform(estimated_por_np)

        sample_size = len(estimated_por_np) * 2 // 3
        per_person_calib_for_fit = random.sample(range(len(estimated_por_np)), sample_size)
        estimated_por_for_fit = np.array([estimated_por_extended[i] for i in per_person_calib_for_fit])
        labels_for_fit = np.array([gaze_labels_np[i] for i in per_person_calib_for_fit])
        estimated_por_extended = torch.tensor(estimated_por_extended,
                     device=estimated_por.device, dtype=torch.float32)

        estimated_por_for_fit = torch.tensor(estimated_por_for_fit,
                     device=estimated_por.device, dtype=torch.float32)
        labels_for_fit = torch.tensor(labels_for_fit,
                     device=estimated_por.device, dtype=torch.float32)
        solution = torch.linalg.lstsq(estimated_por_for_fit, labels_for_fit)
        estimated_por_after_calib = estimated_por_extended @ solution.solution
        estimated_gaze_after_calib_3d = self.from_yaw_pitch_to_3D(estimated_por_after_calib)
        gaze_labels_3d = self.from_yaw_pitch_to_3D(gaze_labels)

        average_error, avg_dist = self.calc_metrics(estimated_gaze_after_calib_3d, gaze_labels_3d)

        loss = estimated_por.sum() / estimated_por.sum() * average_error
        return loss

    def predict_step(self, batch, batch_idx):
        self.eval()
        images_left, images_right, frame_id = batch

        estimated_gaze, error_estimate = self.forward(images_left, images_right)
        return estimated_gaze, frame_id


    def on_train_epoch_start(self):
        self.error_list = []

    def training_step(self, batch, batch_idx):

        gaze_labels, left_images, right_images, label = batch
        # subject_ids = label['id']
        # session_numbers = label['session_number']

        #gaze_labels, left_images, right_images, subject_ids, L_file, R_file, error_weight = batch
        if self.params.binocular_mode:
            estimated_por, _, left_features, right_features, estimated_por_left,\
                estimated_por_right = self.forward_bino(left_images, right_images)
        else:
            estimated_por, _ = self.forward_mono(left_images)
        #As we are not doing back-prop but least squares we work in eval() mode only
        batch_size = left_features.shape[0]
        if self.params.embedding_loss:
            batch_inner_product = torch.einsum('ij, ij -> i ', left_features.squeeze(), right_features.squeeze())
            cosine_dist = 1 - batch_inner_product
            avg_cos_dist = cosine_dist.mean()
            self.log('L_R_cos_dist', avg_cos_dist, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)

        else:
            w = 0
            error_weight = torch.tensor([(1 + w * x) for x in label['is_stationary']], device=left_images.device)

            if self.params.train_inline_calib:
                loss_f = self.compute_loss_diff
            else:
                loss_f = self.compute_loss_no_calib
            avg_error_both, avg_cos_dist_both, median_err_both = loss_f(estimated_por, gaze_labels, error_weight)
            avg_error_left, avg_cos_dist_left, median_err_left = loss_f(estimated_por_left,                                                  gaze_labels,
                                                                                            error_weight)
            avg_error_right, avg_cos_dist_right, median_err_right = loss_f(estimated_por_right,
                                                                                                gaze_labels,
                                                                                                error_weight)
            avg_cos_dist = avg_cos_dist_both + avg_cos_dist_left + avg_cos_dist_right
            #self.error_list.append(float(avg_error.detach().cpu().numpy()))
            self.log('avg_error_deg', avg_error_both, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
            self.log('avg_cos_dist', avg_cos_dist_both, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)

            self.log('median_err', median_err_both, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)

        lr = self.get_current_lr()
        # log
        self.log('lr', lr, on_step=True, on_epoch=False, logger=True, sync_dist=False, batch_size=batch_size)

        return avg_cos_dist
        #return avg_error


    def calc_metrics_np(self, estimated_gaze, gaze_lables):
        batch_inner_product = np.einsum('ij, ij -> i ', estimated_gaze, gaze_lables)
        cosine_dist = 1 - batch_inner_product
        avg_dist_per_sample = np.mean(cosine_dist)
        eps = 1e-10
        batch_inner_product = np.clip(batch_inner_product, -1+eps, 1-eps )

        error_deg = np.arccos(batch_inner_product) / np.pi * 180.0
        average_error_deg_per_sample = np.mean(error_deg)
        median_error = np.median(error_deg)


        return average_error_deg_per_sample, avg_dist_per_sample, error_deg, median_error

    def calc_metrics(self, estimated_gaze, gaze_lables, error_weight):
        batch_inner_product = torch.einsum('ij, ij -> i ', estimated_gaze, gaze_lables)
        cosine_dist = 1 - batch_inner_product
        #Weighted average by point stationarity attribute
        avg_dist_per_sample = (cosine_dist * error_weight).sum() / error_weight.sum()
        #avg_dist_per_sample = torch.mean(cosine_dist)
        eps = 1e-10
        batch_inner_product = torch.clamp(batch_inner_product, min=-1+eps, max=1-eps )
        error_deg = torch.acos(batch_inner_product) / torch.pi * 180.0

        error_median = torch.median(error_deg)

        average_error_deg_per_sample = (error_deg * error_weight).sum() / error_weight.sum()

        return average_error_deg_per_sample, avg_dist_per_sample, error_median

    def training_epoch_end(self, outputs):
        return None


    def farthest_point_sampling(self, points, num_points):
        """
        Performs farthest point sampling on a set of 3D points using cosine distance.

        Args:
        - points (numpy array of shape (N, 3)): the set of points to sample from
        - num_points (int): the number of points to sample

        Returns:
        - indices (numpy array of shape (num_points,)): the indices of the sampled points in the original set
        """
        points = normalize(points, axis=1)
        indices = [np.argmax(points[:,0])]
        #indices = [0]

        # Compute the cosine distances between the first point and all the other points
        cos_dists = 1 - np.dot(points[0], points.T) / (np.linalg.norm(points[0]) * np.linalg.norm(points, axis=1))

        # Iterate until the desired number of points is reached
        while len(indices) < num_points:
            # Find the index of the point with the maximum cosine distance to the current set of sampled points
            max_index = np.argmax(cos_dists)

            # Add the index to the list of sampled indices
            indices.append(max_index)

            # Compute the cosine distances between the newly added point and all the other points
            cos_dists_new = 1 - np.dot(points[max_index], points.T) / (
                        np.linalg.norm(points[max_index]) * np.linalg.norm(points, axis=1))

            # Update the cosine distances to be the minimum between the current and the new distances
            cos_dists = np.minimum(cos_dists, cos_dists_new)

        # Convert the list of sampled indices to a numpy array
        indices = np.array(indices)

        return indices


    def compute_fit_ridge(self, estimated_por_calc, gt_calc, poly_deg=2, kernel='linear'):
        poly = PolynomialFeatures(poly_deg)
        model = KernelRidge(alpha=0.001, kernel=kernel)
        #model = KernelRidge(alpha=0.001, kernel='cosine')
        #model = MLPRegressor(hidden_layer_sizes=10, activation='tanh')
        estimated_gaze_poly = poly.fit_transform(estimated_por_calc)
        #estimated_gaze_poly = estimated_por_calc

        #model = Ridge()
        model.fit(estimated_gaze_poly, gt_calc)
        return model, poly


    def compute_fit(self, estimated_por_calc, gt_calc, poly_deg=1):
        poly = PolynomialFeatures(poly_deg)
        estimated_gaze_poly = poly.fit_transform(estimated_por_calc)
        model = LinearRegression()
        #model = Ridge()
        model.fit(estimated_gaze_poly, gt_calc)
        return model, poly


    def compute_fit_svr(self,estimated_por_calc, gt_calc, poly_deg=1):
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

        source_x = estimated_por_calc[:, 0]
        source_y = estimated_por_calc[:, 1]

        # Separate the target points into x and y coordinates
        target_x = gt_calc[:, 0]
        target_y = gt_calc[:, 1]

        # Create RBF interpolators for the x and y mappings
        svr_x = svr.fit(estimated_por_calc, target_x)
        svr_y = svr.fit(estimated_por_calc, target_y)

        return svr_x, svr_y

        return svr


    def compute_fit_knn(self,estimated_por_calc, gt_calc, poly_deg=1):
        knn = KNeighborsRegressor(n_neighbors=2)
        knn.fit(estimated_por_calc, gt_calc)
        return knn

    def compute_fit_rbf(self, estimated_por_calc, gt_calc, poly_deg=1):
        source_x = estimated_por_calc[:, 0]
        source_y = estimated_por_calc[:, 1]

        # Separate the target points into x and y coordinates
        target_x = gt_calc[:, 0]
        target_y = gt_calc[:, 1]

        # Create RBF interpolators for the x and y mappings
        rbf_x = Rbf(source_x, source_y, target_x, function='multiquadric')
        rbf_y = Rbf(source_x, source_y, target_y, function='multiquadric')

        return rbf_x, rbf_y


    def calc_validation_loss(self, person_data):
        person_data_size = len(person_data)
        #self.person_training_set_size = 10
        #Group the person data by the gt
        # Split the data by gaze labels to train/test. Compute calibration on train and test on test
        if self.params.calib_sampling_mathod == 'FPS':
            gt = person_data['gt'].to_numpy()
            gt = np.concatenate(gt, axis=0).reshape(person_data_size, 2)
            fps_indexes = self.farthest_point_sampling(gt, self.params.validation_fit_size)
            calib_train_data_for_person = person_data.iloc[fps_indexes]
            a = calib_train_data_for_person['gt'].to_numpy()
            a = np.concatenate(a, axis=0).reshape(len(a), 2)
            calib_train_data_for_person = person_data[person_data['gt'].apply(lambda point: point in a)]

        else:
            #random sampling
            self.train_data_for_person = person_data.sample(n=self.params.validation_fit_size)

        print('person calib train data size: ' + str(len(calib_train_data_for_person)))
        gt_calc = calib_train_data_for_person['gt'].to_numpy()
        gt_calc = np.concatenate(gt_calc, axis=0).reshape(len(gt_calc), 2)
        #gt_calc = self.from_3D_to_yaw_pitch_np(gt_calc)
        estimated_por_calc = calib_train_data_for_person['por_estimation'].to_numpy()
        estimated_por_calc = np.concatenate(estimated_por_calc, axis=0).reshape(len(gt_calc), 3)

        features_calc = calib_train_data_for_person['features'].to_numpy()
        features_calc = np.concatenate(features_calc, axis=0).reshape(len(gt_calc), 32)

        fit_transform_feature, poly_feature = self.compute_fit(features_calc, gt_calc, poly_deg=2)

        fit_transform_por, poly_por = self.compute_fit(estimated_por_calc, gt_calc, poly_deg=self.params.poly_calib_degree_train)

        print('all person labels count: ' + str(len(person_data)))
        for gt in gt_calc:
            person_data = person_data[~person_data['gt'].apply(lambda x: np.array_equal(x, gt))]

        calib_val_data_for_person = person_data
        print('person calibration test size: ' + str(len(calib_val_data_for_person)))
        #calib_val_data_for_person = person_data.drop(self.train_data_for_person.index)


        calib_features = calib_val_data_for_person['features'].to_numpy()
        calib_features = np.concatenate(calib_features, axis=0).reshape(len(calib_val_data_for_person), 32)

        calib_por_estimation = calib_val_data_for_person['por_estimation'].to_numpy()
        calib_por_estimation = np.concatenate(calib_por_estimation, axis=0).reshape(len(calib_val_data_for_person), 3)

        gt_val = calib_val_data_for_person['gt'].to_numpy()
        gt_val = np.concatenate(gt_val, axis=0).reshape(len(calib_val_data_for_person), 2)

        calib_por_estimation_fit = poly_por.fit_transform(calib_por_estimation)
        after_calib_val_data = fit_transform_por.predict(calib_por_estimation_fit)
        gt_val = self.from_yaw_pitch_to_3D_np(gt_val)
        after_calib_val_data = self.from_yaw_pitch_to_3D_np(after_calib_val_data)

        calib_features_fit = poly_feature.fit_transform(calib_features)
        after_calib_feature_data = fit_transform_feature.predict(calib_features_fit)
        after_calib_val_data_feature = self.from_yaw_pitch_to_3D_np(after_calib_feature_data)

        #calib_por_estimation_fit = self.from_yaw_pitch_to_3D_np(calib_por_estimation)
        avg_error_after_calib_por, _ = self.calc_metrics_np(after_calib_val_data, gt_val)
        avg_error_before_calib, _ = self.calc_metrics_np(calib_por_estimation, gt_val)
        avg_error_after_calib_feature, _ = self.calc_metrics_np(after_calib_val_data_feature, gt_val)

        return avg_error_after_calib_por, avg_error_after_calib_feature, avg_error_before_calib
        #Apply the tansfomration on all the person's data

        #Compute error



    def calc_calib_knn(self, person_data):
        # Take the data for fitting from the first session (calibration). Take both dynamic and
        # Stationary labels. Use only stationary data to evaluate the fitting.
        data_for_fitting = person_data[person_data['session_number'] == 1]
        #data_for_fitting = data_for_fitting[data_for_fitting['is_stationary'] == 1]
        data_for_testing = person_data[person_data['session_number'] > 1]
        data_for_testing = data_for_testing[data_for_testing['is_stationary'] == 1]
        gt_fit = data_for_fitting['gt'].to_numpy()
        por_fit = data_for_fitting['por_estimation'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)
        por_fit = np.concatenate(por_fit, axis=0).reshape(len(por_fit), 3)
        poly_deg = 2

        gt_fit = self.from_3D_to_yaw_pitch_np(gt_fit)
        por_fit = self.from_3D_to_yaw_pitch_np(por_fit)

        knn_model = self.compute_fit_svr(por_fit, gt_fit, poly_deg)
        # else:
        #     fit_model, poly_model = self.compute_fit(por_fit, gt_fit, poly_deg)

        gt_test = data_for_testing['gt'].to_numpy()
        por_test = data_for_testing['por_estimation'].to_numpy()

        gt_test = np.concatenate(gt_test, axis=0).reshape(len(gt_test), 3)
        por_test = np.concatenate(por_test, axis=0).reshape(len(por_test), 3)


        por_test = self.from_3D_to_yaw_pitch_np(por_test)

        transformed_test_por = knn_model.predict(por_test)
        #compute poly feature for test:

        transformed_test_por = self.from_yaw_pitch_to_3D_np(transformed_test_por)

        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por, gt_test)
        return average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib

    def calc_calib_rbf(self, person_data):
        # Take the data for fitting from the first session (calibration). Take both dynamic and
        # Stationary labels. Use only stationary data to evaluate the fitting.
        data_for_fitting = person_data[person_data['session_number'] == 1]
        #data_for_fitting = data_for_fitting[data_for_fitting['is_stationary'] == 1]
        data_for_testing = person_data[person_data['session_number'] > 1]
        data_for_testing = data_for_testing[data_for_testing['is_stationary'] == 1]
        gt_fit = data_for_fitting['gt'].to_numpy()
        por_fit = data_for_fitting['por_estimation'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)
        por_fit = np.concatenate(por_fit, axis=0).reshape(len(por_fit), 3)
        poly_deg = 3

        gt_fit = self.from_3D_to_yaw_pitch_np(gt_fit)
        por_fit = self.from_3D_to_yaw_pitch_np(por_fit)

        rbf_x, rbf_y = self.compute_fit_svr(por_fit, gt_fit, poly_deg)
        # else:
        #     fit_model, poly_model = self.compute_fit(por_fit, gt_fit, poly_deg)

        gt_test = data_for_testing['gt'].to_numpy()
        por_test = data_for_testing['por_estimation'].to_numpy()

        gt_test = np.concatenate(gt_test, axis=0).reshape(len(gt_test), 3)
        por_test = np.concatenate(por_test, axis=0).reshape(len(por_test), 3)


        por_test = self.from_3D_to_yaw_pitch_np(por_test)

        fitted_x = rbf_x.predict(por_test)
        fitted_y = rbf_y.predict(por_test)
        fitted_x = np.expand_dims(fitted_x, axis=1)
        fitted_y = np.expand_dims(fitted_y, axis=1)
        transformed_test_por = np.concatenate((fitted_x, fitted_y), axis=1)
        #compute poly feature for test:

        transformed_test_por = self.from_yaw_pitch_to_3D_np(transformed_test_por)

        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por, gt_test)
        return average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib


    def compute_local_fit_grid(self, gt_fit, por_fit, poly_model):
        # compute a grid of transformations by weighting
        x = np.linspace(-0.5, 0.5, 30)
        y = np.linspace(-0.5, 0.5, 30)

        por_fit_poly = poly_model.fit_transform(por_fit)
        #estimated_gaze_poly = estimated_por_calc


        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        transformation_per_grid_point = []
        for point in grid_points:
            distances = np.sqrt(((gt_fit - point) ** 2).sum(axis=1))
            #distances_sm = self.softmax(distances, 0.5)
            distances_sm = softmax(distances)
            model = Ridge(0.001)
            model.fit(por_fit_poly, gt_fit, sample_weight=distances_sm)
            transformation_per_grid_point.append(model)

        return grid_points, transformation_per_grid_point

    def compute_local_fit(self, gt_fit, estimated_after_global_fit):
        #For each point in gt:
            #find K nearest neighbor.
            #Compute local fitting model
        k = 10
        knn_gt = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn_gt.fit(gt_fit)
        local_models = []
        transformed_points = []
        for p in estimated_after_global_fit:
            p = p.reshape(1, -1)
            distances, indices = knn_gt.kneighbors(p)
            local_gt = gt_fit[indices].reshape(k,2)
            local_por = estimated_after_global_fit[indices].reshape(k,2)
            #model = LinearRegression()
            model = Ridge(0.001)
            model.fit(local_por, local_gt)
            local_models.append(model)
            new_p = model.predict(p)
            transformed_points.append(new_p)
            #transformed_points.append(p)

        transformed_points = np.array(transformed_points)
        transformed_points = np.squeeze(transformed_points, axis=1)

        return local_models, knn_gt, transformed_points



    def compute_global_fit_3d(self, gt_fit, por_fit, max_polar_angle=60, number_of_calib_points=17):
        poly_deg = 2
        fit_model, poly_model = self.compute_fit_ridge(por_fit, gt_fit, poly_deg)
        por_fit_poly = poly_model.fit_transform(por_fit)
        por_fit_poly = normalize(por_fit_poly)
        #por_fit_poly = por_fit
        transformed_fit_por = fit_model.predict(por_fit_poly)
        transformed_fit_por = normalize(transformed_fit_por)

        return fit_model, poly_model, transformed_fit_por


    def compute_ensemble_fit(self, gt_fit, por_fit, max_polar_angle=60, number_of_calib_points=17):
        poly_models = []
        fit_models = []
        for poly_deg in range(1,5):
            fit_model, poly_model = self.compute_fit_ridge(por_fit, gt_fit, poly_deg, kernel='linear')
            por_fit_poly = poly_model.fit_transform(por_fit)
            transformed_fit_por = fit_model.predict(por_fit_poly)
            poly_models.append(poly_model)
            fit_models.append(fit_model)

        return poly_models, fit_models



    def compute_global_fit(self, gt_fit, por_fit, max_polar_angle=60, number_of_calib_points=17):
        poly_deg = 4
        fit_model, poly_model = self.compute_fit_ridge(por_fit, gt_fit, poly_deg)
        por_fit_poly = poly_model.fit_transform(por_fit)
        transformed_fit_por = fit_model.predict(por_fit_poly)

        return fit_model, poly_model, transformed_fit_por

    def apply_ensemble_calib(self, poly_models, fit_models,
                              por_test_2d):
        aggregate_por_test = []
        for poly_model, fit_model in zip(poly_models, fit_models):
            por_test_poly = poly_model.fit_transform(por_test_2d)
            t_por_test = fit_model.predict(por_test_poly)
            aggregate_por_test.append(t_por_test)

        aggregate_por_test = np.array(aggregate_por_test)
        avg = np.mean(aggregate_por_test, axis=0)
        return avg

    def apply_mutistep_calib(self,global_model, model_per_point, knn_model, poly_model, por_test_2d):
        #First apply the global model
        por_test_poly = poly_model.fit_transform(por_test_2d)
        por_test_after_global_t = global_model.predict(por_test_poly)
        return por_test_after_global_t

        #After global fit, find the nearest neighbot to each point in por_test_after_global_t to the gt and
        #take the local model from there
        transformed_points = []
        for p in por_test_after_global_t:
            p = p.reshape(1,-1)
            distances, indices = knn_model.kneighbors(p)
            weights = softmax(np.power(distances, -1)).transpose((1,0))
            ps = []
            for index in indices[0]:
                local_model = model_per_point[index]
                new_p = local_model.predict(p)
                ps.append(new_p)
            ps = np.array(ps)
            ps = np.squeeze(ps, axis=1)
            ps = ps * weights
            new_p = np.sum(ps, axis=0)
            transformed_points.append(new_p)

        transformed_points = np.array(transformed_points)
        return transformed_points

    def calc_3d_calib(self, person_data, max_polar_angle=60, number_of_calib_points=17):
        #First compute a fit using all points

        data_for_fitting = person_data[person_data['session_number'] == 1]
        data_for_fitting = data_for_fitting[data_for_fitting['is_stationary'] == 1]
        gt_fit = data_for_fitting['gt'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)
        por_fit = data_for_fitting['por_estimation'].to_numpy()
        por_fit = np.concatenate(por_fit, axis=0).reshape(len(por_fit), 3)

        global_model, poly_model, estimated_after_global_fit = self.compute_global_fit_3d(gt_fit, por_fit, max_polar_angle=60, number_of_calib_points=17)
        #model_per_point, knn_model_for_gt, transformed_points_2d = self.compute_local_fit(gt_fit_2d, estimated_after_global_fit)

        #transformed_fit_3d = self.from_yaw_pitch_to_3D_np(transformed_points_2d)
        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(
            estimated_after_global_fit, gt_fit)



        data_for_testing = person_data[person_data['session_number'] > 1]
        data_for_testing = data_for_testing[data_for_testing['is_stationary'] == 1]

        #Each point in gt_fit holds a local model.
        #For each point in por_test:
        #1. Apply the global transform.
        #2. Apply a local transform taken from the nearest point in gt_fit (or a weighted average transorm of the knn weighted by distance)
        por_test = data_for_testing['por_estimation'].to_numpy()
        por_test = np.concatenate(por_test, axis=0).reshape(len(por_test), 3)

        poly_por_test = poly_model.fit_transform(por_test)
        #poly_por_test = por_test
        poly_por_test = normalize(poly_por_test)
        transformed_test_por = global_model.predict(poly_por_test)
        transformed_test_por = normalize(transformed_test_por)
        # transformed_test_por = self.apply_mutistep_calib(global_model, model_per_point, knn_model_for_gt, poly_model,
        #                                                  por_test_2d)
        # transformed_test_por = self.from_yaw_pitch_to_3D_np(transformed_test_por)
        gt_test = data_for_testing['gt'].to_numpy()
        gt_test = np.concatenate(gt_test, axis=0).reshape(len(gt_test), 3)

        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por, gt_test)
        error_deg_after_calib = np.expand_dims(error_deg_after_calib, axis=1)
        return average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib, None

    def softmax(self, z, temperature=1.0):
        """
        Compute the softmax of vector z with a temperature parameter.

        Args:
        z (np.array): Input array.
        temperature (float): Temperature parameter (T > 0). Lower temperature makes the distribution more peaky.

        Returns:
        np.array: Softmax output.
        """
        e_z = np.exp((z - np.max(z)) / temperature)  # Subtract max for numerical stability
        return e_z / e_z.sum()

    def apply_grid_calib(self, poly_model, grid_points, transformations_per_grid_point, global_model,
                          por_test_2d):
        por_test_poly = poly_model.fit_transform(por_test_2d)
        por_test_after_global_t = global_model.predict(por_test_poly)
        transformed_points = []
        for i, p in enumerate(por_test_after_global_t):
            # compute distances from p to all grid points
            distances = np.sqrt(((grid_points - p) ** 2).sum(axis=1))
            min_index = np.argmin(distances)
            p_before_transformation = por_test_poly[i].reshape(1, -1)

            indexes = [min_index]
            ps = []
            for model_index in indexes:
                model = transformations_per_grid_point[model_index]
                new_p = model.predict(p_before_transformation)
                ps.append(new_p)

            new_p_global = por_test_after_global_t[i]
            ps.append(new_p_global)
            points_array = np.vstack(ps)
            average_point = np.mean(points_array, axis=0)
            transformed_points.append(average_point)

        transformed_points = np.array(transformed_points).squeeze()
        return transformed_points




    def calc_ensemble_calib(self, person_data, max_polar_angle=60, number_of_calib_points=17):
        #First compute a fit using all points

        data_for_fitting = person_data[person_data['session_number'] == 1]
        data_for_fitting = data_for_fitting[data_for_fitting['is_stationary'] == 1]
        gt_fit = data_for_fitting['gt'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)

        if number_of_calib_points < len(gt_fit):
            fps_indexes = self.farthest_point_sampling(gt_fit, number_of_calib_points)
            data_for_fitting = data_for_fitting.iloc[fps_indexes]

        por_fit = data_for_fitting['por_estimation'].to_numpy()
        por_fit = np.concatenate(por_fit, axis=0).reshape(len(por_fit), 3)

        gt_fit = data_for_fitting['gt'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)

        gt_fit_2d = self.from_3D_to_yaw_pitch_np(gt_fit)
        por_fit_2d = self.from_3D_to_yaw_pitch_np(por_fit)


        poly_models, fit_models  = self.compute_ensemble_fit(gt_fit_2d, por_fit_2d, max_polar_angle=60, number_of_calib_points=17)

        data_for_testing = person_data[person_data['session_number'] > 1]
        data_for_testing = data_for_testing[data_for_testing['is_stationary'] == 1]

        #Each point in gt_fit holds a local model.
        #For each point in por_test:
        #1. Apply the global transform.
        #2. Apply a local transform taken from the nearest point in gt_fit (or a weighted average transorm of the knn weighted by distance)
        por_test = data_for_testing['por_estimation'].to_numpy()
        por_test = np.concatenate(por_test, axis=0).reshape(len(por_test), 3)
        por_test_2d = self.from_3D_to_yaw_pitch_np(por_test)

        transformed_test_por = self.apply_ensemble_calib(poly_models, fit_models,
                                                     por_test_2d)
        # transformed_test_por = self.apply_mutistep_calib(global_model, model_per_point, knn_model_for_gt, poly_model,
        #                                                  por_test_2d)
        transformed_test_por = self.from_yaw_pitch_to_3D_np(transformed_test_por)
        gt_test = data_for_testing['gt'].to_numpy()
        gt_test = np.concatenate(gt_test, axis=0).reshape(len(gt_test), 3)

        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por, gt_test)
        error_deg_after_calib = np.expand_dims(error_deg_after_calib, axis=1)

        return average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib, None



    def calc_piecewise_calib(self, person_data, max_polar_angle=60, number_of_calib_points=17):
        #First compute a fit using all points

        data_for_fitting = person_data[person_data['session_number'] == 1]
        data_for_fitting = data_for_fitting[data_for_fitting['is_stationary'] == 1]
        gt_fit = data_for_fitting['gt'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)
        por_fit = data_for_fitting['por_estimation'].to_numpy()
        por_fit = np.concatenate(por_fit, axis=0).reshape(len(por_fit), 3)
        gt_fit_2d = self.from_3D_to_yaw_pitch_np(gt_fit)
        por_fit_2d = self.from_3D_to_yaw_pitch_np(por_fit)

        global_model, poly_model, estimated_after_global_fit = self.compute_global_fit(gt_fit_2d, por_fit_2d, max_polar_angle=60, number_of_calib_points=17)
        grid_points, transformations_per_grid_point = self.compute_local_fit_grid(gt_fit_2d, por_fit_2d, poly_model)
        # model_per_point, knn_model_for_gt, transformed_points_2d = self.compute_local_fit(gt_fit_2d,
        #                                                                                   estimated_after_global_fit)

        # transformed_fit_3d = self.from_yaw_pitch_to_3D_np(transformed_points_2d)
        # average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(
        #     transformed_fit_3d, gt_fit)



        data_for_testing = person_data[person_data['session_number'] > 1]
        data_for_testing = data_for_testing[data_for_testing['is_stationary'] == 1]

        #Each point in gt_fit holds a local model.
        #For each point in por_test:
        #1. Apply the global transform.
        #2. Apply a local transform taken from the nearest point in gt_fit (or a weighted average transorm of the knn weighted by distance)
        por_test = data_for_testing['por_estimation'].to_numpy()
        por_test = np.concatenate(por_test, axis=0).reshape(len(por_test), 3)
        por_test_2d = self.from_3D_to_yaw_pitch_np(por_test)

        transformed_test_por = self.apply_grid_calib(poly_model, grid_points, transformations_per_grid_point, global_model,
                                                     por_test_2d)
        # transformed_test_por = self.apply_mutistep_calib(global_model, model_per_point, knn_model_for_gt, poly_model,
        #                                                  por_test_2d)
        transformed_test_por = self.from_yaw_pitch_to_3D_np(transformed_test_por)
        gt_test = data_for_testing['gt'].to_numpy()
        gt_test = np.concatenate(gt_test, axis=0).reshape(len(gt_test), 3)

        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por, gt_test)
        error_deg_after_calib = np.expand_dims(error_deg_after_calib, axis=1)

        return average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib, None



        pass


    def calc_calib(self, person_data, max_polar_angle=60, number_of_calib_points=17):
        # Take the data for fitting from the first session (calibration). Take both dynamic and
        # Stationary labels. Use only stationary data to evaluate the fitting.
        data_for_fitting = person_data[person_data['session_number'] == 1]
        data_for_fitting = data_for_fitting[data_for_fitting['is_stationary'] == 1]

        #gil - calibration testing
        #data_for_fitting = data_for_fitting.sample(n=10, random_state=4)
        #data_for_fitting = data_for_fitting[data_for_fitting['is_stationary'] == 1]
        data_for_testing = person_data[person_data['session_number'] > 1]
        data_for_testing = data_for_testing[data_for_testing['is_stationary'] == 1]

        gt_fit = data_for_fitting['gt'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)

        fps_indexes = self.farthest_point_sampling(gt_fit, number_of_calib_points)
        data_for_fitting = data_for_fitting.iloc[fps_indexes]

        gt_fit = data_for_fitting['gt'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)

        z = np.zeros_like(gt_fit)
        z[:,2] = 1.0
        _, _, polar, _ = self.calc_metrics_np(gt_fit, z)
        data_for_fitting['polar'] = polar
        data_for_fitting = data_for_fitting[data_for_fitting['polar'] < max_polar_angle]



    #    data_for_fitting = data_for_fitting.sort_values('polar')
    #    data_for_fitting = data_for_fitting.iloc[0:15]
        gt_fit = data_for_fitting['gt'].to_numpy()
        gt_fit = np.concatenate(gt_fit, axis=0).reshape(len(gt_fit), 3)

        gt_for_error_analysis = data_for_testing['gt'].to_numpy()

        por_fit = data_for_fitting['por_estimation'].to_numpy()

        por_fit = np.concatenate(por_fit, axis=0).reshape(len(por_fit), 3)

        poly_deg = 3

        gt_fit = self.from_3D_to_yaw_pitch_np(gt_fit)
        por_fit = self.from_3D_to_yaw_pitch_np(por_fit)

        tree = False
        if tree:
            regressor = DecisionTreeRegressor(random_state=42, min_samples_split=5)
            regressor.fit(por_fit, gt_fit)
        else:
            fit_model, poly_model = self.compute_fit_ridge(por_fit, gt_fit, poly_deg)
            #fit_model, poly_model = self.compute_fit(por_fit, gt_fit, poly_deg)

        gt_test = data_for_testing['gt'].to_numpy()
        gt_test = np.concatenate(gt_test, axis=0).reshape(len(gt_test), 3)
        z = np.zeros_like(gt_test)
        z[:,2] = 1.0

        _, _, polar, _ = self.calc_metrics_np(gt_test, z)
        data_for_testing['polar'] = polar
        data_for_testing = data_for_testing[data_for_testing['polar'] < max_polar_angle]

        gt_for_error_analysis = data_for_testing['gt'].to_numpy()
        gt_for_error_analysis = np.concatenate(gt_for_error_analysis, axis=0).reshape(len(gt_for_error_analysis), 3)

        gt_for_error_analysis = self.from_3D_to_yaw_pitch_np(gt_for_error_analysis)
        gt_test = data_for_testing['gt'].to_numpy()
        por_test = data_for_testing['por_estimation'].to_numpy()

        gt_test = np.concatenate(gt_test, axis=0).reshape(len(gt_test), 3)
        por_test = np.concatenate(por_test, axis=0).reshape(len(por_test), 3)


        por_test = self.from_3D_to_yaw_pitch_np(por_test)

        #compute poly feature for test:
        if tree:
            transformed_test_por = regressor.predict(por_test)
        else:
            estimated_por_poly = poly_model.fit_transform(por_test)
            transformed_test_por = fit_model.predict(estimated_por_poly)

        transformed_test_por = self.from_yaw_pitch_to_3D_np(transformed_test_por)

        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por, gt_test)
        error_deg_after_calib = np.expand_dims(error_deg_after_calib, axis=1)
        error_per_label = np.concatenate((gt_for_error_analysis, error_deg_after_calib), axis=1)


        return average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib, error_per_label




    def create_error_map(self, data, desc='All'):

        # Separate the points into x, y, and z
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        # Create grid for the heatmap
        grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

        # Interpolate the z values onto the grid
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='nearest')

        vmin = 0
        vmax = 10

        plt.figure()
        # Plot the heatmap
        plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Error')
        plt.scatter(x, y, c=z, edgecolors='w', s=20, vmin=vmin, vmax=vmax)  # Optional: plot original points
        plt.xlabel('X Location')
        plt.ylabel('Y Location')
        plt.title('Heatmap of Errors For ' + desc)
        plt.show()

    def on_validation_epoch_end(self):

        print('on_validation_end called')

        #self.validation_table = pd.DataFrame(self.validation_table_data)

        #self.validation_table.to_pickle('/home/gilsh/Gaze/sgaze/dataframe_all.pkl')
        #return
        self.validation_table = pd.read_pickle('/home/gilsh/Gaze/sgaze/dataframe_all.pkl')
        #self.validation_table = pd.read_pickle('/home/gilsh/dataframe_all.pkl')
        #data = torch.load('/home/gilsh/dataframe.pkl', map_location=torch.device('cpu'))
        #self.validation_table = pd.DataFrame(data)
        group_by_id = self.validation_table.groupby('subject_id')
        after_calib_avg_err_list = []
        after_calib_med_err_list = []
        #Error data contains the 2d gaze label and the associated error (3 number for each gaze point)
        error_data = []
        for group_id_name, group_id_df in group_by_id:
            gt = group_id_df['gt'].to_numpy()
            por = group_id_df['por_estimation'].to_numpy()
            gt = np.concatenate(gt, axis=0).reshape(len(gt), 3)
            por = np.concatenate(por, axis=0).reshape(len(por), 3)

            average_error_before_calib, avg_dist_before_calib, error_deg_before_calib, median_error_before = self.calc_metrics_np(por, gt)

            average_error_after_calib, \
            avg_dist_after_calib,\
            error_deg_after_calib,\
            median_error_after,\
            error_per_label = self.calc_ensemble_calib(group_id_df, max_polar_angle=60, number_of_calib_points=8)

            #self.create_error_map(error_per_label, group_id_name)
            #self.validation_table['error_deg'] = error_deg
            self.log('val_avg_error_id_{}_before_calib'.format(group_id_name), average_error_before_calib, on_step=False, on_epoch=True,
                 sync_dist=False)
            self.log('val_avg_error_id_{}_after_calib'.format(group_id_name), average_error_after_calib, on_step=False, on_epoch=True,
                 sync_dist=False)
            self.log('val_med_error_id_{}_before_calib'.format(group_id_name), median_error_before, on_step=False, on_epoch=True,
                 sync_dist=False)
            self.log('val_med_error_id_{}_after_calib'.format(group_id_name), median_error_after, on_step=False, on_epoch=True,
                 sync_dist=False)

            after_calib_avg_err_list.append(average_error_after_calib)
            after_calib_med_err_list.append(median_error_after)

        average_error_after_calib = sum(after_calib_avg_err_list) / len(after_calib_avg_err_list)
        med_error_after_calib = sum(after_calib_med_err_list) / len(after_calib_med_err_list)
        std_error = np.std(np.array(after_calib_avg_err_list))

        self.log('val_error_after_calib_std', std_error, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('val_error_after_calib', average_error_after_calib, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('val_med_error_after_calib', med_error_after_calib, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

            # if self.validation_it % 20 == 0:
            #     gt = gt[:,0:2]
            #     por = por[:,0:2]
            #     for i in range(len(gt)):
            #         plt.plot([gt[i, 0], por[i, 0]], [gt[i, 1], por[i, 1]], color='green')
            #
            #     plt.scatter(gt[:, 0], gt[:, 1], color='red')
            #     plt.scatter(por[:, 0], por[:, 1], color='blue')
            #     # for i in range(len(gt)):
            #     #     plt.arrow(por[i][0], por[i][1], gt[i][0] - por[i][0], por[i][1] - gt[i][1], head_width=0.01,
            #     #               head_length=0.01, fc='red', ec='red')
            #
            #     wandb.log({"Validation {} pairs ".format(self.validation_it): plt})
            #self.validation_it = self.validation_it + 1
        return



        #self.validation_table = self.validation_table.append(rows, ignore_index=True)
        # grouped_by_person = self.validation_table.groupby(['subject_id'])
        # val_results = {}
        # sum_before_calib = 0
        # sum_after_calib_por = 0
        # sum_after_calib_feature = 0
        # for person_id, person_data in grouped_by_person:
        #     res_after_calib_por, res_after_calib_feature, res_before_calib = self.calc_validation_loss(person_data)
        #     #val_results[person_id] = res
        #     sum_before_calib = sum_before_calib + res_before_calib
        #     sum_after_calib_por = sum_after_calib_por + res_after_calib_por
        #     sum_after_calib_feature = sum_after_calib_feature + res_after_calib_feature
        #     self.log('val_error_person_{}_por'.format(person_id), res_after_calib_por, on_step=False, on_epoch=True,
        #          sync_dist=False)
        #     self.log('val_error_person_{}_feature'.format(person_id), res_after_calib_feature, on_step=False, on_epoch=True,
        #          sync_dist=False)
        #
        #
        # avg_val_before_calib_error = sum_before_calib / len(grouped_by_person)
        # avg_val_after_calib_por_error = sum_after_calib_por / len(grouped_by_person)
        # avg_val_after_calib_feature_error = sum_after_calib_feature / len(grouped_by_person)
        # #self.logger.log_metric('avg_val_error', avg_val_error)
        # self.log('avg_val_error_before_calib', avg_val_before_calib_error, on_step=False, on_epoch=True,
        #          sync_dist=False)
        # self.log('avg_val_error_after_calib_por', avg_val_after_calib_por_error, on_step=False, on_epoch=True,
        #          sync_dist=False)
        # self.log('avg_val_error_after_calib_feature', avg_val_after_calib_feature_error, on_step=False, on_epoch=True,
        #          sync_dist=False)



    def on_validation_epoch_start(self):
        #self.to('cpu')
        #self.validation_table = pd.DataFrame(columns=['subject_id', 'por_estimation', 'gt'])
        self.validation_table_data = []


    def validation_step(self, batch, batch_idx):
        #self.train()
        # self.freeze_batch_norm()
        # for param in self.parameters():
        #     param.requires_grad = False

        gaze_labels, left_images, right_images, label = batch
        w = 2
        error_weight = torch.tensor([(1 + w * x) for x in label['is_stationary']],device=self.device)
        subject_ids = label['id']
        session_numbers = label['session_number']
        is_stationary = label['is_stationary']
        #gaze_labels, left_images, right_images, person_id, L_file, R_file, error_weight = batch
        #As we are not doing back-prop but least squares we work in eval() mode only
        if self.params.binocular_mode:
            estimated_pors, features, left_features, right_features,_,_ = self.forward_bino(left_images, right_images)
        else:
            estimated_pors, features = self.forward_mono(left_images)

        avg_error, avg_cos_dist, median_err = self.compute_loss_no_calib(estimated_pors, gaze_labels, error_weight)
        # if self.params.train_inline_calib:
        #     avg_error, avg_cos_dist = self.compute_loss_diff(estimated_pors, gaze_labels, error_weight)
        # else:
        #     avg_error, avg_cos_dist = self.compute_loss_no_calib(estimated_pors, gaze_labels, error_weight)




        #subject_ids = list(person_id)
        estimated_pors = estimated_pors.cpu().float().numpy().tolist()
        features = features.cpu().float().numpy().tolist()
        gaze_labels = gaze_labels.cpu().float().numpy()

        for subject_id, estimated_por, gaze_label, feature, session_number, is_station in zip(subject_ids, estimated_pors, gaze_labels, features, session_numbers, is_stationary):
            new_row = {
                'is_stationary': is_station.cpu(),
                'session_number': session_number.cpu(),
                'subject_id': subject_id,
                'por_estimation': estimated_por,
                'gt': gaze_label,
                'features': feature
            }
            self.validation_table_data.append(new_row)

        return avg_error, median_err

    def validation_epoch_end(self, outputs):
        average_error_deg_per_sample = []
        median_error_deg_per_sample = []
        for item in outputs:
            average_error_deg_per_sample.append(item[0])
            median_error_deg_per_sample.append(item[1])

        average_error_deg_per_sample = torch.tensor(average_error_deg_per_sample)
        median_error_deg_per_sample = torch.tensor(median_error_deg_per_sample)

        total_avg_deg_error = torch.mean(average_error_deg_per_sample)
        total_med_deg_error = torch.mean(median_error_deg_per_sample)

        self.log('validation_avg_deg_error', total_avg_deg_error, on_epoch=True, sync_dist=True)
        self.log('validation_med_deg_error', total_med_deg_error, on_epoch=True, sync_dist=True)
        #avg = sum(self.error_list) / len(self.error_list)
        # a = np.array(self.error_list)
        # b = average_error_deg_per_sample.cpu().numpy()
        # np.savetxt('/home/gilsh/temp/train_error.txt', a)
        # np.savetxt('/home/gilsh/temp/val_error.txt', b)
        #self.log('training_avg_deg_error', avg, on_epoch=True, sync_dist=True)



    def configure_optimizers(self):
        #opt = torch.optim.SGD(self.parameters(),lr=0.0)
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams['hparams'].lr,
                                weight_decay=self.hparams['hparams'].weight_decay)

        scheduler = lr_scheduler.MultiStepLR(opt,
                                             milestones=self.hparams['hparams'].lr_milestones,
                                             gamma=self.hparams['hparams'].lr_gamma)

        return [opt], [scheduler]

    def split_parameters(self, module):
        params_decay = []
        params_no_decay = []
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                params_no_decay.extend([*m.parameters()])
            elif len(list(m.children())) == 0:
                params_decay.extend([*m.parameters()])
        assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
        return params_decay, params_no_decay