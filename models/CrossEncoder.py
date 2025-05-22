import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
import pytorch_lightning
import torch
import numpy as np
from models.singleEye3DGazeTimm import singleEye3DGazeTimmModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import MDS
from PIL import Image, ImageOps
import random
import torchvision
import pandas as pd
from models.mbnv2 import MobileNet_v2
from models.VAE import VAE
import matplotlib.pyplot as plt
import torch.nn.init as init
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
#plt.ioff()
import joblib
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


def diffusion_map(X, sigma=1, dim=2):
    # Compute the affinity matrix
    distances = squareform(pdist(X, 'euclidean'))
    K = np.exp(-distances ** 2 / sigma ** 2)

    # Construct the Markov matrix
    row_sums = K.sum(axis=1)
    P = K / row_sums[:, np.newaxis]

    # Eigen decomposition
    eigenvalues, eigenvectors = eigh(P)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Return the projection onto the first 'dim' non-trivial eigenvectors
    return eigenvectors[:, 1:dim + 1]


class CrossEncoder(LightningModule):

    def __init__(self, hparams, channels=1):
        super(CrossEncoder, self).__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()  # sets self.hparams
        self.params = self.hparams['hparams']
        args = self.hparams['hparams']
        self.cvm_vae_negative_margin = args.cvm_vae_negative_margin
        self.gamma = 1.0
        bottleneckLayerDetails = [
            (1, 4, 1, 2),
            (6, 8, 1, 2),
            (6, 16, 1, 2),
            (6, 32, 1, 2),
            (6, 48, 1, 2),
            (6, 64, 1, 2),
        ]
        self.gaze_dim = 30
        self.res_loss_weight = 0.3
        self.fc_mean = torch.nn.Linear(self.params.backbone_feature_size, self.params.backbone_feature_size)
        self.fc_log_var = torch.nn.Linear(self.params.backbone_feature_size, self.params.backbone_feature_size)
        self.projection = torch.nn.Linear(self.params.backbone_feature_size, 2)
        self.error_list = []
        self.encoder = MobileNet_v2(bottleneckLayerDetails, width_multiplier=self.params.width_multiplier, in_fts=1)
        self.vae = VAE(self.params)
        self.backbone_feaure_size = self.params.backbone_feature_size

        self.channels = 1
        self.decoder = Decoder(latent_dim=128)
        self.alpha = torch.tensor(args.alpha)
        self.margin = torch.tensor(args.triplet_loss_margin)
        self.zero = torch.tensor(0.0)
        #self.load_from_checkpoint(hparams.ckpt_path, strict=False)
        if hparams.ckpt_path != '':
            self.load_state_dict(torch.load(hparams.ckpt_path, map_location=self.device)['state_dict'], strict=True)
        elif hparams.vae_path != '':
            self.vae.load_state_dict(torch.load(hparams.vae_path, map_location=self.device)['state_dict'], strict=True)
            self.encoder.load_state_dict(self.vae.encoder.state_dict(), strict=True)
            #copy the weights from vae to contrastive encoder
        else:
            print('apply init weights')
            self.encoder.apply(init_weights)

        #Freeze vae
        for param in self.vae.parameters():
            param.requires_grad = False

        # for param in self.parameters():
        #     param.requires_grad = False


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std



    def training_validation_step_ssl(self, x):

        loss, kld = self.cross_encoder_loss(x)

        self.log('ce_loss_val', loss, on_step=False, on_epoch=True,
                 sync_dist=True)
        self.log('kld_val', kld, on_step=False, on_epoch=True,
                 sync_dist=True)


    def training_validation_step_supervised(self, x):
        gaze_labels, left_images, right_images, label = x

        z_left, mu_left, log_var_left = self.variational_embedding(left_images)
        z_right, mu_right, log_var_right = self.variational_embedding(right_images)

        z_dim = z_left.shape[1]
        #gaze_dim = 5
        eye_id_dim = z_dim - self.gaze_dim
        # gaze_left, eye_id_left = torch.split(z_left, [gaze_dim, eye_id_dim], dim=1)
        # gaze_right, eye_id_right = torch.split(z_right, [gaze_dim, eye_id_dim], dim=1)

        gaze_left, eye_id_left = torch.split(mu_left, [self.gaze_dim, eye_id_dim], dim=1)
        gaze_right, eye_id_right = torch.split(mu_right, [self.gaze_dim, eye_id_dim], dim=1)

        gaze_lefts = gaze_left.cpu().float().numpy()
        gaze_rights = gaze_right.cpu().float().numpy()
        gaze_labels = gaze_labels.cpu().float().numpy()
        person_ids = label['person_id']
        session_numbers = label['session_number'].cpu().int().numpy()

        for person_id, gaze_label,gaze_left, gaze_right, session_number in\
                zip(person_ids, gaze_labels, gaze_lefts, gaze_rights, session_numbers ):
            new_row = {
                'mu_left': gaze_left,
                'mu_right': gaze_right,
                'session_number': session_number,
                'person_id': person_id,
                'gt': gaze_label
            }
            self.validation_table_data.append(new_row)

        pass

    def draw_scatter_plot(self, pca_features, gaze_labels):
        # Creating the plot with larger dimensions
        fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size if needed
        ax.set_xlim(pca_features[:, 0].min(), pca_features[:, 0].max())
        ax.set_ylim(pca_features[:, 1].min(), pca_features[:, 1].max())

        # Adding images to the scatter plot with a very large zoom factor
        for i in range(len(gaze_labels)):

            #image = img_to_array(image, scale_factor=0.2)
            x, y = pca_features[i, 0], pca_features[i, 1]
            label_text = f"({gaze_labels[i,0]:.2f}, {gaze_labels[i,1]:.2f}, {gaze_labels[i,2]:.2f})"
            ax.text(x, y, label_text, fontsize=9, ha='right', va='bottom')

        ax.set_xlabel('Contrastive embeddings x')
        ax.set_ylabel('Contrastive embedding y')
        plt.title('Contrastive embeddings with ground truth labels')
        plt.show()
        pass

    def apply_pca_and_apply_on_test(self, features_calib, features_test, gaze_labels, side='left'):

        features_calib = np.concatenate(features_calib, axis=0).reshape(len(features_calib), len(features_calib[0]))
        features_test = np.concatenate(features_test, axis=0).reshape(len(features_test), len(features_test[0]))

        #scaler = StandardScaler()
        #features_calib_scaled = scaler.fit_transform(features_calib)
        pca = PCA(n_components=10)
        #pca = joblib.load('/home/gilsh/temp/pca_model_' + side + '_10d.pkl')

        # test_features_pca = pca.fit_transform(features_test)
        # calib_features_pca = pca.transform(features_calib)
        calib_features_pca = pca.fit_transform(features_calib)
        test_features_pca = pca.transform(features_test)
        mds = MDS(n_components=2, random_state=42)

        # Fit the MDS model and transform the data
        calib_features_pca_mds = mds.fit_transform(features_calib)
        if side == 'left':
            self.draw_scatter_plot(calib_features_pca_mds, gaze_labels)
        else:
            self.draw_scatter_plot(calib_features_pca_mds, gaze_labels)
        # tsne = TSNE(n_components=2, metric='cosine')
        # calib_tsne_features = tsne.fit_transform(features_calib)
        # calib_dmp_features = diffusion_map(features_test_pca, sigma=2, dim=2)
        # self.draw_scatter_plot(calib_tsne_features, gaze_labels, left_images)
        # self.draw_scatter_plot(calib_pca_features, gaze_labels, left_images)


        return calib_features_pca, test_features_pca

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

    def apply_ensemble_calib(self, poly_models, fit_models, por_test_2d ):
        aggregate_por_test = []
        for poly_model, fit_model in zip(poly_models, fit_models):
            por_test_poly = poly_model.fit_transform(por_test_2d)
            t_por_test = fit_model.predict(por_test_poly)
            aggregate_por_test.append(t_por_test)

        aggregate_por_test = np.array(aggregate_por_test)
        avg = np.mean(aggregate_por_test, axis=0)
        return avg

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


    def extract_ssl_data(self, group_ssl_df, session_cutoff):
        data_for_calib_all = group_ssl_df[group_ssl_df['session_number'] <= session_cutoff]

        left_features_calib_all = data_for_calib_all['mu_left'].values
        left_features_calib_all = np.concatenate(left_features_calib_all, axis=0).reshape(len(left_features_calib_all),
                                                                                  len(left_features_calib_all[0]))
        right_features_calib_all = data_for_calib_all['mu_right'].values
        right_features_calib_all = np.concatenate(right_features_calib_all, axis=0).reshape(len(right_features_calib_all),
                                                                                  len(right_features_calib_all[0]))
        gaze_labels_calib_all = data_for_calib_all['gt'].values
        gaze_labels_calib_all = np.concatenate(data_for_calib_all, axis=0).reshape(len(data_for_calib_all), 3)

    def draw_scatter_plot(self, set_A, set_B, set_C):

        """
        set_A: List of tuples (x, y) representing points in set A
        set_B: List of tuples (x, y) representing points in set B
        set_C: List of tuples (x, y) representing points in set C (used as labels)
        """

        # Check that all sets have the same number of points
        assert len(set_A) == len(set_B) == len(set_C), "All sets must have the same number of points."

        plt.figure(figsize=(8, 8))

        for a, b, c in zip(set_A, set_B, set_C):
            # Plot point from A
            plt.scatter(a[0], a[1], color='blue',
                        label='Left Embeddings' if 'Left Embeddings' not in plt.gca().get_legend_handles_labels()[1] else "")
            # Plot point from B
            plt.scatter(b[0], b[1], color='green',
                        label='Right Embeddings' if 'Right Embeddings' not in plt.gca().get_legend_handles_labels()[1] else "")
            # Draw line between a and b
            plt.plot([a[0], b[0]], [a[1], b[1]], color='black', linestyle='--')
            # Calculate midpoint between a and b
            #midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            # Use the coordinates of point c as the label
            label = f"({c[0]:.2f}, {c[1]:.2f})"
            # Add text label at the midpoint
            plt.text(a[0], a[1], label, fontsize=10, ha='center', va='center', color='red')
            plt.text(b[0], b[1], label, fontsize=10, ha='center', va='center', color='red')

        # Set axis labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('CVM Left and Right Embeddings')

        # Add a legend
        plt.legend()

        # Set equal scaling for x and y axis
        plt.gca().set_aspect('equal', adjustable='box')

        plt.grid(True)
        plt.show()

    def compute_ensemble_fit_merge(self, gaze_labels_calib_fit_2d,
                                        left_features_calib_fit_2d,
                                        right_features_calib_fit_2d,
                                        left_features_test,
                                        right_features_test
                                        ):

        por_test_2d = np.concatenate((left_features_test, right_features_test), axis=1)
        por_calib_2d = np.concatenate((left_features_calib_fit_2d, right_features_calib_fit_2d), axis=1)
        poly_model, fit_models = self.compute_ensemble_fit(gaze_labels_calib_fit_2d, por_calib_2d,
                                                           start_deg=1, end_deg=4)

        #left test


        transformed_test_por = self.apply_ensemble_calib(poly_model, fit_models,
                                                     por_test_2d )

        transformed_test_por_3d = self.from_yaw_pitch_to_3D_np(transformed_test_por)
        return transformed_test_por_3d

    def compute_ensemble_fit_separate(self, gaze_labels_calib_fit_2d,
                                        left_features_calib_fit_2d,
                                        right_features_calib_fit_2d,
                                        left_features_test,
                                        right_features_test
                                        ):
        poly_model_l, fit_models_l = self.compute_ensemble_fit(gaze_labels_calib_fit_2d, left_features_calib_fit_2d, start_deg=1, end_deg=2)
        poly_model_r, fit_models_r = self.compute_ensemble_fit(gaze_labels_calib_fit_2d, right_features_calib_fit_2d, start_deg=1, end_deg=2)

        #left test
        transformed_test_por_l = self.apply_ensemble_calib(poly_model_l, fit_models_l, left_features_test)
        transformed_test_por_r = self.apply_ensemble_calib(poly_model_r, fit_models_r, right_features_test)
        transformed_test_por = (transformed_test_por_l + transformed_test_por_r) / 2.0
        transformed_test_por_3d = self.from_yaw_pitch_to_3D_np(transformed_test_por)
        return transformed_test_por_3d



    def calc_ensemble_calib(self, data_for_calib_for_fit, data_for_test):


        #data_for_calib_for_fit = group_calib_df[group_calib_df['session_number'] <= session_cutoff]

        left_features_calib_fit_2d = data_for_calib_for_fit['mu_left'].values
        left_features_calib_fit_2d = np.concatenate(left_features_calib_fit_2d, axis=0).reshape(len(left_features_calib_fit_2d),
                                                                                  len(left_features_calib_fit_2d[0]))
        right_features_calib_fit_2d = data_for_calib_for_fit['mu_right'].values
        right_features_calib_fit_2d = np.concatenate(right_features_calib_fit_2d, axis=0).reshape(len(right_features_calib_fit_2d),
                                                                                            len(right_features_calib_fit_2d[
                                                                                                    0]))
        gaze_labels_calib_fit_3d = data_for_calib_for_fit['gt'].values
        gaze_labels_calib_fit_3d = np.concatenate(gaze_labels_calib_fit_3d, axis=0).reshape(len(gaze_labels_calib_fit_3d), 3)

        #data_for_test = group_calib_df[group_calib_df['session_number'] > session_cutoff]

        left_features_test_2d = data_for_test['mu_left'].values
        right_features_test_2d = data_for_test['mu_right'].values
        left_features_test = np.concatenate(left_features_test_2d, axis=0).reshape(len(left_features_test_2d),
                                                                                  len(left_features_test_2d[0]))

        right_features_test = np.concatenate(right_features_test_2d, axis=0).reshape(len(right_features_test_2d),
                                                                                  len(right_features_test_2d[0]))

        gaze_labels_test = data_for_test['gt'].values
        gaze_labels_test = np.concatenate(gaze_labels_test, axis=0).reshape(len(gaze_labels_test), 3)

        # left_features_calib_pca, left_feautres_test_pca = self.apply_pca_and_apply_on_test(left_features_calib, left_features_test, decoded_left, decoded_right
        #                                                                                , gaze_labels_fit, side='left')
        # right_features_calib_pca, right_features_test_pca = self.apply_pca_and_apply_on_test(right_features_calib, right_features_test, decoded_left, decoded_right,
        #                                                                                  gaze_labels_fit, side='right')
        gaze_labels_test_2d = self.from_3D_to_yaw_pitch_np(gaze_labels_test)


        gaze_labels_calib_fit_2d = self.from_3D_to_yaw_pitch_np(gaze_labels_calib_fit_3d)

        #self.draw_scatter_plot(left_features_calib_fit_2d, right_features_calib_fit_2d, gaze_labels_calib_fit_2d)
        separate_L_R = False

        if separate_L_R:
            transformed_test_por_3d = self.compute_ensemble_fit_separate(gaze_labels_calib_fit_2d,
                                                                         left_features_calib_fit_2d,
                                                                         right_features_calib_fit_2d,
                                                                         left_features_test,
                                                                         right_features_test)

        else:
            transformed_test_por_3d = self.compute_ensemble_fit_merge(gaze_labels_calib_fit_2d,
                                                                         left_features_calib_fit_2d,
                                                                         right_features_calib_fit_2d,
                                                                         left_features_test,
                                                                         right_features_test)


        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por_3d, gaze_labels_test)
        error_deg_after_calib = np.expand_dims(error_deg_after_calib, axis=1)

        return average_error_after_calib, average_error_after_calib, error_deg_after_calib

        pass



        #compute pca for right features
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

    def compute_ensemble_fit(self, gt_fit_2d, por_fit_2d, start_deg=1,  end_deg=3):
        poly_models = []
        fit_models = []

        #concatenate por_fit_left_2d and por_fit_right_2d along dim=1
        #por_fit = np.concatenate((por_fit_left_2d, por_fit_right_2d), axis=1)
        for poly_deg in range(start_deg, end_deg):
            fit_model, poly_model = self.compute_fit_ridge(por_fit_2d, gt_fit_2d, poly_deg, kernel='linear')
            por_fit_poly = poly_model.fit_transform(por_fit_2d)
            transformed_fit_por = fit_model.predict(por_fit_poly)
            poly_models.append(poly_model)
            fit_models.append(fit_model)

        return poly_models, fit_models

    def on_validation_epoch_end(self):

        print('on_validation_end called')

        self.validation_table = pd.DataFrame(self.validation_table_data)
        #self.validation_table_ssl = pd.DataFrame(self.validation_table_ssl)
        self.validation_table.to_pickle('/home/gilsh/temp/dataframe_ce_vanilla_res_loss.pkl')
        #return
        #self.validation_table = pd.read_pickle('/home/gilsh/temp/dataframe_cvm_multi_neg.pkl')
        #self.validation_table = pd.read_pickle('/home/gilsh/temp/dataframe_ce_vanilla.pkl')
        #self.validation_table = pd.read_pickle('/home/gilsh/temp/dataframe_ce_vanilla_mu.pkl')
        #self.validation_table = pd.read_pickle('/home/gilsh/temp/dataframe_ce_vanilla_mu_bf.pkl')
        group_by_id = self.validation_table.groupby('person_id')
        Ks = [17, 30, 40, 50, 60]
        num_iterations = 10
        avg_test_size = []
        calibration_data = pd.DataFrame(columns=['K', 'id', 'iter', 'error', 'test_size'])
        for K in Ks:
            for group_id_name, group_id_df in group_by_id:
                #compute PCA for left and right features
                # Sample K rows from the DataFrame
                for k in range(num_iterations):
                    data_for_fit = group_id_df.sample(n=K, random_state=42)
                    data_for_test = group_id_df.drop(data_for_fit.index)
                    average_error_after_calib,\
                    median_error_after,\
                    error_per_label = self.calc_ensemble_calib(data_for_fit, data_for_test)
                    row = {'K': K, 'id': group_id_name, 'iter': k, 'error': average_error_after_calib,
                           'test_size': len(data_for_test)}
                    calibration_data.loc[len(calibration_data)] = row

                # after_calib_avg_err_list_per_id = sum(after_calib_avg_err_list_per_id) / len(after_calib_avg_err_list_per_id)
                # self.log('val_avg_error_id_{}_after_calib'.format(group_id_name), after_calib_avg_err_list_per_id, on_step=False, on_epoch=True,
                #      sync_dist=False)

        for K in Ks:
            calibration_data_per_k = calibration_data[calibration_data['K']==K]
            print( 'K = {}, test_size = {}, acc = {}, std = {}'.format(K, calibration_data_per_k['test_size'].mean(),
                                                                                calibration_data_per_k['error'].mean(),
                                                                                calibration_data_per_k['error'].std()))

        average_error_after_calib = calibration_data['error'].mean()
        std_error =  calibration_data['error'].std()
        test_size = calibration_data['test_size'].mean()
        avg_K = calibration_data['K'].mean()
        self.log('val_error_after_calib_std', std_error, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('val_error_after_calib', average_error_after_calib, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('fitting_size', avg_K, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        self.log('avg_testing_size', test_size, on_step=False,
                     on_epoch=True,
                     sync_dist=False)

        return

    def on_validation_epoch_start(self):
        #self.to('cpu')
        #self.validation_table = pd.DataFrame(columns=['subject_id', 'por_estimation', 'gt'])
        self.validation_table_data = []
        self.validation_table_ssl = []

    def validation_step(self, x, batch_idx, dataloader_idx=0):

        #return self.validation(x)
        if dataloader_idx == 0:
            #supervised
            return self.training_validation_step_supervised(x)
        else:
            #ssl
            return self.training_validation_step_ssl(x)


    def variational_embedding(self, images):
        encoded = self.encoder(images)
        mu_left = self.fc_mean(encoded.squeeze(2).squeeze(2))
        log_var_left = self.fc_log_var(encoded.squeeze(2).squeeze(2))
        z = self.reparameterize(mu_left, log_var_left)

        return z, mu_left, log_var_left

    def generate_derangement(self, n):
        while True:
            # Generate a permutation of in dices
            perm = torch.randperm(n)
            # Check if any element is in its original position
            if not any(perm == torch.arange(n)):
                return perm



    def loss_kld(self, mu, log_var):
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return KLD

    def loss_same_eye_different_gaze(self, anchor_gaze_negative_eye_id,
                                      negative_gaze_anchor_eye_id,
                                      anchor_images,
                                      negative_images):

        anchor_decode = self.decoder(anchor_gaze_negative_eye_id)
        negative_decode = self.decoder(negative_gaze_anchor_eye_id)
        anchor_mse = torch.nn.functional.mse_loss(anchor_decode, anchor_images, reduction='mean')
        negative_mse = torch.nn.functional.mse_loss(negative_decode, negative_images, reduction='mean')

        res_loss = torch.abs(anchor_images - anchor_decode - (negative_images - negative_decode)).mean()
        loss = (anchor_mse + negative_mse) / 2.0

        loss = (1 - self.res_loss_weight) * loss +  self.res_loss_weight * res_loss
        return loss

    def loss_diff_eye_same_gaze(self, positive_gaze_anchor_eye_id,
                                      anchor_gaze_positive_eye_id,
                                      anchor_images,
                                      positive_images):

        anchor_decode = self.decoder(positive_gaze_anchor_eye_id)
        positive_decode = self.decoder(anchor_gaze_positive_eye_id)
        anchor_mse = torch.nn.functional.mse_loss(anchor_decode, anchor_images, reduction='mean')
        positive_mse = torch.nn.functional.mse_loss(positive_decode, positive_images, reduction='mean')
        loss = (anchor_mse + positive_mse) / 2.0
        res_loss = torch.abs(anchor_images - anchor_decode - (positive_images - positive_decode) ).mean()
        loss = (1 - self.res_loss_weight) * loss + self.res_loss_weight * res_loss
        return loss

    def cross_encoder_loss(self, x):
        gaze_labels, left_images, right_images, label = x

        z_left, mu_left, log_var_left = self.variational_embedding(left_images)
        z_right, mu_right, log_var_right = self.variational_embedding(right_images)

        z_dim = z_left.shape[1]
        #gaze_dim = 10
        eye_id_dim = z_dim - self.gaze_dim
        if self.training:
            gaze_left, eye_id_left = torch.split(z_left, [self.gaze_dim, eye_id_dim], dim=1)
            gaze_right, eye_id_right = torch.split(z_right, [self.gaze_dim, eye_id_dim], dim=1)
        else:
            gaze_left, eye_id_left = torch.split(mu_left, [self.gaze_dim, eye_id_dim], dim=1)
            gaze_right, eye_id_right = torch.split(mu_right, [self.gaze_dim, eye_id_dim], dim=1)

        batch_size = gaze_labels.shape[0]
        index = self.generate_derangement(batch_size)

        if random.random() < 0.5:
            anchor_gaze = gaze_left
            anchor_eye_id = eye_id_left
            positive_gaze = gaze_right
            positive_eye_id = eye_id_right
            negative_gaze = gaze_left[index]
            negative_eye_id = eye_id_left[index]
            negative_images = left_images[index]
            anchor_images = left_images
            positive_images = right_images


        else:
            anchor_gaze = gaze_right
            anchor_eye_id = eye_id_right
            positive_gaze = gaze_left
            positive_eye_id = eye_id_left
            negative_gaze = gaze_right[index]
            negative_eye_id = eye_id_right[index]
            negative_images = right_images[index]
            anchor_images = right_images
            positive_images = left_images

        positive_gaze_anchor_eye_id = torch.cat((positive_gaze, anchor_eye_id), dim=1)
        anchor_gaze_positive_eye_id = torch.cat((anchor_gaze, positive_eye_id), dim=1)
        anchor_gaze_negative_eye_id = torch.cat((anchor_gaze, negative_eye_id), dim=1)
        negative_gaze_anchor_eye_id = torch.cat((negative_gaze, anchor_eye_id), dim=1)

        loss_gaze = torch.abs(positive_gaze - anchor_gaze).mean()
        loss_id = torch.abs(anchor_eye_id - negative_eye_id).mean()

        loss_same_eye_different_gaze = self.loss_same_eye_different_gaze(anchor_gaze_negative_eye_id,
                                                                    negative_gaze_anchor_eye_id,
                                                                    anchor_images,
                                                                    negative_images)

        loss_diff_eye_same_gaze = self.loss_diff_eye_same_gaze(positive_gaze_anchor_eye_id,
                                                             anchor_gaze_positive_eye_id,
                                                             anchor_images,
                                                             positive_images)

        alpha = 1.0
        ce_loss = (loss_same_eye_different_gaze + loss_diff_eye_same_gaze) / 2.0
        loss_same_property = (loss_gaze + loss_id) / 2.0
        loss = alpha * ce_loss + (1-alpha)*loss_same_property


        kld_left = self.loss_kld(mu_left, log_var_left)
        kld_right = self.loss_kld(mu_right, log_var_right)

        kld = (kld_left + kld_right) / 2.0

        loss = loss * self.gamma + kld * (1 - self.gamma)


        return loss, kld

    def compute_triplet_loss_label_oracle(self, anchor, positive, negatives, anchor_gt, negative_gts):

        #compute the euclidean distance between anchor to positive
        #negatives = torch.cat(negatives).reshape_as(anchor_gt)
        # negatives = torch.cat(negatives)
        # negative_gts = torch.cat(negative_gts)
        batch_size = anchor.shape[0]
        d_ap = torch.norm(anchor - positive, dim=1)
        sum_negatives = torch.zeros_like(d_ap)
        sum_labels_negatives = torch.zeros_like(d_ap)
        for negative in negatives:
            d_an = torch.norm(anchor - negative, dim=1)
            sum_negatives = sum_negatives + d_an

        #d_an = torch.norm(anchor - negative, dim=1)
        for negative_gt in negative_gts:
            d_label = torch.norm(anchor_gt - negative_gt, dim=1)
            sum_labels_negatives = sum_labels_negatives + d_label


        loss = torch.abs(sum_negatives - sum_labels_negatives)
        loss = torch.mean(loss)
        return loss


        pass

    def oracle_fixed_loss(self, x):
        gaze_labels, left_images, right_images, label = x

        z_proj_left, z_left, mu_left, log_var_left,_ = self.variational_embedding(left_images, vae=False)
        z_proj_right, z_right, mu_right, log_var_right,_ = self.variational_embedding(right_images, vae=False)

        batch_size = gaze_labels.shape[0]
        index = self.generate_derangement(batch_size)

        if random.random() < 0.5:
            left_perm = z_proj_left[index]
            anchor = z_proj_left
            positive = z_proj_right
            negative = left_perm
        else:
            right_perm = z_proj_right[index]
            anchor = z_proj_right
            positive = z_proj_left
            negative = right_perm


        # triplet_loss = (self.compute_triplet_loss(anchor, positive, negative) +
        #                self.compute_triplet_loss(positive, anchor, negative)) / 2.0
        triplet_loss = self.compute_triplet_loss(anchor, positive, negative)

        kld_left = self.loss_kld(mu_left, log_var_left)
        kld_right = self.loss_kld(mu_right, log_var_right)

        kld = (kld_left + kld_right) / 2.0

        loss = triplet_loss * self.gamma + kld * (1 - self.gamma)

        return loss, kld

    def from_3D_to_yaw_pitch(self, vec_3d):
        x = vec_3d[:,0]
        y = vec_3d[:, 1]
        z = vec_3d[:, 2]
        pitch = torch.asin(y).unsqueeze(dim=1)
        yaw = torch.atan2(x, z).unsqueeze(dim=1)

        final = torch.cat((pitch,yaw),dim=1)
        return final

    def oracle_static_labels_loss(self, x):
        gaze_labels, left_images, right_images, label = x

        gaze_labels_2d = self.from_3D_to_yaw_pitch(gaze_labels)

        z_proj_left, z_left, mu_left, log_var_left,_ = self.variational_embedding(left_images, vae=False)
        z_proj_right, z_right, mu_right, log_var_right,_ = self.variational_embedding(right_images, vae=False)

        batch_size = gaze_labels.shape[0]
        #index = self.generate_derangement(batch_size)

        number_of_negatives = batch_size
        indexes = []
        negatives = []
        negative_gts = []
        for i in range(number_of_negatives):
            indexes.append(self.generate_derangement(batch_size))

        if random.random() < 0.5:
            for i in range(number_of_negatives):
                negatives.append(z_proj_left[indexes[i]])
            anchor = z_proj_left
            positive = z_proj_right
            #negative = left_perm
            #negative = left_perm

        else:
            for i in range(number_of_negatives):
                negatives.append(z_proj_left[indexes[i]])
            #right_perm = z_proj_right[index]
            anchor = z_proj_right
            positive = z_proj_left
            #negative = right_perm

        anchor_gt = gaze_labels_2d

        for i in range(number_of_negatives):
            negative_gts.append(gaze_labels_2d[indexes[i]])

        negative_gts = torch.cat(negative_gts).reshape(number_of_negatives, batch_size, 2)
        negatives = torch.cat(negatives).reshape(number_of_negatives, batch_size, 2)
        # triplet_loss = (self.compute_triplet_loss(anchor, positive, negative) +
        #                self.compute_triplet_loss(positive, anchor, negative)) / 2.0
        #triplet_loss = self.compute_triplet_loss(anchor, positive, negative)
        triplet_loss = (self.compute_triplet_loss_label_oracle(anchor, positive, negatives, anchor_gt, negative_gts)
        + self.compute_triplet_loss_label_oracle(positive, anchor, negatives, anchor_gt, negative_gts)) / 2.0

        kld_left = self.loss_kld(mu_left, log_var_left)
        kld_right = self.loss_kld(mu_right, log_var_right)

        kld = (kld_left + kld_right) / 2.0

        loss = triplet_loss * self.gamma + kld * (1 - self.gamma)

        return loss, kld

    def training_step(self, x):

        loss, kld = self.cross_encoder_loss(x)

        self.log('loss_train', loss, on_step=True, on_epoch=True,
                 sync_dist=True)
        self.log('loss_train_kld', kld, on_step=True, on_epoch=True,
                 sync_dist=True)
        return loss

    def configure_optimizers(self):
        #opt = torch.optim.SGD(self.parameters(),lr=0.0)
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams['hparams'].lr,
                                weight_decay=self.hparams['hparams'].weight_decay)

        scheduler = lr_scheduler.MultiStepLR(opt,
                                             milestones=self.hparams['hparams'].lr_milestones,
                                             gamma=self.hparams['hparams'].lr_gamma)

        return [opt], [scheduler]


def np_to_pil_image(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return Image.fromarray(arr)

def tensor_to_pil_image(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return torchvision.transforms.ToPILImage()(tensor)

def loss_function_vae(recon_x, x, mu, log_var):
    MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE, KLD

def loss_function_ae(recon_x, x):
    MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    return MSE


def deactivate_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


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
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
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
        if right_backbone_features is None:
            return self.forward_mono(left_backbone_features)
        else:
            return self.forward_bino(left_backbone_features, right_backbone_features)

#############################################################################################

class Decoder(LightningModule):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # Initial projection to 7x7x512
        self.latent_dim = latent_dim
        self.initial_channel_dim = 256
        #self.initial_channel_dim = 512
        self.fc = torch.nn.Linear(self.latent_dim, 5 * 5 * self.initial_channel_dim)

        # Upsampling blocks to double the spatial dimensions
        self.block1 = self._upsample_block(self.initial_channel_dim, self.initial_channel_dim // 2)  # Output: 10x10
        self.block2 = self._upsample_block(self.initial_channel_dim // 2, self.initial_channel_dim // 4)  # Output: 20x20
        self.block3 = self._upsample_block(self.initial_channel_dim // 4, self.initial_channel_dim // 8)  # Output: 40x40
        self.block4 = self._upsample_block(self.initial_channel_dim // 8, self.initial_channel_dim // 16)  # Output: 80x80
        self.block5 = self._upsample_block(self.initial_channel_dim // 16,
                                           self.initial_channel_dim // 32)  # Output: 160x160
        self.block6 = self._upsample_block(self.initial_channel_dim // 32,
                                           self.initial_channel_dim // 64)  # Output: 320x320

        self.block7 = torch.nn.Sequential(torch.nn.Upsample(mode='nearest', size=(400,400)),  # Double the spatial dimensions
                torch.nn.Conv2d(self.initial_channel_dim // 64, 1, kernel_size=3, padding=1),
                torch.nn.Tanh()
            )



        # Final convolution to adjust to 3 channels for RGB
        #self.final_conv = nn.Conv2d(self.initial_channel_dim // 16, 3, kernel_size=3, padding=1)

    def _upsample_block(self, in_channels, out_channels,size=None):

        scale_factor = 2 if size==None else None
        if scale_factor == 2:
            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                #torch.nn.Upsample(scale_factor=scale_factor, mode='nearest', size=size),  # Double the spatial dimensions
                #torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Tanh()
            )
        else:
            block = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest', size=size),  # Double the spatial dimensions
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.Tanh()
            )

        return block

    def forward(self, x):
        x = x.squeeze()
        x = self.fc(x)
        x = x.view(-1, self.initial_channel_dim, 5, 5)  # Reshape to a spatial dimension

        # Sequentially pass through upsampling blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        #x = torch.tanh(x)  # Use sigmoid to ensure output pixel values are in [0, 1]
        return x


# class Decoder(LightningModule):
#     def __init__(self, hparams, latent_dim: int):
#         """Decoder.
#
#                Args:
#                   num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
#                   base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
#                   latent_dim : Dimensionality of latent representation z
#                   act_fn : Activation function used throughout the decoder network
#                """
#         super(Decoder, self).__init__()
#         self.save_hyperparameters()  # sets self.hparams
#         args = self.hparams['hparams']
#         width_mult_to_depth = {
#             1.0: 64,
#             0.75: 24,
#             2.0: 128,
#             3.0: 192,
#         }
#         k = 128
#         self.model = torch.nn.Sequential(
#             # First layer: latent_dim -> 1024, with upscaling
#             torch.nn.Linear(latent_dim, k * 5 * 5),
#             torch.nn.ReLU(True),
#             # Reshape to a 4D tensor for convolutional layers
#             torch.nn.Unflatten(1, (k, 5, 5)),
#             # Upsample to 4x4
#             torch.nn.ConvTranspose2d(k, k // 2, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 8x8
#             torch.nn.ConvTranspose2d(k // 2, k // 4, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 16x16
#             torch.nn.ConvTranspose2d(k // 4, k // 16, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 32x32
#             torch.nn.ConvTranspose2d(k // 16, k // 32, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             # Upsample to 64x64
#             torch.nn.ConvTranspose2d(k // 32, k // 64, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#             #torch.nn.ReLU(True),
#             # Upsample to 128x128
#             torch.nn.ConvTranspose2d(k // 64, k // 64, kernel_size=4, stride=2, padding=1),
#             torch.nn.ReLU(True),
#
#             # Upsample to 256x256
#             torch.nn.ConvTranspose2d(k // 64, 1, kernel_size=4, stride=2, padding=1),
#             torch.nn.Tanh()  # Normalize the output to [-1, 1]
#         )
#
#     def forward(self, z):
#         z = z.squeeze()
#         x = self.model(z)
#         return x



def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        #torch.nn.init.normal_(m.weight, mean=0.0, std=5.0)
        #torch.nn.init.uniform(m.weight,0,10)
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

#############################################################################################




