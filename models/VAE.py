import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.core import LightningModule
import pytorch_lightning
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import MDS
from PIL import Image, ImageOps
import random
import torchvision
import pandas as pd
from models.mbnv2 import MobileNet_v2
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


class VAE(LightningModule):

    def __init__(self, hparams, channels=1):
        super(VAE, self).__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()  # sets self.hparams
        self.params = self.hparams['hparams']
        args = self.hparams['hparams']
        self.gamma = 0.9
        bottleneckLayerDetails = [
            # (expansion, out_dimension, number_of_times, stride)
            (1, 4, 1, 2),
            (6, 8, 1, 2),
            (6, 16, 1, 2),
            (6, 32, 1, 2),
            (6, 48, 1, 2),
            (6, 64, 1, 2),
        ]

        self.fc_mean = torch.nn.Linear(self.params.backbone_feature_size, self.params.backbone_feature_size)
        self.fc_log_var = torch.nn.Linear(self.params.backbone_feature_size, self.params.backbone_feature_size)
        self.error_list = []
        self.encoder = MobileNet_v2(bottleneckLayerDetails, width_multiplier=self.params.width_multiplier, in_fts=1)

        self.backbone_feaure_size = self.params.backbone_feature_size

        self.channels = 1
        self.validation_it = 0
        self.decoder = Decoder(latent_dim=128, inital_channel_dim=self.params.vae_decoder_initial_channel_dim)
        self.alpha = torch.tensor(args.alpha)
        self.zero = torch.tensor(0.0)
        #self.load_from_checkpoint(hparams.ckpt_path, strict=False)
        # if hparams.ckpt_path != '':
        #     self.load_state_dict(torch.load(hparams.ckpt_path, map_location=self.device)['state_dict'], strict=True)
        # else:
        #     print('apply init weights')
        #     self.encoder.apply(init_weights)

        self.ae_type = 'vae'

        #Freeze backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # for param in self.parameters():
        #     param.requires_grad = False


    def reparameterize(self, mu, log_var):
        st = torch.exp(0.5 * log_var)
        eps = torch.randn_like(st)
        return mu + eps * st


    def training_validation_step_ssl(self, x):
        gaze_labels, left_images, right_images, label = x

        left_images_encoded = self.encoder(left_images)
        right_images_encoded = self.encoder(right_images)

        if self.ae_type == 'vae':
            mu_left = self.fc_mean(left_images_encoded.squeeze(2).squeeze(2))
            log_var_left = self.fc_log_var(left_images_encoded.squeeze(2).squeeze(2))
            z_left = self.reparameterize(mu_left, log_var_left)

            decoded_left = self.decoder(z_left)

            loss_left_mse,  loss_left_kld = loss_function_vae(decoded_left, left_images,mu_left,log_var_left)

            mu_right = self.fc_mean(right_images_encoded.squeeze(2).squeeze(2))
            log_var_right = self.fc_log_var(right_images_encoded.squeeze(2).squeeze(2))
            z_right = self.reparameterize(mu_right, log_var_right)
            decoded_right = self.decoder(z_right)

            loss_right_mse, loss_right_kld = loss_function_vae(decoded_right, right_images, mu_right, log_var_right)

            loss_mse = (loss_left_mse + loss_right_mse) / 2.0
            loss_kld = (loss_left_kld + loss_right_kld) / 2.0

            loss = loss_mse * self.gamma + loss_kld * (1 - self.gamma)


            self.log('loss_val_mse', loss_mse, on_step=True, on_epoch=True,
                     sync_dist=True)
            self.log('loss_val_kld', loss_kld, on_step=True, on_epoch=True,
                     sync_dist=True)

        else:

            decoded_left = self.decoder(left_images_encoded)
            decoded_right = self.decoder(right_images_encoded)
            loss_left = loss_function_ae(decoded_left, left_images)
            loss_right = loss_function_ae(decoded_right, right_images)

            loss = (loss_right + loss_left) / 2.0

            self.log('loss_val_mse', loss, on_step=True, on_epoch=True,
                     sync_dist=True)

        if 0:
            batch_size = left_images.shape[0]
            for i in tqdm(range(batch_size)):
                person_id = label['person_id'][i]
                sample_id =  int(label['Unnamed: 0_x'][i].cpu().numpy())
                session_id = int(label['session_number'][i].cpu().numpy())
                gridx = float(label['gridx'][i].cpu().numpy())
                gridy = float(label['gridy'][i].cpu().numpy())
                new_row = {
                    'sample_id': sample_id,
                    'log_var_left': log_var_left.cpu()[i],
                    'mu_left': mu_left.cpu()[i],
                    'log_var_right': log_var_right.cpu()[i],
                    'mu_right': mu_right.cpu()[i],
                    'session_number': session_id,
                    'person_id': person_id,
                    'gridx': gridx,
                    'gridy': gridy,
                    'decoded_left': tensor_to_pil_image(decoded_left[i].cpu()),
                    'decoded_right': tensor_to_pil_image(decoded_right[i].cpu())
                }
                df = pd.DataFrame(new_row)
                out_path = os.path.join('/home/gilsh/temp/', 'val_output')
                os.makedirs(out_path, exist_ok=True)
                df_name = os.path.join(out_path,
                                       'person_id_{}_sample_id_{}_gridx_{}_gridy_{}.pkl'.format(person_id, sample_id,
                                                                                         gridx, gridy))
                df.to_pickle(df_name)
            #self.validation_table_data.append(new_row)

        return loss

    def training_validation_step_supervised(self, x):
        gaze_labels, left_images, right_images, label = x

        left_images_encoded = self.encoder(left_images)
        right_images_encoded = self.encoder(right_images)

        mu_lefts = self.fc_mean(left_images_encoded.squeeze(2).squeeze(2))
        log_var_left = self.fc_log_var(left_images_encoded.squeeze(2).squeeze(2))
        z_lefts = self.reparameterize(mu_lefts, log_var_left)
        decoded_left = self.decoder(z_lefts)

        mu_rights = self.fc_mean(right_images_encoded.squeeze(2).squeeze(2))
        log_var_right = self.fc_log_var(right_images_encoded.squeeze(2).squeeze(2))
        z_rights = self.reparameterize(mu_rights, log_var_right)
        decoded_right = self.decoder(z_rights)

        z_lefts = z_lefts.cpu().float().numpy()
        z_rights = z_rights.cpu().float().numpy()
        decoded_rights = decoded_right.cpu().float().numpy()
        decoded_lefts = decoded_left.cpu().float().numpy()
        gaze_labels = gaze_labels.cpu().float().numpy()
        person_ids = label['person_id']
        session_numbers = label['session_number']

        for person_id, gaze_label,z_left, z_right, session_number, decoded_left, decoded_right in\
                zip(person_ids, gaze_labels, z_lefts, z_rights, session_numbers, decoded_lefts, decoded_rights):
            new_row = {
                'z_left': z_left,
                'z_right': z_right,
                'session_number': session_number.cpu(),
                'person_id': person_id,
                'gt': gaze_label,
                'decoded_left': decoded_left,
                'decoded_right': decoded_right
            }
            self.validation_table_data.append(new_row)

        pass

    def draw_scatter_plot(self, pca_features, gaze_labels, images):
        # Creating the plot with larger dimensions
        fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size if needed
        ax.set_xlim(pca_features[:, 0].min(), pca_features[:, 0].max())
        ax.set_ylim(pca_features[:, 1].min(), pca_features[:, 1].max())

        # Adding images to the scatter plot with a very large zoom factor
        for i, image in enumerate(images):

            #image = img_to_array(image, scale_factor=0.2)
            x, y = pca_features[i, 0], pca_features[i, 1]
            label_text = f"({gaze_labels[i,0]:.2f}, {gaze_labels[i,1]:.2f}, {gaze_labels[i,2]:.2f})"
            ax.text(x, y, label_text, fontsize=9, ha='right', va='bottom')
            # render image on plot in the location x, y
            imagebox = OffsetImage(image, zoom=0.10, cmap='gray', alpha=0.7)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

        ax.set_xlabel('VAE First PCA Axis')
        ax.set_ylabel('VAE Second PCA Axis')
        plt.title('Decoded VAE images placed on VAE PCA 2D Embedding')
        plt.show()
        pass

    def apply_pca_and_apply_on_test(self, features_calib, features_test, decoded_left, decoded_right,
                                    gaze_labels, side='left'):

        features_calib = np.concatenate(features_calib, axis=0).reshape(len(features_calib), len(features_calib[0]))
        features_test = np.concatenate(features_test, axis=0).reshape(len(features_test), len(features_test[0]))

        left_images = []
        right_images = []

        for d_l, d_r in zip(decoded_left, decoded_right):
            left_images.append(np_to_pil_image(d_l))
            right_images.append(np_to_pil_image(d_r))


        #scaler = StandardScaler()
        #features_calib_scaled = scaler.fit_transform(features_calib)
        pca = PCA(n_components=10)
        #pca = joblib.load('/home/gilsh/temp/pca_model_' + side + '_10d.pkl')

        # test_features_pca = pca.fit_transform(features_test)
        # calib_features_pca = pca.transform(features_calib)
        calib_features_pca = pca.fit_transform(features_calib)
        test_features_pca = pca.transform(features_test)
        mds = MDS(n_components=5, random_state=42)

        # Fit the MDS model and transform the data
        calib_features_pca_mds = mds.fit_transform(features_calib)
        test_features_pca_mds = mds.transform(features_test)
        # if side == 'left':
        #     self.draw_scatter_plot(calib_features_pca_mds, gaze_labels, left_images)
        # else:
        #     self.draw_scatter_plot(calib_features_pca_mds, gaze_labels, right_images)
        # tsne = TSNE(n_components=2, metric='cosine')
        # calib_tsne_features = tsne.fit_transform(features_calib)
        # calib_dmp_features = diffusion_map(features_test_pca, sigma=2, dim=2)
        # self.draw_scatter_plot(calib_tsne_features, gaze_labels, left_images)
        # self.draw_scatter_plot(calib_pca_features, gaze_labels, left_images)

        return calib_features_pca_mds, test_features_pca_mds
        #return calib_features_pca, test_features_pca

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

    def calc_ensemble_calib(self, group_id_df, max_polar_angle=60, number_of_calib_points=8):
        session_cutoff = 4
        data_for_calib = group_id_df[group_id_df['session_number'] <= session_cutoff]
        left_features_calib = data_for_calib['z_left'].values
        left_features_calib = np.concatenate(left_features_calib, axis=0).reshape(len(left_features_calib), len(left_features_calib[0]))
        right_features_calib = data_for_calib['z_right'].values
        right_features_calib = np.concatenate(right_features_calib, axis=0).reshape(len(right_features_calib),
                                                                                  len(right_features_calib[0]))
        gaze_labels_fit = data_for_calib['gt'].values
        gaze_labels_fit = np.concatenate(gaze_labels_fit, axis=0).reshape(len(gaze_labels_fit), 3)
        decoded_left = data_for_calib['decoded_left'].values
        decoded_right = data_for_calib['decoded_right'].values
        decoded_left = np.concatenate(decoded_left, axis=0).reshape(len(decoded_left), decoded_left[0].shape[1], decoded_left[0].shape[2])
        decoded_right = np.concatenate(decoded_right, axis=0).reshape(len(decoded_right), decoded_right[0].shape[1],
                                                                    decoded_right[0].shape[2])

        data_for_test = group_id_df[group_id_df['session_number'] > session_cutoff]
        left_features_test = data_for_test['z_left'].values
        right_features_test = data_for_test['z_right'].values
        left_features_test = np.concatenate(left_features_test, axis=0).reshape(len(left_features_test),
                                                                                  len(left_features_test[0]))

        right_features_test = np.concatenate(right_features_test, axis=0).reshape(len(right_features_test),
                                                                                  len(right_features_test[0]))

        gaze_labels_test = data_for_test['gt'].values
        gaze_labels_test = np.concatenate(gaze_labels_test, axis=0).reshape(len(gaze_labels_test), 3)

        left_features_calib_pca, left_feautres_test_pca = self.apply_pca_and_apply_on_test(left_features_calib, left_features_test, decoded_left, decoded_right
                                                                                       , gaze_labels_fit, side='left')
        right_features_calib_pca, right_features_test_pca = self.apply_pca_and_apply_on_test(right_features_calib, right_features_test, decoded_left, decoded_right,
                                                                                         gaze_labels_fit, side='right')
        gaze_labels_test_2d = self.from_3D_to_yaw_pitch_np(gaze_labels_test)
        gaze_labels_calib_2d = self.from_3D_to_yaw_pitch_np(gaze_labels_fit)

        left_fit_models, _= self.compute_ensemble_fit(gaze_labels_calib_2d, left_features_calib_pca)

        #left test
        transformed_test_por_left = left_fit_models.predict(left_feautres_test_pca)
        # transformed_test_por_left = self.apply_ensemble_calib(left_poly_models, left_fit_models,
        #                                              left_feautres_test)
        #transformed_test_por_left = self.from_yaw_pitch_to_3D_np(transformed_test_por_left)

        #right test
        right_fit_models, _ = self.compute_ensemble_fit(gaze_labels_calib_2d, right_features_calib_pca)

        transformed_test_por_right = right_fit_models.predict(right_features_test_pca)
        # transformed_test_por_right = self.apply_ensemble_calib(right_poly_models, right_fit_models,
        #                                                  right_features_test)

        transformed_test_por_right = self.from_yaw_pitch_to_3D_np(transformed_test_por_right)
        transformed_test_por_left = self.from_yaw_pitch_to_3D_np(transformed_test_por_left)

        transformed_test_por = (transformed_test_por_right + transformed_test_por_left) / 2.0
        norms = np.linalg.norm(transformed_test_por, axis=1, keepdims=True)

        # Avoid division by zero by setting zero norms to one
        norms[norms == 0] = 1
        transformed_test_por = transformed_test_por / norms


        average_error_after_calib, avg_dist_after_calib, error_deg_after_calib, median_err_after_calib = self.calc_metrics_np(transformed_test_por, gaze_labels_test)
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

    def compute_ensemble_fit(self, gt_fit, por_fit, max_polar_angle=60, number_of_calib_points=17):
        poly_models = []
        fit_models = []
        model = KernelRidge(alpha=0.1, kernel='linear')

        poly = None
        #model = Ridge()
        model.fit(por_fit, gt_fit)
        return model, poly

        # for poly_deg in range(1,5):
        #     fit_model, poly_model = self.compute_fit_ridge(por_fit, gt_fit, poly_deg, kernel='linear')
        #     por_fit_poly = poly_model.fit_transform(por_fit)
        #     transformed_fit_por = fit_model.predict(por_fit_poly)
        #     poly_models.append(poly_model)
        #     fit_models.append(fit_model)

        return poly_models, fit_models

    def on_validation_epoch_end(self):


        print('on_validation_end called')

        self.validation_table = pd.DataFrame(self.validation_table_data)

        #self.validation_table.to_pickle('/home/gilsh/temp/dataframe_all.pkl')
        #return
        #self.validation_table = pd.read_pickle('/home/gilsh/Gaze/sgaze/dataframe_all.pkl')
        #self.validation_table = pd.read_pickle('/home/gilsh/temp/dataframe_all.pkl')
        group_by_id = self.validation_table.groupby('person_id')
        after_calib_avg_err_list = []
        after_calib_med_err_list = []

        error_data = []
        for group_id_name, group_id_df in group_by_id:

            #compute PCA for left and right features

            average_error_after_calib,\
            median_error_after,\
            error_per_label = self.calc_ensemble_calib(group_id_df, max_polar_angle=60, number_of_calib_points=8)

            self.log('val_avg_error_id_{}_after_calib'.format(group_id_name), average_error_after_calib, on_step=False, on_epoch=True,
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

        return

    def on_validation_epoch_start(self):
        #self.to('cpu')
        #self.validation_table = pd.DataFrame(columns=['subject_id', 'por_estimation', 'gt'])
        self.validation_table_data = []

    def validation_step(self, x, batch_idx, dataloader_idx=0):

        if dataloader_idx == 0:
            #ssl
            return self.training_validation_step_ssl(x)
        else:
            #supervised
            return self.training_validation_step_supervised(x)




    def training_step(self, x):
        gaze_labels, left_images, right_images, label = x

        left_images_encoded = self.encoder(left_images)
        right_images_encoded = self.encoder(right_images)

        if self.ae_type == 'vae':
            mu_left = self.fc_mean(left_images_encoded.squeeze(2).squeeze(2))
            log_var_left = self.fc_log_var(left_images_encoded.squeeze(2).squeeze(2))
            z_left = self.reparameterize(mu_left, log_var_left)
            decoded_left = self.decoder(z_left)
            loss_left_mse,  loss_left_kld = loss_function_vae(decoded_left, left_images,mu_left,log_var_left)

            mu_right = self.fc_mean(right_images_encoded.squeeze(2).squeeze(2))
            log_var_right = self.fc_log_var(right_images_encoded.squeeze(2).squeeze(2))
            z_right = self.reparameterize(mu_right, log_var_right)
            decoded_right = self.decoder(z_right)
            loss_right_mse, loss_right_kld = loss_function_vae(decoded_right, right_images, mu_right, log_var_right)


            loss_mse = (loss_left_mse + loss_right_mse) / 2.0
            loss_kld = (loss_left_kld + loss_right_kld) / 2.0
            loss = loss_mse * self.gamma + loss_kld * (1 - self.gamma)

            # m = decoded_left.min()
            # decoded_left = decoded_left - m
            # decoded_left = decoded_left / decoded_left.max()
            # m = decoded_right.min()
            # decoded_right = decoded_right - m
            # decoded_right = decoded_right / decoded_right.max()

            #torchvision.utils.save_image(decoded_right , 'decoded_right.png')
            #torchvision.utils.save_image(decoded_left, 'decoded_left.png')
            self.log('loss_train_mse', loss_mse, on_step=True, on_epoch=True,
                     sync_dist=True)
            self.log('loss_train_kld', loss_kld, on_step=True, on_epoch=True,
                     sync_dist=True)

        else:

            decoded_left = self.decoder(left_images_encoded)
            decoded_right = self.decoder(right_images_encoded)
            DEBUG = False


            loss_left = loss_function_ae(decoded_left, left_images)
            loss_right = loss_function_ae(decoded_right, right_images)

            loss = (loss_right + loss_left) / 2
            if DEBUG:
                m = decoded_left.min()
                decoded_left = decoded_left - m
                decoded_left = decoded_left / decoded_left.max()
                m = decoded_right.min()
                decoded_right = decoded_right - m
                decoded_right = decoded_right / decoded_right.max()
                torchvision.utils.save_image(decoded_right , 'decoded_right.png')
                torchvision.utils.save_image(decoded_left, 'decoded_left.png')

            self.log('loss_train_mse', loss, on_step=True, on_epoch=True,
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
    def __init__(self, latent_dim, inital_channel_dim):
        super(Decoder, self).__init__()
        # Initial projection to 7x7x512
        self.latent_dim = latent_dim
        self.initial_channel_dim = inital_channel_dim
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




