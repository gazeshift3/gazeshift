import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# import matplotlib as mpl
# from matplotlib import pyplot as plt
# import random

#mpl.use('TkAgg')


from Datasets.VRGazeDatasetUnsupervised import VRGazeDatasetUnsupervised
from Datasets.VRGazeSinglePersonDataset import VRGazeSinglePersonDataset
import os
import pandas as pd
import glob
import numpy as np
import torch
from sklearn.preprocessing import normalize
import random
from torch.utils.data.sampler import Sampler

import torch
import math
import random
from torch.utils.data import Sampler


def yaw_pitch_to_xyz(yaw, pitch):
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    return x, y, z

# class PerPersonBatchSampler(Sampler):
#     def __init__(self, training_set_df, batch_size, drop_last=False, num_replicas=None, rank=None):
#         # Dataset information
#         self.df = training_set_df
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#
#         # Generate the batches initially
#         self._generate_batches()
#
#     def _generate_batches(self):
#         self.all_batches = []
#         group_by_id = self.df.groupby('id')
#
#         # Create batches per person
#         for group_id_name, group_id_df in group_by_id:
#             number_of_batches_for_person = len(group_id_df) // self.batch_size
#             for i in range(number_of_batches_for_person):
#                 batch_index = group_id_df.sample(n=self.batch_size, replace=False).index
#                 self.all_batches.append(batch_index)
#
#             # Handle leftover samples that don't form a full batch
#             leftover = len(group_id_df) % self.batch_size
#             if leftover != 0 and not self.drop_last:
#                 # Create a smaller batch with the leftover samples
#                 batch_index = group_id_df.sample(n=leftover, replace=False).index
#                 self.all_batches.append(batch_index)
#
#         # Randomize self.all_batches
#         random.shuffle(self.all_batches)
#
#         # Divide batches across replicas (GPUs)
#         self.num_samples = math.ceil(len(self.all_batches) / self.num_replicas)
#         self.total_size = self.num_samples * self.num_replicas
#
#         # Ensure each GPU gets a distinct set of batches
#         self.all_batches = self.all_batches[self.rank:self.total_size:self.num_replicas]
#
#     def set_epoch(self, epoch):
#         """Used to shuffle the data differently at each epoch for DDP"""
#         random.seed(epoch)
#         random.shuffle(self.all_batches)
#
#     def __iter__(self):
#         return iter(self.all_batches)
#
#     def __len__(self):
#         return len(self.all_batches)


class PerPersonBatchSampler(Sampler):
    def __init__(self, training_set_df, batch_size, drop_last=False):
        self.df = training_set_df
        self.batch_size = batch_size
        self.drop_last = drop_last  # Add this line
        self._generate_batches()

    def _generate_batches(self):
        self.all_batches = []
        group_by_id = self.df.groupby('person_id')
        for group_id_name, group_id_df in group_by_id:
            number_of_batches_for_person = len(group_id_df) // self.batch_size
            for i in range(number_of_batches_for_person):
                batch_index = group_id_df.sample(n=self.batch_size, replace=False).index
                self.all_batches.append(batch_index)

            # Handle leftover samples that don't form a full batch
            leftover = len(group_id_df) % self.batch_size
            if leftover != 0 and not self.drop_last:
                # Create a smaller batch with the leftover samples
                batch_index = group_id_df.sample(n=leftover, replace=False).index
                self.all_batches.append(batch_index)

        # Randomize self.all_batches
        random.shuffle(self.all_batches)

    def __iter__(self):
        return iter(self.all_batches)

    def __len__(self):
        return len(self.all_batches)


# class PerPersonBatchSampler(Sampler):
#     def __init__(self, training_set_df, batch_size):
#         self.df = training_set_df
#         self.batch_size = batch_size
#         self._generate_batches()
#
#     def _generate_batches(self):
#         self.all_batches = []
#         batches_count = 0
#         # Sample equal number of indices for each class per batch
#         group_by_id = self.df.groupby('id')
#         for group_id_name, group_id_df in group_by_id:
#             number_of_batches_for_person = len(group_id_df) // self.batch_size
#             for i in range(number_of_batches_for_person):
#                 batch_index = group_id_df.sample(n=self.batch_size, replace=False).index
#                 self.all_batches.append(batch_index)
#                 batches_count += 1
#             # if batches_count == self.batch_size:
#             #     self.all_batches = all_batches
#             #     return
#         #randomize self.all_bacthes
#         random.shuffle(self.all_batches)
#         pass
#     def __iter__(self):
#         return iter(self.all_batches)
#
#     def __len__(self):
#         return len(self.all_batches)

class VRGazeDataModuleUnsupervised(pl.LightningDataModule):
    """
    This data module is creating a person and eye specific datasets.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        #self.person_training_set_size = args.person_training_set_size
        self.data_dir = args.dataset_path[0]
        self.subject_id = args.person_id_calib
        self.channels = args.channels
        self.eye_type = args.eye_type
        #read NVGaze lables
        self.training_set_size = args.training_set_size
        self.NUMBER_OF_GRID_CELLS_FROM_CENTER_TO_FOV_EDGE = 32
        self.BINOCULAR_FOV = 90
        self.unsupervised = args.unsupervised
        self.create_ssl_dataloader = False
        #self.nvgaze_labels = self.read_ts_labels()
        if not args.predict:
            self.nvgaze_labels = self.read_ts_labels_df()

        if args.batch_per_person:
            self.per_person_batch_train_sampler = PerPersonBatchSampler(self.training_set_ssl, self.hparams.args.train_batch_size)

        # self.per_person_batch_val_supervised_sampler = PerPersonBatchSampler(self.validation_set_supervised,
        #                                                                      self.hparams.args.val_batch_size)

        #self.split_train_validation()
        self.input_height = args.input_height
        self.input_width = args.input_width

        pass


    def angles_to_grid(self, azimouth, altitude):

        c = self.NUMBER_OF_GRID_CELLS_FROM_CENTER_TO_FOV_EDGE * np.tan(np.radians(altitude))
        x = c * np.cos(np.radians(azimouth))
        y = c * np.sin(np.radians(azimouth))
        return x, y

    def grid_to_vec(self, gridx, gridy):

        gridz = np.zeros_like(gridx) + self.NUMBER_OF_GRID_CELLS_FROM_CENTER_TO_FOV_EDGE
        gridx = np.expand_dims(gridx, axis=1)
        gridy = np.expand_dims(gridy, axis=1) * 49.0 / 27.0
        gridz = np.expand_dims(gridz, axis=1)
        vec = np.concatenate((gridx, gridy, gridz ),axis=1)
        vec = normalize(vec,axis=1)
        return vec



    def grid_to_angles(self, gridx, gridy):


        dist_from_eye_to_grid = self.NUMBER_OF_GRID_CELLS_FROM_CENTER_TO_FOV_EDGE / np.tan(np.radians(self.BINOCULAR_FOV/2))
        #azimuth is the angle from positive x axis
        azimuth = np.degrees(np.arctan2(gridy, gridx))
        length = np.sqrt(gridx * gridx + gridy * gridy)
        altitude = np.degrees(np.arctan2(length, dist_from_eye_to_grid))
        return azimuth, altitude


    def find_files(self, directory, file_name):
        found_files = []

        # Walk through the directory and its subdirectories
        for root, dirs, files in os.walk(directory):
            # Check if the file_name exists in the list of files in the current directory
            if file_name in files:
                # If found, add the full path to the list of found_files
                found_files.append(os.path.join(root, file_name))

        return found_files


    def read_all_data(self,meta_df, data_type):
        # Every row in the meta file is a dataset
        all_df = []
        for index, row in meta_df.iterrows():
            dataset_path = row['Relative Path']
            if type(dataset_path) is not str:
                continue

            dataset_df_path = os.path.join(self.data_dir, dataset_path, data_type)

            if not os.path.exists(dataset_df_path):
                continue
            df = pd.read_csv(dataset_df_path)
            all_df.append(df)

        all_df = pd.concat(all_df)
        all_df['id'] = all_df['person_id'].str.slice(2)
        merged_df = pd.merge(all_df, meta_df, on='id', how='left')
        return merged_df

    def read_ts_labels_df(self):

        self.data_dir = '/stage/algo-datasets/DB/GazeEstimation/ThundersoftVRSet/VRGaze'
        training_set_path = '/stage/algo-datasets/DB/GazeEstimation/ThundersoftVRSet/VRGaze/metadata_train.csv'
        df = pd.read_csv(training_set_path)
        df[['person_id']] = df.apply(
            lambda row: pd.Series(row['image_l'][0:5]),
            axis=1
        )

        self.training_set_ssl = df

        supervised_valset_path = '/stage/algo-datasets/DB/GazeEstimation/ThundersoftVRSet/VRGaze/metadata_val_static_points.csv'
        df = pd.read_csv(supervised_valset_path)
        df[['vec_3d_x', 'vec_3d_y', 'vec_3d_z']] = df.apply(
            lambda row: pd.Series(yaw_pitch_to_xyz(row['gaze_yaw_degrees'] * math.pi / 180.0, row['gaze_pitch_degrees'] * math.pi / 180.0)),
            axis=1
        )

        df[['person_id']] = df.apply(
            lambda row: pd.Series(row['image_l'][0:5]),
            axis=1
        )
        self.validation_set_supervised = df
        # print('Training set size {}, validation set ssl size {}, validation set supervised size {}'.format(len(self.training_set_ssl),
        #                                                                                                    len(self.validation_set_ssl),
        #                                                                                                    len(self.validation_set_supervised)))

        pass
        return
        ########################################
        meta_file_path = os.path.join(self.data_dir, 'metadata.csv')
        meta_pd = pd.read_csv(meta_file_path, sep='\t', lineterminator='\r', dtype={'id':str})
        meta_pd = meta_pd[meta_pd['id'].notna()]

        unfiltered_data_pd = self.read_all_data(meta_pd, 'unfiltered_labels.df')
        filtered_data_pd = self.read_all_data(meta_pd, 'filtered_labels.df')

        unfiltered_data_pd = unfiltered_data_pd.sample(frac=1).reset_index(drop=True)
        filtered_data_pd = filtered_data_pd.sample(frac=1).reset_index(drop=True)

        self.validation_set_supervised = filtered_data_pd[filtered_data_pd['Validation'] == 1.0]
        self.validation_set_supervised = self.validation_set_supervised[self.validation_set_supervised['is_stationary'] == 1.0]

        if self.hparams.args.unsupervised:
            self.training_set_ssl = unfiltered_data_pd[unfiltered_data_pd['Train'] == 1.0]
            self.validation_set_ssl = unfiltered_data_pd[unfiltered_data_pd['Validation'] == 1.0]
        else:
            self.training_set_ssl = filtered_data_pd[filtered_data_pd['Train'] == 1.0]
            self.training_set_ssl = self.training_set_ssl[
                self.training_set_ssl['is_stationary'] == 1.0]
            self.validation_set_ssl = self.validation_set_supervised


        self.training_set_ssl.reset_index(inplace=True, drop=True)
        self.validation_set_ssl.reset_index(inplace=True, drop=True)
        self.validation_set_supervised.reset_index(inplace=True, drop=True)
        print('Training set size {}, validation set ssl size {}, validation set supervised size {}'.format(len(self.training_set_ssl),
                                                                                                           len(self.validation_set_ssl),
                                                                                                           len(self.validation_set_supervised)))

        pass







    def farthest_point_sampling(self, points, num_points):
        """
        Performs farthest point sampling on a set of 3D points using cosine distance.

        Args:
        - points (numpy array of shape (N, 3)): the set of points to sample from
        - num_points (int): the number of points to sample

        Returns:
        - indices (numpy array of shape (num_points,)): the indices of the sampled points in the original set
        """
        # Initialize the list of sampled indices with the index of the first point
        #points = np.array(df['gaze_gt_vec'])
        # a = [x[1:-1].split() for x in points]
        # out_p = []
        # for p in a:
        #     float_p = [float(x) for x in p]
        #     out_p.append(float_p)
        # points = np.array(out_p)

        indices = [0]

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




    def create_person_specific_train_validation(self):
        all_data = self.test_labels.loc[(self.test_labels['subject_id'] == self.person_id_calib) &
                                             (self.test_labels['eye_type'] == self.eye_type)]

        all_data = self.verify_image_exist(all_data)

        if self.hparams.args.calib_sampling_mathod == 'FPS':
            fps_indexes = self.farthest_point_sampling(all_data, self.person_training_set_size)
            self.train_data_for_person = all_data.iloc[fps_indexes]

        else:
            #random sampling
            self.train_data_for_person = all_data.sample(n=self.person_training_set_size)

        self.val_data_for_person = all_data.drop(self.train_data_for_person.index)

        pass

    def verify_image_exist(self, data):
        for index, row in data.iterrows():
            seq = row.image.split('\\')[0]
            im = row.image.split('\\')[1]
            img_path = os.path.join(self.test_dir, seq, im + '.png')
            if not os.path.exists(img_path):
                data.drop(index, inplace=True)
        return data


    def prepare_data(self):
        # no downloading is needed
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ts_dataset_train = VRGazeDatasetUnsupervised(self.hparams.args, self.data_dir, self.training_set_ssl,stage='train')
            #self.ts_dataset_val_ssl = TSDatasetSelfSupervision(self.hparams.args, self.data_dir, self.validation_set_ssl,stage='validate')
            self.ts_dataset_val_supervised = VRGazeSinglePersonDataset(self.hparams.args, self.data_dir,
                                                               self.validation_set_supervised, stage='val')


        # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     pass
        #     #self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        #
        # if stage == "validate":
        #     self.ts_dataset_val_ssl = TSDatasetSelfSupervision(self.hparams.args, self.data_dir, self.validation_set_ssl,stage='validate')
        #     self.ts_dataset_val_supervised = TSSinglePersonDataset(self.hparams.args, self.data_dir,
        #                                                        self.validation_set_supervised, stage='validate')
        #
        # if stage == "predict":
        #     self.ts_dataset_predict = TSDatasetSelfSupervision(self.hparams.args, self.hparams.args.input_dir, stage='predict')


    def train_dataloader(self):
        #sampler = torch.utils.data.RandomSampler(self.open_eds_gaze_dataset_train, replacement=True, num_samples=self.hparams.args.train_batch_size, generator=None)
        #return DataLoader(self.ts_dataset_train, batch_size=self.hparams.args.train_batch_size, num_workers=self.hparams.args.num_workers, shuffle=True)

        if self.hparams.args.batch_per_person:
            return DataLoader(self.ts_dataset_train, batch_sampler=self.per_person_batch_train_sampler, num_workers=self.hparams.args.num_workers)
        else:
            return DataLoader(self.ts_dataset_train, batch_size=self.hparams.args.train_batch_size,
                              num_workers=self.hparams.args.num_workers, shuffle=True)

    def val_dataloader(self):
    #    return DataLoader(self.ts_dataset_val, batch_size=self.hparams.args.val_batch_size, num_workers=self.hparams.args.num_workers, shuffle=False)
        data_loaders = []
        supervised_dataloader = DataLoader(self.ts_dataset_val_supervised, batch_size=self.hparams.args.val_batch_size,
                                       num_workers=self.hparams.args.num_workers)
        #data_loaders.append(supervised_dataloader)

        # if self.create_ssl_dataloader:
        #     ssl_dataloader = DataLoader(self.ts_dataset_val_ssl, batch_sampler=self.per_person_batch_val_ssl_sampler,
        #                   num_workers=self.hparams.args.num_workers)
        #     data_loaders.append(ssl_dataloader)

        return supervised_dataloader

    def test_dataloader(self):
        return DataLoader(self.ts_dataset_val, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.ts_dataset_predict, batch_size=1)
