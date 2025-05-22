import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

import matplotlib as mpl
from matplotlib import pyplot as plt
import random

mpl.use('TkAgg')

from Datasets.NVGazeMonocularDataset import NVGazeMonocularDataset
from Datasets.VRGazeSinglePersonDataset import TSSinglePersonDataset
import os
import pandas as pd
import glob
import numpy as np
import torch
from sklearn.preprocessing import normalize
import random
from torch.utils.data.sampler import Sampler


class PerPersonBatchSampler(Sampler):
    def __init__(self, training_set_df, batch_size):
        self.df = training_set_df
        self.batch_size = batch_size
        self._generate_batches()

    def _generate_batches(self):
        self.all_batches = []
        batches_count = 0
        # Sample equal number of indices for each class per batch
        group_by_id = self.df.groupby('id')
        for group_id_name, group_id_df in group_by_id:
            number_of_batches_for_person = len(group_id_df) // self.batch_size
            for i in range(number_of_batches_for_person):
                batch_index = group_id_df.sample(n=self.batch_size, replace=False).index
                self.all_batches.append(batch_index)
                batches_count += 1
            # if batches_count == self.batch_size:
            #     self.all_batches = all_batches
            #     return
        pass
    def __iter__(self):
        return iter(self.all_batches)

    def __len__(self):
        return len(self.all_batches)

class TSSinglePersonDataModule(pl.LightningDataModule):
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
        #self.nvgaze_labels = self.read_ts_labels()
        if not args.predict:
            self.nvgaze_labels = self.read_ts_labels_df()

        if args.batch_per_person:
            self.per_person_batch_train_sampler = PerPersonBatchSampler(self.training_set, self.hparams.args.train_batch_size)
            self.per_person_batch_val_sampler = PerPersonBatchSampler(self.validation_set,
                                                                        self.hparams.args.val_batch_size)
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


    def read_all_data(self,meta_df):
        # Every row in the meta file is a dataset
        all_df = []
        for index, row in meta_df.iterrows():
            dataset_path = row['Relative Path']
            if type(dataset_path) is not str:
                continue
            if self.unsupervised:
                dataset_df_path = os.path.join(self.data_dir, dataset_path, 'unfiltered_labels.df')
            else:
                dataset_df_path = os.path.join(self.data_dir, dataset_path, 'filtered_labels.df')

            if not os.path.exists(dataset_df_path):
                continue
            df = pd.read_csv(dataset_df_path)
            all_df.append(df)

        all_df = pd.concat(all_df)
        all_df['id'] = all_df['person_id'].str.slice(2)
        merged_df = pd.merge(all_df, meta_df, on='id', how='left')
        return merged_df

    def read_ts_labels_df(self):
        meta_file_path = os.path.join(self.data_dir, 'metadata.csv')
        meta_pd = pd.read_csv(meta_file_path, sep='\t', lineterminator='\r', dtype={'id':str})
        meta_pd = meta_pd[meta_pd['id'].notna()]
        all_data_pd = self.read_all_data(meta_pd)

        #ONLY_STATIONARY = False
        self.training_set = all_data_pd[all_data_pd['Train'] == 1.0]
        if self.hparams.args.only_stationary_train_point:
            self.training_set = self.training_set[self.training_set['is_stationary'] == 1.0]

        self.validation_set = all_data_pd[all_data_pd['Validation'] == 1.0]

        if self.hparams.args.only_stationary_validation_point:
            self.validation_set = self.validation_set[self.validation_set['is_stationary'] == 1.0]


        #Gil - validation now contain both stationary and dynamic
        #self.validation_set = self.validation_set[self.validation_set['is_stationary'] == 1.0]
        self.training_set.reset_index(inplace=True, drop=True)
        self.validation_set.reset_index(inplace=True, drop=True)
        print('Training set size {}, validation set size {}'.format(len(self.training_set), len(self.validation_set)))
        pass
        # df_file_path = os.path.join(self.data_dir, 'filtered_labels.df')
        # labels = pd.read_csv(df_file_path)
        # person_id = 'id013'
        # person_specific_data = labels[labels['person_id'].isin([person_id])]
        # stationary_target_labels = person_specific_data[person_specific_data['is_stationary'] == 1]
        # moving_target_labels = person_specific_data[person_specific_data['is_stationary'] == 0]
        # total_count_of_stationary_labels = len(stationary_target_labels)
        # val_size = total_count_of_stationary_labels // 2
        #
        # val_size = 91
        # # zero_point_indexes = stationary_target_labels[
        # #     (stationary_target_labels['gridx_x'] == 0.0) & (stationary_target_labels['gridy_x'] == 0.0)]
        #
        # #stationary_target_labels_no_zero = stationary_target_labels.drop(zero_point_indexes.index)
        # #stationary_target_labels = stationary_target_labels_no_zero
        # gx = np.expand_dims(stationary_target_labels['vec_3d_x'].to_numpy(), axis=1)
        # gy = np.expand_dims(stationary_target_labels['vec_3d_y'].to_numpy(), axis=1)
        # gz = np.expand_dims(stationary_target_labels['vec_3d_z'].to_numpy(), axis=1)
        #
        # points = np.concatenate((gx, gy, gz), axis=1)
        # #points = np.concatenate((gx, gy), axis=1)
        # indexes = self.farthest_point_sampling(points, 17)
        # training_set = stationary_target_labels.iloc[indexes]
        # #validation_set = stationary_target_labels.sample(n=val_size, random_state=42)
        # validation_set = stationary_target_labels.drop(training_set.index)
        # #self.training_set = pd.concat([moving_target_labels,  training_set])
        # self.training_set = training_set
        # self.validation_set = validation_set
        #
        # print('Training set size: {}, Validation set size: {}'.format(len(self.training_set), len(self.validation_set)))






    # def read_ts_labels(self):
    #     all_meta_files = self.find_files(self.data_dir, 'metadata.txt')
    #     data_lines = []
    #     for file in all_meta_files:
    #         loop_number = os.path.dirname(file)
    #         session_number = os.path.dirname(loop_number)
    #         person_dir = os.path.dirname(session_number)
    #
    #         loop_number = int(os.path.basename(loop_number))
    #         session_number = int(os.path.basename(session_number))
    #         person_id = os.path.basename(person_dir)[0:5]
    #         dir_path = os.path.dirname(file)
    #         with open(file, 'r') as f:
    #             lines = f.readlines()
    #             for line in lines:
    #                 ll = line.split(' ')
    #                 frame_id_L = os.path.join(dir_path, ll[0] + '_L.png')
    #                 frame_id_R = os.path.join(dir_path, ll[0] + '_R_fliped.png')
    #                 gridx = float(ll[4])
    #                 gridy = float(ll[5])
    #                 data_lines.append([person_id, session_number, loop_number,frame_id_L, frame_id_R, gridx, gridy ])
    #
    #     all_data = pd.DataFrame(data_lines, columns=['person_id', 'session_number', 'loop_number', 'frame_id_L', 'frame_id_R', 'gridx',
    #                                      'gridy'])
    #     gridx = all_data['gridx'].to_numpy()
    #     gridy = all_data['gridy'].to_numpy()
    #     gaze_azimouth, gaze_altitude = self.grid_to_angles(gridx, gridy)
    #     vector_3d = self.grid_to_vec(gridx, gridy)
    #     #x, y = self.angles_to_grid(gaze_azimouth, gaze_altitude)
    #     all_data['gaze_azimouth'] = gaze_azimouth
    #     all_data['gaze_altitude'] = gaze_altitude
    #     all_data['por_3d_vec'] = vector_3d.tolist()
    #     user_id = 'id001'
    #     #all_data = all_data.loc[all_data['eye'] == self.eye_type]
    #     #all_data = all_data.loc[(all_data['subject_id'] == self.subject_id)]
    #
    #     training_data = all_data.loc[all_data['person_id'].isin([user_id])]
    #     validation_data = all_data.loc[all_data['person_id'].isin([user_id])]
    #
    #     self.training_set = training_data
    #     self.val_set = validation_data
    #         # On adapter mode the training set is sampled from multiple subjects.
    #
    #     gt = torch.tensor(vector_3d)
    #     gt = gt.to(torch.float32)
    #     #fps_indexes = random.sample(range(len(gt)), self.training_set_size)
    #     fps_indexes = range(len(gt))
    #     #fps_indexes = self.farthest_point_sampling(gt, self.training_set_size)
    #     fps_gt_x = training_data['gridx'].to_numpy()[fps_indexes, np.newaxis]
    #     fps_gt_y = training_data['gridy'].to_numpy()[fps_indexes, np.newaxis]
    #     gt_fps = np.concatenate((fps_gt_x, fps_gt_y), axis=1)
    #     train_data_for_person = training_data[training_data[['gridx', 'gridy']].apply(list, axis=1).isin(gt_fps.tolist())]
    #     val_data_for_person = validation_data.drop(train_data_for_person.index)
    #     self.training_set = train_data_for_person
    #     self.val_set = val_data_for_person
    #
    #     print('Training set size: {}, Validation set size: {}'.format(len(self.training_set), len(self.val_set)))
    #
    #
    #     return all_data


    # def split_train_validation(self, train_fraction=0.1):
    #     """
    #     train_fraction: the fraction of all data assigned to the trainig set. The rest (1-train_fraction) is
    #     assigned to the validation set
    #     """
    #
    #     self.training_set = self.nvgaze_labels.sample(self.training_set_size, random_state=42)
    #     self.val_set = self.nvgaze_labels.drop(self.training_set.index)

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

    # def farthest_point_sampling(self, points, num_points):
    #     """
    #     Performs farthest point sampling on a set of 3D points using cosine distance.
    #
    #     Args:
    #     - points (numpy array of shape (N, 3)): the set of points to sample from
    #     - num_points (int): the number of points to sample
    #
    #     Returns:
    #     - indices (numpy array of shape (num_points,)): the indices of the sampled points in the original set
    #     """
    #     points = normalize(points, axis=1)
    #     indices = [[np.argmax(points[:,0])][0]]
    #     #indices = [0]
    #
    #     # Compute the cosine distances between the first point and all the other points
    #     cos_dists = 1 - np.dot(points[indices[0]], points.T)
    #
    #     # Iterate until the desired number of points is reached
    #     while len(indices) < num_points:
    #         # Find the index of the point with the maximum cosine distance to the current set of sampled points
    #         max_index = np.argmax(cos_dists)
    #
    #         # Add the index to the list of sampled indices
    #         indices.append(max_index)
    #
    #         # Compute the cosine distances between the newly added point and all the other points
    #         cos_dists_new = 1 - np.dot(points[max_index], points.T)
    #
    #         # Update the cosine distances to be the minimum between the current and the new distances
    #         cos_dists = np.minimum(cos_dists, cos_dists_new)
    #
    #     # Convert the list of sampled indices to a numpy array
    #     indices = np.array(indices)
    #
    #     return indices



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

        #gil - all validation data
        self.val_data_for_person = all_data.drop(self.train_data_for_person.index)
        #self.val_data_for_person = all_data

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
            self.ts_dataset_train = TSSinglePersonDataset(self.hparams.args, self.data_dir, self.training_set,stage='train')
            self.ts_dataset_val = TSSinglePersonDataset(self.hparams.args, self.data_dir, self.validation_set,stage='validate')

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass
            #self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "validate":
            self.ts_dataset_val = TSSinglePersonDataset(self.hparams.args, self.data_dir, self.validation_set,
                                                        stage='validate')

        if stage == "predict":
            self.ts_dataset_predict = TSSinglePersonDataset(self.hparams.args, self.hparams.args.input_dir, stage='predict')


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

        if self.hparams.args.batch_per_person:
            return DataLoader(self.ts_dataset_val, batch_sampler=self.per_person_batch_val_sampler,
                              num_workers=self.hparams.args.num_workers)
        else:
            return DataLoader(self.ts_dataset_val, batch_size=self.hparams.args.val_batch_size,
                          num_workers=self.hparams.args.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ts_dataset_val, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.ts_dataset_predict, batch_size=1)
