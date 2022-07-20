import scipy.io as sio
import os
import numpy as np
import torch
import torch.utils.data as data

import pdb

class dataset(data.Dataset):
    def __init__(self, dataset_path, subset):
        self.dataset_path = dataset_path
        self.subset = subset

        self.feature_path = os.path.join(dataset_path, 'res101.mat')
        self.att_splits = os.path.join(dataset_path, 'att_splits.mat')

        features_labels = sio.loadmat(self.feature_path)
        self.features = features_labels['features']
        self.features = self.features.astype(float)
        self.features = self.features.transpose(1, 0)

        self.labels = features_labels['labels'] - 1 #classes index start from 0
        self.labels = self.labels.astype(int)

        att_splits = sio.loadmat(self.att_splits)
        self.train = att_splits['trainval_loc'] - 1  #samples index start from 0
        self.test_seen = att_splits['test_seen_loc'] - 1
        self.test_unseen = att_splits['test_unseen_loc'] - 1

        self.train_features = self.features[self.train].squeeze()
        mean = np.mean(self.train_features, 0).reshape(1, -1)
        self.train_features = self.train_features - mean
        self.train_labels = self.labels[self.train].squeeze()

        self.test_seen_features = self.features[self.test_seen].squeeze()
        self.test_seen_features = self.test_seen_features - mean
        self.test_seen_labels = self.labels[self.test_seen].squeeze()

        self.test_unseen_features = self.features[self.test_unseen].squeeze()
        self.test_unseen_features = self.test_unseen_features - mean
        self.test_unseen_labels = self.labels[self.test_unseen].squeeze()

        # test_ids = np.unique(self.test_unseen_labels).reshape(-1,1)
        # test_ids = test_ids.astype(int)
        # np.savetxt("AWA1/test_ids.txt", test_ids)

        # train_ids = np.unique(self.test_seen_labels).reshape(-1,1)
        # train_ids = train_ids.astype(int)
        # np.savetxt("AWA1/train_ids.txt", train_ids)
        # pdb.set_trace()
        # print (np.mean(self.features.transpose(1, 0), 1))
        # print (np.mean(self.train_features.transpose(1,0), 1))
        # print (np.mean(self.test_seen_features.transpose(1,0), 1))
        # print (np.mean(self.test_unseen_features.transpose(1,0), 1))
        # pdb.set_trace()
        # np.save("test_unseen_features.npy", self.test_unseen_features)
        # np.save("test_unseen_labels.npy", self.test_unseen_labels)

    def __getitem__(self, index):

        if self.subset == 'train':
            features = self.train_features[index]
            labels = self.train_labels[index]         
        if self.subset == 'test_seen':
            features = self.test_seen_features[index]
            labels = self.test_seen_labels[index] 
        if self.subset == 'test_unseen':
            features = self.test_unseen_features[index]
            labels = self.test_unseen_labels[index]

        norm = np.linalg.norm(features, ord=2, axis=0, keepdims=True)
        features = features / np.repeat(norm, features.shape[0], 0)

        return features, labels

    def __len__(self):

        if self.subset == 'train':
            return self.train_features.shape[0]
        if self.subset == 'test_seen':
            return self.test_seen_features.shape[0]
        if self.subset == 'test_unseen':
            return self.test_unseen_features.shape[0]