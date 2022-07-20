# only mapping sematic features from S to V
# using the cosine similarity

import argparse
import os
import shutil
import sys
import time

import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import dataset
import plain_net
from utils import AverageMeter
from utils.logger import Logger
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='5', help='GPU to use [default: GPU 0]')
parser.add_argument('--logdir', default='GZSL-SZSL_wo_C', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 500]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--loss', default='log', help='Loss function [defaultL l2]')
parser.add_argument('--seeds', type=int, default=2)
parser.add_argument('--lamda', type=float, default=0.3)
parser.add_argument('--rho1', type=float, default=40)
parser.add_argument('--rho2', type=float, default=6)
parser.add_argument('--rho3', type=float, default=30)
parser.add_argument('--Hunseen', dest='Hunseen', action='store_true', default=True)
parser.add_argument('--NoNorm', dest='NORM', action='store_false', default=True)

FLAGS = parser.parse_args()

SEED = FLAGS.seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

DATASET_CLASSES = 50
TRAIN_CLASSES = 40
TEST_CLASSES = 10
DATASET_ATTRIBUTES = 85  # dimention

MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
# DECAY_STEP = FLAGS.decay_step
# DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.logdir
LOSS = FLAGS.loss
RHO1 = FLAGS.rho1
RHO2 = FLAGS.rho2
RHO3 = FLAGS.rho3
Hunseen = FLAGS.Hunseen
# Dunseen = FLAGS.Dunseen
NORM = FLAGS.NORM
lamda = FLAGS.lamda

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

data_path = 'AWA2'

NUM_GPU = len(GPU_INDEX.split(','))
batch_size = FLAGS.batch_size
BATCH_SIZE = FLAGS.batch_size * NUM_GPU
TEST_BS = 16
print ('Batch Size = %d' % BATCH_SIZE)
name_file = sys.argv[0]

SEEN_IDS = np.loadtxt(data_path + '/train_ids.txt').astype(np.int32).tolist()
UNSEEN_IDS = np.loadtxt(data_path + '/test_ids.txt').astype(np.int32).tolist()

#######################################################################
if not os.path.exists(LOG_DIR.strip().split('/')[0]):
    os.mkdir(LOG_DIR.strip().split('/')[0])

if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)

os.mkdir(LOG_DIR)
#######################################################################

os.system('cp %s %s' % (name_file, LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
print (str(FLAGS))

logger = Logger(LOG_DIR + '/logs')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def main():
    # create model
    model = plain_net.AttributeNetwork(DATASET_ATTRIBUTES, 1024, 2048, SEED)
    model.train().cuda()

    # model = nn.DataParallel(model)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print ('Model is ready to Go !')

    # define loss function (criterion) and optimizer
    # criterion = nn.MSELoss(size_average=True).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    if OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), BASE_LEARNING_RATE, 0.9, weight_decay=0.000, nesterov=True)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), BASE_LEARNING_RATE, weight_decay=0.0001)
    elif OPTIMIZER == 'rmsp':
        optimizer = torch.optim.RMSprop(model.parameters(), BASE_LEARNING_RATE, weight_decay=0.00)
    print ('Loss func and Optimizer are ready to Go !')

    # Data loading code
    data1 = dataset.dataset(data_path, 'train')
    data2 = dataset.dataset(data_path, 'test_seen')
    data3 = dataset.dataset(data_path, 'test_unseen')
    train_loader = DataLoader(data1, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
    test_seen_loader = DataLoader(data2, batch_size=TEST_BS, num_workers=1, shuffle=False)
    test_unseen_loader = DataLoader(data3, batch_size=TEST_BS, num_workers=1, shuffle=False)
    print ('Data is ready to Go !')

    # prepare for the search space
    attributes = sio.loadmat(data_path + '/att_splits.mat')['att'].T

    train_ids = np.loadtxt(data_path + '/train_ids.txt').reshape(-1, 1).astype(np.uint16)
    train_attributes = attributes[train_ids].squeeze()

    test_ids = np.loadtxt(data_path + '/test_ids.txt').reshape(-1, 1).astype(np.uint16)
    test_attributes = attributes[test_ids].squeeze()

    epoch_id = 1
    epoch_id_best = 0
    best_acc = 0
    best_acc_GZSL = 0
    while True:
        # # #adjust_learning_rate
        #    if epoch_id % 15 == 0 and optimizer.param_groups[0]['lr'] > BASE_LEARNING_RATE/1000000:
        #      optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

        # test for every five epoches
        if epoch_id % 1 == 0:
            log_string('Validating at Epoch %d ---------------------------------' % (epoch_id))
            zsl_acc = test(test_unseen_loader, test_ids, test_attributes, model, criterion, 'zsl')
            gzsl_unseen_acc = test(test_unseen_loader, np.arange(DATASET_CLASSES), attributes, model, criterion,
                                   'gzsl_unseen')
            gzsl_seen_acc = test(test_seen_loader, np.arange(DATASET_CLASSES), attributes, model, criterion,
                                 'gzsl_seen')
            H = 2 * gzsl_seen_acc * gzsl_unseen_acc / (gzsl_unseen_acc + gzsl_seen_acc)
            log_string('zsl=%.4f' % (zsl_acc))
            log_string('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_acc, gzsl_unseen_acc, H))

            # remember best acc and save model
            if zsl_acc > best_acc:
                best_acc = zsl_acc
                epoch_id_best = epoch_id - 1
                torch.save(model.state_dict(), LOG_DIR + '/best' + str(epoch_id_best) + '.pth')
                log_string('**************There is the best net!!!*************')
            if H > best_acc_GZSL:
                best_acc_GZSL = H
                epoch_id_best_GZSL = epoch_id - 1
                torch.save(model.state_dict(), LOG_DIR + '/GZSL_best' + str(epoch_id_best_GZSL) + '.pth')
                log_string('There is the best net!!!')

            # info_test = {'zsl_acc': zsl_acc.item(),
            #              'gzsl_seen_acc': gzsl_seen_acc.item(),
            #              'gzsl_unseen_acc': gzsl_unseen_acc.item(),
            #              'H': H}
            # for tag, value in info_test.items():
            #     logger.scalar_summary(tag, value, epoch_id)

        # train for one epoch
        train_loss, train_acc, epoch_time = train(train_loader,
                            train_ids, train_attributes,  attributes, model, criterion, optimizer)
        log_string('[Epoch %d, epoch_time is %2.2f, loss is %.10f, mean_accuracy is %.4f, lr: %.10f]'
                   % (epoch_id, epoch_time, train_loss, train_acc, optimizer.param_groups[0]['lr']))

        # the max epoch
        if epoch_id == MAX_EPOCH+1:
            break
        epoch_id += 1
        info = {'train_loss': train_loss,
                'train_accuracy': train_acc.item()}

def train(train_loader, search_ids, search_attributes, regular_attributes, model, criterion, optimizer):
    end = time.time()

    # switch to train mode
    model.train()

    losses = AverageMeter()

    # prepare the search space
    search_attributes = Variable(torch.from_numpy(search_attributes)).cuda().float()
    regular_attributes = Variable(torch.from_numpy(regular_attributes)).cuda().float()

    # prepare the accuracy
    rewards = torch.zeros(search_ids.shape).cuda()
    num = torch.zeros(search_ids.shape).cuda()

    for batch_features, batch_labels in train_loader:

        # prepare the batch features and labels
        batch_labels = Variable(batch_labels).cuda().float()
        batch_size = batch_labels.shape[0]
        batch_features = Variable(batch_features).cuda().float()

        # re_batch_labels
        re_batch_labels = []
        for label in batch_labels.data.cpu().numpy():
            index = np.argwhere(search_ids == label)
            re_batch_labels.append(index[0][0])
        Groundtruth = torch.LongTensor(re_batch_labels).cuda()

        # pdb.set_trace()
        # compute features from visual space to semantic space
        output, W = model(search_attributes, batch_features, re_batch_labels)
        output2, W2 = model(regular_attributes, batch_features, re_batch_labels)
        Wnorm = W.norm(2, 1, keepdim=True)
        W2norm = W2.norm(2, 1, keepdim=True)
        if NORM:
            output = output / Wnorm.transpose(1, 0).repeat(batch_size, 1)
            output2 = output2 / W2norm.transpose(1, 0).repeat(batch_size, 1)
        # loss functions
        output = output * RHO1
        loss1 = criterion(output, Groundtruth)

        # *********diversity1***********************
        Walign = W / Wnorm
        Gram1 = torch.mm(Walign, Walign.transpose(1, 0))
        Gram2 = torch.mm(search_attributes, search_attributes.transpose(1, 0))

        M = -np.ones([TRAIN_CLASSES,TRAIN_CLASSES])
        M = M - np.diag(M.diagonal()) + np.eye(TRAIN_CLASSES)
        # pdb.set_trace()
        Gram3 = torch.Tensor(M).cuda()

        Gram = (1-lamda)*(Gram1-Gram3) + lamda*(Gram1 - Gram2)
        R = torch.mean(torch.pow(Gram,2))

        # *********diversity2***********************
        if Hunseen:
            # output2 = torch.softmax(output2 * RHO2, 1)
            # H = -torch.mean(torch.log2(output2) * output2)
            seen_output2 = output2[:, SEEN_IDS]
            unseen_output2 = output2[:, UNSEEN_IDS]
            Calibration = torch.cat((seen_output2 * RHO2, unseen_output2 * RHO3), 1)
            Calibration = torch.softmax(Calibration, 1)
            H = -torch.mean(torch.log2(Calibration) * Calibration)
            # loss = loss1 + H + R
        else:
            loss = loss1 + R

        # measure accuracy and record loss
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for accuracy
        _, predict_labels = torch.max(output.data, 1)

        rewards_train = [1 if re_batch_labels[j] == predict_labels[j].item() else 0 for j in range(predict_labels.shape[0])]
        search_rewards = torch.zeros(search_ids.shape)
        search_num = torch.zeros(search_ids.shape)

        for i in range(batch_size):
            label = batch_labels.data.cpu().numpy()[i]
            search_rewards[np.argwhere(search_ids == label)[0][0]] += rewards_train[i]
            search_num[np.argwhere(search_ids == label)[0][0]] += 1
        rewards += search_rewards.cuda()

        num += search_num.cuda()

    accuracy = rewards / (num + 1e-5)
    mean_acc = torch.sum(accuracy) / search_ids.shape[0]
    epoch_time = time.time() - end

    return losses.avg, mean_acc, epoch_time


def test(test_loader, search_ids, search_attributes, model, criterion, str):
    end = time.time()

    # switch to train mode
    model.eval()

    # prepare the search space
    search_attributes = Variable(torch.from_numpy(search_attributes)).cuda().float()

    # prepare the accuracy
    rewards = torch.zeros(search_ids.shape).cuda()
    num = torch.zeros(search_ids.shape).cuda()

    for batch_features, batch_labels in test_loader:

        # prepare the batch features and labels
        batch_size = batch_labels.shape[0]
        batch_labels = Variable(batch_labels).cuda().float()
        batch_features = Variable(batch_features).cuda().float()

        # re_batch_labels
        re_batch_labels = []
        for label in batch_labels.data.cpu().numpy():
            index = np.argwhere(search_ids == label)
            re_batch_labels.append(index[0][0])

        # compute features from visual space to sematic space
        output, W = model(search_attributes, batch_features, re_batch_labels)
        Wnorm = W.norm(2, 1, keepdim=True)

        if NORM:
            output = output / Wnorm.transpose(1, 0).repeat(batch_size, 1)
        output = torch.softmax(output, 1)

        _, predict_labels = torch.max(output.data, 1)
        rewards_train = [1 if re_batch_labels[j] == predict_labels[j].item() else 0 for j in range(predict_labels.shape[0])]

        search_rewards = torch.zeros(search_ids.shape)
        search_num = torch.zeros(search_ids.shape)

        for i in range(batch_size):
            label = batch_labels.data.cpu().numpy()[i]
            search_rewards[np.argwhere(search_ids == label)[0][0]] += rewards_train[i]
            search_num[np.argwhere(search_ids == label)[0][0]] += 1
        rewards += search_rewards.cuda()
        num += search_num.cuda()

    accuracy = rewards / (num + 1e-5)

    if str == 'zsl':
        mean_acc = torch.mean(accuracy)
    if str == 'gzsl_seen':
        mean_acc = torch.sum(accuracy) / TRAIN_CLASSES
    if str == 'gzsl_unseen':
        mean_acc = torch.sum(accuracy) / TEST_CLASSES

    epoch_time = time.time() - end
    return mean_acc


if __name__ == "__main__":
    main()
