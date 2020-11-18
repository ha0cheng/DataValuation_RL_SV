# -*- coding: utf-8 -*-
import data_loading
import argparse
import os
import shutil
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from my_op import MLP
from resnet import  weight_net_scale
from load_corrupted_data import  Adult
from data_deletion_budget import data_deletion
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Weight Training')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--dv_epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--batch_size', '--batch-size', default=1000, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--meta_lr', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_lr', default=1e-3, type=float,
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--compression_rate', type=float, default=0.5)
parser.add_argument('--alpha_start', default=0.1, type=float)
parser.add_argument('--alpha_end', default=1000, type=float)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)


args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

print(args)

writer = SummaryWriter(comment='meta', filename_suffix="test_meta")
def build_dataset(x_train, y_train, x_valid, y_valid, x_test, y_test):

    '''
    Function: build the data loader of train, valid, and test.
    input: the (x,y) data of train, valid, and test
    output:  train_loader, valid_loader, test_loader, train_data, and train_loader_noshuffle
    '''

    train_data = Adult(x_train, y_train)
    valid_data = Adult(x_valid, y_valid)
    test_data = Adult(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)
    train_loader_Noshuffle = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
    return  train_loader, valid_loader, test_loader, train_data, train_loader_Noshuffle






def test(meta_MLP, test_loader):
    '''
    Test the performance of model meta_MLP using valid/test dataset
    '''

    meta_MLP.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.float().to(device), targets.to(device)
            outputs = meta_MLP(inputs)
            test_loss +=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy

def update_params(fc, lr_inner, source_params=None):

    for i, j in zip(fc.parameters(), source_params):
        i.data.sub_(lr_inner * j.data)

def train(train_loader, valid_loader, meta_MLP, weight_net, meta_optimizer, weight_net_optimizer, epoch):
    global alpha_step, alpha
    print('\nEpoch: %d' % epoch)

    max_weight_net_loss = -10000

    valid_loader_iter = iter(valid_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        weight_net.scale_factor = weight_net.scale_factor + alpha_step
        meta_MLP.train()
        weight_net.train()


#==========================================================================

#clone the meta_MLP and make a fake upgrade by using the weighted loss in the train dataset
        inputs, targets = inputs.float().to(device), targets.to(device)
        clone_meta_MLP = MLP(108, 250, 250, classes=args.num_classes).cuda()
        clone_meta_MLP.load_state_dict(meta_MLP.state_dict())

# forward
        outputs = clone_meta_MLP(inputs)

#compute the loss
        clone_cost = F.cross_entropy(outputs, targets, reduce=False)
        clone_cost_v = torch.reshape(clone_cost, (len(clone_cost), 1))

# compute the weight and weighted_loss
        weight_net_input = torch.cat((inputs, targets.float().reshape(inputs.shape[0], 1),clone_cost_v), dim=1)
        clone_weight = weight_net(weight_net_input.data)
        clone_weighted_loss = torch.sum(clone_cost_v * clone_weight)/len(clone_cost_v)

# upgrade the clone_meta_MLP
        clone_meta_MLP.zero_grad()
        grads = torch.autograd.grad(clone_weighted_loss, (clone_meta_MLP.parameters()), create_graph=True)
        meta_lr = args.meta_lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))
        update_params(clone_meta_MLP, lr_inner=meta_lr, source_params=grads) # upgrade by hand
        del grads


#=====================================================================================================

# upgrade the weight_net by using the non-weighted loss in the valid dataset
        try:
            inputs_val, targets_val = next(valid_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(valid_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = clone_meta_MLP(inputs_val.float())
        targets_val = targets_val.long()

# cross_entropy use the format of (1-p), to maximize the loss
        cost = F.cross_entropy(1-y_g_hat, targets_val, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        loss1 = torch.sum(cost_v)/len(cost_v)
        loss2 = (clone_weight.norm(1) / float(clone_weight.size(0)) - args.compression_rate) ** 2
        weight_net_loss = loss1 + 10*loss2

# upgrade the weight_net
        weight_net_optimizer.zero_grad()
        weight_net_loss.backward()
        weight_net_optimizer.step()

#========================更新predict model================================================================
        outputs = meta_MLP(inputs.float())
        meta_cost = F.cross_entropy(outputs, targets, reduce=False)
        meta_cost_v = torch.reshape(meta_cost, (len(meta_cost), 1))


        with torch.no_grad():
            meta_weight = weight_net(weight_net_input)

#compute the two_side_rate, which is the percent of weight (>0.9 or <1), tmp is the weight tensor
        tmp = meta_weight.data.cpu().numpy()
        two_side_rate = (np.sum(tmp > 0.9) + np.sum(tmp < 0.1))/len(tmp)

        if (weight_net_loss > max_weight_net_loss) and (two_side_rate >= 0):
            max_weight_net_loss = weight_net_loss
            weight_list = tmp

#weighted loss function
        meta_loss = torch.sum(meta_cost_v * meta_weight)/len(meta_cost_v)

# upgrade the meta_MLP
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()


        writer.add_scalars('loss',{"clone_loss": clone_weighted_loss.item(),
                                                "weight_net_loss": weight_net_loss.item(),
                                                "meta_loss": meta_loss.item()}, epoch)

        if epoch % 50 ==0:
            plt.hist(tmp)
            plt.show()


        print('Epoch: [%d/%d]\t'
              'Iters: [%d/%d]\t'
              'Loss: %.4f\t'
              'MetaLoss:%.4f\t'
              'Two side rate:%.f' % (
                  (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, weight_net_loss,
                  meta_loss, two_side_rate*100))

    return weight_list




def main():

#-----------------------------data----------------------------------------

    # Data loading, refer to DVRL
    # The number of training and validation samples
    dict_no = dict()
    dict_no['train'] = 1000
    dict_no['valid'] = 400

    # _ = data_loading.load_tabular_data(data_name, dict_no, 0.2)
    # print('Finished data loading.')

    # Data preprocessing
    # Normalization methods: 'minmax' or 'standard'
    normalization = 'minmax'

    # Extracts features and labels. Then, normalizes features
    x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = \
        data_loading.preprocess_data(normalization, 'train.csv',
                                     'valid.csv', 'test.csv')


    train_loader, valid_loader, test_loader, train_data, train_loader_Noshuffle = build_dataset(x_train, y_train, x_valid, y_valid, x_test, y_test)

    print('Finished data preprocess.')



#-------------------------------------2.模型-----------------------------------

# alpha: sigmoid function's scale factor;
# alpha_start: the length of step which alpha adds

    global alpha_step, alpha
    alpha = args.alpha_start
    alpha_step = (args.alpha_end - args.alpha_start) / float(args.epochs * len(train_loader))

# meta_MLP: a 3-layer multiperceptron for classification
# weight_net: a 3-layer multipercetion to compute the weight of data
    args.num_classes = 2
    meta_MLP = MLP(108, 250, 250, classes=args.num_classes).cuda()
    weight_net = weight_net_scale(110, 200, 200, 1, scale_factor=alpha).cuda()


#-------------------------------------3.损失函数--------------------------------


#-------------------------------------4.优化器----------------------------------

# Define the optimizer for meta_MLP and weight_net

    meta_optimizer = torch.optim.SGD(meta_MLP.parameters(), args.meta_lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    weight_net_optimizer = torch.optim.Adam(weight_net.params(), args.weight_lr,
                             weight_decay=1e-4)

#-------------------------------------5.训练------------------------------------

# Start training

    best_acc = 0
    print('start training')
    for epoch in range(args.epochs):
        value_list = train(train_loader, valid_loader, meta_MLP, weight_net, meta_optimizer, weight_net_optimizer, epoch)
        valid_acc = test(meta_MLP=meta_MLP, test_loader=valid_loader)
        print(valid_acc)

        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(weight_net.state_dict(), "./model/weight_net.pkl")
            torch.save(meta_MLP.state_dict(), "./model/meta_MLP.pkl")

    writer.close()
    meta_MLP.load_state_dict(torch.load("./model/meta_MLP.pkl"))
    test_acc = test(meta_MLP=meta_MLP, test_loader=test_loader)


#----------------------------6.删除数据效果评估--------------------------------------
    # data deletion experiment, reserve args.compression_rate of data, do the training and see the result

    # random.shuffle(value_list)
    data_deletion(np.squeeze(value_list), args, x_train, y_train, valid_loader,test_loader)
    print('best validation accuracy:', test_acc)
    print('test accuracy:', best_acc)


if __name__ == '__main__':
    main()
