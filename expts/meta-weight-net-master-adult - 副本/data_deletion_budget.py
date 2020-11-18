import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from my_op import MLP
from sklearn import metrics
from load_corrupted_data import Adult_dv

'''
此函数的作用是按照v_list的权重分数，在训练集中去除budget比例的低价值（高价值数据用上面的排序）的数据后，
用剩下的数据集训练一个模型再验证他的性能。
性能
v_list: 数据的权重分数列表
budget: 去除数据的比例，如20%
x_train: 训练集的x
y_train: 训练集的y
args: 各种参数
test_loader: 测试集
'''


def data_deletion(v_list, args, x_train, y_train , valid_loader, test_loader):
    # plt.hist(v_list)
    # plt.show()

    # sort the weight from the lowest to the highest
    index_list = np.argsort(v_list).tolist()


    # ===================1.构建数据集================
    # build the dataset according to the compression_rate
    train_loader_dv = build_dataset_dv(x_train, y_train, index_list, args)

    # ===================2.模型及初始化=======================
    MLP_dv = MLP(108, 250, 250, classes=args.num_classes).cuda()
    for m in MLP_dv.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)

    # ===================3.优化=============================
    DV_optimizer_model = torch.optim.SGD(MLP_dv.parameters(), args.meta_lr,
                                         momentum=args.momentum, weight_decay=args.weight_decay)

    # ===================4.训练==============================
    best_acc = 0
    for epoch in range(args.dv_epochs):
        # adjust_learning_rate(DV_optimizer_model, epoch)
        train_dv(train_loader_dv, MLP_dv, DV_optimizer_model, epoch)
        valid_acc = test(meta_MLP=MLP_dv, test_loader=valid_loader)
        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(MLP_dv.state_dict(), "./MLP_dv.pkl")

    MLP_dv.load_state_dict(torch.load("./MLP_dv.pkl"))
    test_acc = test(meta_MLP=MLP_dv, test_loader=test_loader)



def train_dv(train_loader, model, optimizer_model, epoch):
    print('\nEpoch: %d' % epoch)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()  # 启用 BatchNormalization 和 Dropout
        inputs, targets = inputs.float().cuda(), targets.cuda()  # 将tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        outputs = model(inputs)
        targets = targets.long()
        loss = F.cross_entropy(outputs, targets, reduce=False)
        loss1 = torch.mean(loss)
        # prec_train = accuracy(outputs.data, targets.data)

        optimizer_model.zero_grad()
        loss1.backward()
        optimizer_model.step()
        print("loss:",loss1.data)
        return loss
        # if (batch_idx + 1) % 50 == 0:
        #     print('Epoch: [%d/%d]\t'
        #           'Iters: [%d/%d]\t'
        #           'Loss: %.4f\t'
        #           'Prec@1 %.2f\t' % (
        #               (epoch + 1), epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
        #               (loss / (batch_idx + 1)), prec_train))


def build_dataset_dv(x_data, y_data, index_list, args):
    '''
    function: build the train dataset according to the compression_rate
    '''
    train_data_dv = Adult_dv(x_data, y_data, index_list, args.compression_rate)
    train_loader_dv = torch.utils.data.DataLoader(
        train_data_dv, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)

    return train_loader_dv


def accuracy(outputs, targets):
    output = np.array(outputs.cpu())
    target = np.array(targets.cpu())
    tmp = []
    for a in output:
        if a[0] < 0.5:
            tmp.append(1)
        else:
            tmp.append(0)

    return metrics.accuracy_score(np.array(tmp), target)


def test(meta_MLP, test_loader):
    meta_MLP.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.float().cuda(), targets.cuda()
            outputs = meta_MLP(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy
