import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
from torch.nn import Softmax
from tools.utils import Log
import time


class Trainer(object):

    def __init__(self, model1, model2, train_loader, val_loader, optimizer1, optimizer2, model_name, log_dir,
                 num_classes, num_views=12):
        self.num_classes = num_classes
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.model1 = model1
        self.model2 = model2
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.softmax_layer = Softmax(dim=1)
        self.log = Log()
        self.rank_criterion = nn.MarginRankingLoss(margin=0.05)
        self.hinge = nn.MultiMarginLoss()
        self.model1.cuda()
        self.model2.cuda()
        rank_criterion = nn.MarginRankingLoss(margin=0.1)
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)

    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model1.train()
        self.model2.train()
        for epoch in range(n_epochs):
            lr1 = self.optimizer1.state_dict()['param_groups'][0]['lr']
            lr2 = self.optimizer2.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr1', lr1, epoch)
            self.writer.add_scalar('params/lr2', lr2, epoch)

            for i, data in enumerate(self.train_loader):
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(N * V, C, H, W).cuda()
                target1 = Variable(data[0].reshape(N, 1).expand(N, V).reshape(N * V)).cuda().long()
                target2 = Variable(data[0]).cuda().long()
                un_norm_views_cls_tokens, key_parts, global_loss1, global_cls_token1, global_cls_result1 = self.model1(
                    in_data, target1)
                global_loss2, global_cls_result2 = self.model2(un_norm_views_cls_tokens, key_parts, global_cls_token1,
                                                               target2)
                self.writer.add_scalar('train/train_loss', global_loss2, i_acc + i + 1)
                pred = torch.max(global_cls_result2, 1)[1]
                results = pred == target2
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                loss = 0.3*global_loss1 + global_loss2
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()


                log_str = 'epoch %d, step %d: loss %.3f; global_loss1 %.3f; global_loss2 %.3f; ' \
                          'train_acc %.3f' % (epoch + 1, i + 1, loss, 0.3*global_loss1,
                                                              global_loss2, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i

            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    global_loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', global_loss, epoch + 1)
                
            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model1.save(self.log_dir + "_model1", epoch)
                self.model2.save(self.log_dir + "_model2", epoch)

            # adjust learning rate manually 自动调整学习率
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group1 in self.optimizer1.param_groups:
                    param_group1['lr'] = param_group1['lr'] * 0.5
                for param_group2 in self.optimizer2.param_groups:
                    param_group2['lr'] = param_group2['lr'] * 0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        wrong_class = np.zeros(self.num_classes)
        samples_class = np.zeros(self.num_classes)
        all_loss = 0
        self.model1.eval()
        self.model2.eval()
        for _, data in enumerate(self.val_loader, 0):
            N, V, C, H, W = data[1].size()
            in_data = Variable(data[1]).view(N * V, C, H, W).cuda()
            target1 = Variable(data[0].expand(V, -1).contiguous()).view(N * V).cuda().long()
            target2 = Variable(data[0]).cuda().long()
            un_norm_views_cls_tokens, key_parts, global_loss1, global_cls_token1, global_cls_result1 = self.model1(
                in_data, target1)
            global_loss2, global_cls_result2 = self.model2(un_norm_views_cls_tokens, key_parts, global_cls_token1,
                                                           target2)
            pred = torch.max(global_cls_result2, 1)[1]
            all_loss += global_loss2
            results = pred == target2

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target2.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target2.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        self.log.save_test_info('ModelNet40', epoch, acc, loss, val_mean_class_acc)
        self.model1.train()
        self.model2.train()

        return loss, val_overall_acc, val_mean_class_acc
