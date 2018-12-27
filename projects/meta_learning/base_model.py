import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchsummary import summary

import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from collections import defaultdict
from datetime import datetime
import pickle
import os

DEFAULT_SAVE_DIR = 'drive/Research Projects/Meta-Learning/v1/models'
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_BATCHES_TO_PRINT = 100

DEFAULT_CONV_FILTER_SIZES = (24, 24, 24, 24)
DEFAULT_MLP_LAYER_SIZES = (256, 64, 16, 4)


class BasicModel(nn.Module):
    def __init__(self, name, use_mse=False,
                 save_dir=DEFAULT_SAVE_DIR, mse_threshold=0.5):
        super(BasicModel, self).__init__()

        self.name = name
        self.use_mse = use_mse
        self.mse_threshold = mse_threshold
        self.save_dir = save_dir
        self.optimizer = None
        self._init_dir()

        self.train_accuracies = []
        self.train_losses = []
        self.test_accuracies = []
        self.test_losses = []
        self.train_aucs = []
        self.test_aucs = []

    def train_(self, input_img, input_query, label):
        if self.optimizer is None:
            self._create_optimizer()

        self.optimizer.zero_grad()
        output = self(input_img, input_query)

        if self.use_mse:
            #             target = np.zeros((64, 2))
            #             target[np.arange(output.shape[0]), label.data.numpy()] = 1
            #             target = torch.from_numpy(target).to(label.device)
            #             loss = F.mse_loss(output, target)
            output = torch.squeeze(output)
            loss = F.mse_loss(output, label.to(torch.float))
            pred = output.data > self.mse_threshold
            pred = pred.to(torch.long)

        else:
            loss = F.nll_loss(output, label)
            pred = output.data.max(1)[1]

        loss.backward()
        self.optimizer.step()

        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        auc = roc_auc_score(label.data.cpu().numpy(), pred.cpu().numpy())

        return accuracy, loss.item(), auc

    def test_(self, input_img, input_query, label, training=False):
        output = self(input_img, input_query)

        if self.use_mse:
            #             target = np.zeros((64, 2))
            #             target[np.arange(output.shape[0]), label.data.numpy()] = 1
            #             target = torch.from_numpy(target).to(label.device)
            #             loss = F.mse_loss(output, target)
            output = torch.squeeze(output)
            loss = F.mse_loss(output, label.to(torch.float))
            pred = output.data > self.mse_threshold
            pred = pred.to(torch.long)


        else:
            loss = F.nll_loss(output, label)
            pred = output.data.max(1)[1]

        correct = pred.eq(label.data).cpu()
        accuracy = correct.sum() * 100. / len(label)
        auc = roc_auc_score(label.data.cpu().numpy(), pred.cpu().numpy())

        return accuracy, loss.item(), auc, correct

    def save_model(self, epoch=None, **kwargs):
        if epoch is None:
            epoch = len(self.train_accuracies)

        torch.save(self.state_dict(), self._save_path(epoch))

        results = dict(epoch=epoch,
                       train_accuracies=self.train_accuracies,
                       train_losses=self.train_losses,
                       test_accuracies=self.test_accuracies,
                       test_losses=self.test_losses,
                       train_aucs=self.train_aucs,
                       test_aucs=self.test_aucs,
                       **kwargs)

        with open(f'{self._save_dir()}/results.pickle', 'wb') as f:
            pickle.dump(results, f)

    def load_model(self, epoch=None):
        if epoch == 0:
            print('Warning: asked to load model with epoch 0. Ignoring...')
            return

        with open(f'{self._save_dir()}/results.pickle', 'rb') as f:
            results = pickle.load(f)

        self.train_accuracies = results.pop('train_accuracies')
        self.train_losses = results.pop('train_losses')
        self.test_accuracies = results.pop('test_accuracies')
        self.test_losses = results.pop('test_losses')
        self.train_aucs = results.pop('train_aucs')
        self.test_aucs = results.pop('test_aucs')

        if epoch is None:
            epoch = len(self.train_accuracies)

        self.load_state_dict(torch.load(self._save_path(epoch)))

        return results

    def _save_path(self, epoch):
        return f'{self._save_dir()}/epoch_{epoch:02d}.pth'

    def _save_dir(self):
        return f'{self.save_dir}/{self.name}'

    def _init_dir(self):
        os.makedirs(self._save_dir(), exist_ok=True)

        print(os.system(f'ls -laR {self._save_dir()}'))

    def _create_optimizer(self):
        raise NotImplementedError()

    def post_test(self, test_loss):
        pass


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def train_epoch(model, dataloader, cuda=True,
                num_batches_to_print=DEFAULT_NUM_BATCHES_TO_PRINT):
    accuracies = []
    losses = []
    aucs = []
    for b, ((X, Q, y), indices) in enumerate(dataloader):
        if cuda:
            X = X.cuda()
            Q = Q.cuda()
            y = y.cuda()

        images = Variable(X)
        queries = Variable(Q).float()
        labels = Variable(y).long()

        acc, loss, auc = model.train_(images, queries, labels)
        accuracies.append(acc)
        losses.append(loss)
        aucs.append(auc)

        if (b + 1) % num_batches_to_print == 0:
            print(
                f'{now()}: After batch {b + 1}, average acc is {np.mean(accuracies):.3f} and average loss is {np.mean(losses):.3f}')

    model.train_accuracies.append(np.mean(accuracies))
    model.train_losses.append(np.mean(losses))
    model.train_aucs.append(np.mean(aucs))

    return accuracies, losses, aucs


def test(model, dataloader, cuda=True, num_batches_to_test=None, training=False):
    accuracies = []
    losses = []
    aucs = []
    correct_per_query = defaultdict(list)
    for b, ((X, Q, y), indices) in enumerate(dataloader):
        if cuda:
            X = X.cuda()
            Q = Q.cuda()
            y = y.cuda()

        images = Variable(X)
        queries = Variable(Q).float()
        labels = Variable(y).long()

        acc, loss, auc, correct = model.test_(images, queries, labels, training)
        accuracies.append(acc)
        losses.append(loss)
        aucs.append(auc)

        for query, result in zip(queries.cpu(), correct):
            correct_per_query[np.argmax(query.numpy())].append(result.numpy())

        if (num_batches_to_test is not None) and (b >= num_batches_to_test - 1):
            break

    for query_key in correct_per_query:
        correct_per_query[query_key] = np.array(correct_per_query[query_key])

    model.test_accuracies.append(np.mean(accuracies))
    mean_loss = np.mean(losses)
    model.test_losses.append(mean_loss)
    model.post_test(mean_loss)
    model.test_aucs.append(np.mean(aucs))

    return accuracies, losses, aucs, correct_per_query


def mid_train_plot(model, epochs_to_test):
    plt.figure(figsize=(12, 4))
    epoch = len(model.train_losses)
    plt.suptitle(f'After epoch {epoch}')
    train_x_values = np.arange(1, epoch + 1)
    test_x_values = np.arange(1, len(model.test_losses) + 1) * epochs_to_test

    loss_ax = plt.subplot(1, 3, 1)
    loss_ax.set_title('Loss')
    loss_ax.plot(train_x_values, model.train_losses, label='Train')
    loss_ax.plot(test_x_values, model.test_losses, label='Test')
    loss_ax.legend(loc='best')

    acc_ax = plt.subplot(1, 3, 2)
    acc_ax.set_title('Accuracy')
    acc_ax.plot(train_x_values, model.train_accuracies, label='Train')
    acc_ax.plot(test_x_values, model.test_accuracies, label='Test')
    acc_ax.legend(loc='best')

    auc_ax = plt.subplot(1, 3, 3)
    auc_ax.set_title('AUC')
    auc_ax.plot(train_x_values, model.train_aucs, label='Train')
    auc_ax.plot(test_x_values, model.test_aucs, label='Test')
    auc_ax.legend(loc='best')

    plt.show()


def train(model, train_dataloader, test_dataloader, num_epochs=100,
          epochs_to_test=5, epochs_to_graph=None, cuda=True,
          num_batches_to_print=DEFAULT_NUM_BATCHES_TO_PRINT, save=True,
          num_batches_to_test=None, start_epoch=0):
    if epochs_to_graph is None:
        epochs_to_graph = epochs_to_test

    test_correct_per_query = []

    wandb.watch(model)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_acc, train_loss, train_auc = train_epoch(model, train_dataloader, cuda, num_batches_to_print)
        print(
            f'{now()}: After epoch {epoch}, TRAIN average acc is {np.mean(train_acc):.4f}, average loss is {np.mean(train_loss):.4f}, and average AUC is {np.mean(train_auc):.4f}')

        if save:
            model.save_model(test_correct_per_query=test_correct_per_query)

        if epoch % epochs_to_test == 0:
            test_acc, test_loss, test_auc, test_cpq = test(model, test_dataloader, cuda, num_batches_to_test, True)
            print(
                f'{now()}: After epoch {epoch}, TEST average acc is {np.mean(test_acc):.4f}, average loss is {np.mean(test_loss):.4f}, and average AUC is {np.mean(test_auc):.4f}')
            test_correct_per_query.append(test_cpq)
            wandb.log({'Train Accuracy': np.mean(train_acc),
                       'Train Loss': np.mean(train_loss),
                       'Train AUC': np.mean(train_auc),
                       'Test Accuracy': np.mean(test_acc),
                       'Test Loss': np.mean(test_loss),
                       'Test AUC': np.mean(test_auc)},
                      step=epoch)

        if epoch % epochs_to_graph == 0:
            mid_train_plot(model, epochs_to_test)

    # TODO: implement test-error based stopping

    return test_correct_per_query


