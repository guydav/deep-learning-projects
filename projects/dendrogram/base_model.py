import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from collections import defaultdict
from datetime import datetime
import pickle
import os


DEFAULT_SAVE_DIR = 'drive/Research Projects/DendrogramLoss/models'
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_BATCHES_TO_PRINT = 100


class BasicModel(nn.Module):
    def __init__(self, name, should_super_init=True, use_mse=False,
                 save_dir=DEFAULT_SAVE_DIR, mse_threshold=0.5, num_classes=2,
                 loss=None):
        if should_super_init:
            super(BasicModel, self).__init__()

        self.name = name
        self.use_mse = use_mse
        self.mse_threshold = mse_threshold
        self.num_classes = num_classes
        self.multiclass = num_classes != 2
        if self.multiclass:
            self.mlb = MultiLabelBinarizer(np.arange(self.num_classes))

        if loss is not None:
            self.loss = loss
        elif use_mse:
            self.loss = F.mse_loss
        else:
            self.loss = F.nll_loss

        self.save_dir = save_dir
        self.optimizer = None
        self._init_dir()

        self.results = defaultdict(list)

        self.train_accuracies = []
        self.train_losses = []
        self.test_accuracies = []
        self.test_losses = []
        self.train_aucs = []
        self.test_aucs = []
        self.train_correct_rank = []
        self.test_correct_rank = []

    def train_(self, input_img, label):
        if self.optimizer is None:
            self._create_optimizer()

        self.optimizer.zero_grad()
        output = self(input_img)

        np_labels = label.data.cpu().numpy()
        if self.multiclass:
            multiclass_labels = self.mlb.fit_transform(np.expand_dims(np_labels, 1))

        if self.use_mse:
            # Per-class output in multiclass, softmax activation
            if self.multiclass:
                tensor_labels = torch.from_numpy(multiclass_labels).float().to(output.device)
                loss = self.loss(output, tensor_labels)
                pred = output.data.max(1)[1]

            # Single output unit in the two-class case, sigmoid activation
            else:
                output = torch.squeeze(output)
                loss = self.loss(output, label.to(torch.float))
                pred = output.data > self.mse_threshold
                pred = pred.to(torch.long)

        else:
            loss = self.loss(output, label)
            pred = output.data.max(1)[1]

        loss.backward()
        self.optimizer.step()

        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)

        # roc_auc score can fail if the entire batch has the same class
        try:
            np_predictions = pred.cpu().numpy()
            if self.multiclass:
                multiclass_predictions = self.mlb.fit_transform(np.expand_dims(np_predictions, 1))
                auc = roc_auc_score(multiclass_labels, multiclass_predictions)

            else:
                auc = roc_auc_score(np_labels, np_predictions)

        except ValueError:
            auc = None

        answer_indices = self.num_classes - np.argsort(np.argsort(output.data.cpu().numpy(), 1), 1)
        correct_rank = answer_indices[np.arange(output.shape[0]), np_labels]

        return accuracy, loss.item(), auc, correct_rank, pred

    def test_(self, input_img, label, training=False):
        output = self(input_img)

        np_labels = label.data.cpu().numpy()
        if self.multiclass:
            multiclass_labels = self.mlb.fit_transform(np.expand_dims(np_labels, 1))

        if self.use_mse:
            # Per-class output in multiclass, softmax activation
            if self.multiclass:
                tensor_labels = torch.from_numpy(multiclass_labels).float().to(output.device)
                loss = self.loss(output, tensor_labels)
                pred = output.data.max(1)[1]

            # Single output unit in the two-class case, sigmoid activation
            else:
                output = torch.squeeze(output)
                loss = self.loss(output, label.to(torch.float))
                pred = output.data > self.mse_threshold
                pred = pred.to(torch.long)

        else:
            loss = self.loss(output, label)
            pred = output.data.max(1)[1]

        correct = pred.eq(label.data).cpu()
        accuracy = correct.sum() * 100. / len(label)

        try:
            np_predictions = pred.cpu().numpy()
            if self.multiclass:
                multiclass_predictions = self.mlb.fit_transform(np.expand_dims(np_predictions, 1))
                auc = roc_auc_score(multiclass_labels, multiclass_predictions)

            else:
                auc = roc_auc_score(np_labels, np_predictions)

        except ValueError:
            auc = None

        answer_indices = self.num_classes - np.argsort(np.argsort(output.data.cpu().numpy(), 1), 1)
        correct_rank = answer_indices[np.arange(output.shape[0]), np_labels]

        return accuracy, loss.item(), auc, correct_rank, pred

    def save_model(self, epoch=None, save_results=True, **kwargs):
        if epoch is None:
            epoch = len(self.train_accuracies)

        torch.save(self.state_dict(), self._save_path(epoch))
        self.results[epoch] = epoch
        self.results.update(kwargs)
        #         results = dict(epoch=epoch,
        #                        train_accuracies=self.train_accuracies,
        #                        train_losses=self.train_losses,
        #                        test_accuracies=self.test_accuracies,
        #                        test_losses=self.test_losses,
        #                        train_aucs=self.train_aucs,
        #                        test_aucs=self.test_aucs,
        #                        train_correct_rank=self.train_correct_rank,
        #                        test_correct_rank=self.test_correct_rank,
        #                        **kwargs)
        if save_results:
            self.save_results()

    def save_results(self):
        with open(f'{self._save_dir()}/results.pickle', 'wb') as f:
            pickle.dump(self.results, f)

    def load_model(self, epoch=None, load_results=True):
        if epoch == 0:
            print('Warning: asked to load model with epoch 0. Ignoring...')
            return

        if load_results:
            self.load_results()

        elif epoch is None:
            print('Warning: should not set load_results=False without providing epoch #')

        #         self.train_accuracies = results.pop('train_accuracies')
        #         self.train_losses = results.pop('train_losses')
        #         self.test_accuracies = results.pop('test_accuracies')
        #         self.test_losses = results.pop('test_losses')
        #         self.train_aucs = results.pop('train_aucs')
        #         self.test_aucs = results.pop('test_aucs')
        #         self.train_correct_rank = results.pop('train_correct_rank')
        #         self.test_correct_rank = results.pop('test_correct_rank')

        if epoch is None:
            epoch = len(self.train_accuracies)

        # adding support for partial states
        loaded_state = torch.load(self._save_path(epoch))
        state = self.state_dict()
        state.update(loaded_state)
        self.load_state_dict(state)

        return self.results

    def load_results(self):
        with open(f'{self._save_dir()}/results.pickle', 'rb') as f:
            self.results.update(pickle.load(f))

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
    correct_ranks = []
    predictions = []

    for b, (X, y) in enumerate(dataloader):
        if cuda:
            X = X.cuda()
            y = y.cuda()

        images = Variable(X)
        labels = Variable(y).long()

        acc, loss, auc, correct_rank, preds = model.train_(images, labels)
        accuracies.append(acc)
        losses.append(loss)
        if auc is not None: aucs.append(auc)
        correct_ranks.extend(correct_rank)
        predictions.extend(preds)

        if (b + 1) % num_batches_to_print == 0:
            print(
                f'{now()}: After batch {b + 1}, average acc is {np.mean(accuracies):.3f} and average loss is {np.mean(losses):.3f}')

    model.results['train_accuracies'].append(np.mean(accuracies))
    model.results['train_losses'].append(np.mean(losses))
    model.results['train_aucs'].append(np.mean(aucs))
    model.results['train_correct_rank'].append(np.mean(correct_ranks))

    return accuracies, losses, aucs, correct_ranks, predictions


def test(model, dataloader, cuda=True, training=False, return_labels=False):
    accuracies = []
    losses = []
    aucs = []
    correct_ranks = []
    predictions = []
    if return_labels:
        all_labels = []

    for b, (X, y) in enumerate(dataloader):
        if cuda:
            X = X.cuda()
            y = y.cuda()

        images = Variable(X)
        labels = Variable(y).long()
        if return_labels:
            all_labels.extend(labels)

        acc, loss, auc, correct_rank, pred = model.test_(images, labels, training)
        accuracies.append(acc)
        losses.append(loss)
        if auc is not None: aucs.append(auc)
        correct_ranks.extend(correct_rank)
        predictions.extend(pred)

    model.results['test_accuracies'].append(np.mean(accuracies))
    mean_loss = np.mean(losses)
    model.results['test_losses'].append(mean_loss)
    model.post_test(mean_loss)
    model.results['test_aucs'].append(np.mean(aucs))
    model.results['test_correct_rank'].append(np.mean(correct_ranks))

    if return_labels:
        return accuracies, losses, aucs, correct_ranks, predictions, all_labels

    return accuracies, losses, aucs, correct_ranks, predictions


def mid_train_plot(model, epochs_to_test):
    plt.figure(figsize=(16, 4))
    epoch = len(model.results['train_losses'])
    plt.suptitle(f'After epoch {epoch}')
    train_x_values = np.arange(1, epoch + 1)
    test_x_values = np.arange(1, len(model['test_losses']) + 1) * epochs_to_test

    loss_ax = plt.subplot(1, 4, 1)
    loss_ax.set_title('Loss')
    print(train_x_values, model.results['train_losses'])
    print(test_x_values, model.results['test_losses'])
    loss_ax.plot(train_x_values, model.results['train_losses'], label='Train')
    loss_ax.plot(test_x_values, model.results['test_losses'], label='Test')
    loss_ax.legend(loc='best')

    acc_ax = plt.subplot(1, 4, 2)
    acc_ax.set_title('Accuracy')
    acc_ax.plot(train_x_values, model.results['train_accuracies'], label='Train')
    acc_ax.plot(test_x_values, model.results['test_accuracies'], label='Test')
    acc_ax.legend(loc='best')

    auc_ax = plt.subplot(1, 4, 3)
    auc_ax.set_title('AUC')
    auc_ax.plot(train_x_values, model.results['train_aucs'], label='Train')
    auc_ax.plot(test_x_values, model.results['test_aucs'], label='Test')
    auc_ax.legend(loc='best')

    rank_ax = plt.subplot(1, 4, 4)
    rank_ax.set_title('Average Correct Rank')
    rank_ax.plot(train_x_values, model.results['train_correct_rank'], label='Train')
    rank_ax.plot(test_x_values, model.results['test_correct_rank'], label='Test')
    rank_ax.legend(loc='best')

    plt.show()


def train(model, train_dataloader, test_dataloader, num_epochs=100,
          epochs_to_test=5, epochs_to_graph=None, cuda=True,
          num_batches_to_print=DEFAULT_NUM_BATCHES_TO_PRINT, save=True,
          start_epoch=0, watch=True):
    if epochs_to_graph is None:
        epochs_to_graph = epochs_to_test

    if watch:
        wandb.watch(model)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_acc, train_loss, train_auc, train_correct_rank, train_predictions = train_epoch(model, train_dataloader,
                                                                                              cuda,
                                                                                              num_batches_to_print)
        print(
            f'{now()}: After epoch {epoch}, TRAIN acc is {np.mean(train_acc):.4f},  loss is {np.mean(train_loss):.4f}, AUC is {np.mean(train_auc):.4f}, and correct rank is {np.mean(train_correct_rank):.4f}')

        if save:
            model.save_model()

        if epoch % epochs_to_test == 0:
            test_acc, test_loss, test_auc, test_correct_rank, test_predictions = test(model, test_dataloader, cuda,
                                                                                      True)
            print(
                f'{now()}: After epoch {epoch}, TEST acc is {np.mean(test_acc):.4f}, loss is {np.mean(test_loss):.4f}, AUC is {np.mean(test_auc):.4f}, and correct rank is {np.mean(test_correct_rank):.4f}')
            wandb.log({'Train Accuracy': np.mean(train_acc),
                       'Train Loss': np.mean(train_loss),
                       'Train AUC': np.mean(train_auc),
                       'Test Accuracy': np.mean(test_acc),
                       'Test Loss': np.mean(test_loss),
                       'Test AUC': np.mean(test_auc),
                       'Train Correct Rank': np.mean(train_correct_rank),
                       'Test Correct Rank': np.mean(test_correct_rank)},
                      step=epoch)

        if epoch % epochs_to_graph == 0:
            mid_train_plot(model, epochs_to_test)

    # TODO: implement test-error based stopping
