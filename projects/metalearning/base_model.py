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
from sklearn.preprocessing import MultiLabelBinarizer

from collections import defaultdict
from datetime import datetime
import pickle
import os

DEFAULT_SAVE_DIR = 'drive/Research Projects/Meta-Learning/v1/models'
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_BATCHES_TO_PRINT = 10000

DEFAULT_CONV_FILTER_SIZES = (24, 24, 24, 24)
DEFAULT_MLP_LAYER_SIZES = (256, 64, 16, 4)


class BasicModel(nn.Module):
    def __init__(self, name, should_super_init=True, use_mse=False,
                 save_dir=DEFAULT_SAVE_DIR, mse_threshold=0.5, num_classes=2,
                 loss=None, use_query=True, compute_correct_rank=False):
        if should_super_init:
            super(BasicModel, self).__init__()

        self.name = name
        self.use_mse = use_mse
        self.mse_threshold = mse_threshold
        self.num_classes = num_classes
        self.use_query = use_query
        self.compute_correct_rank = compute_correct_rank

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

    def train_(self, input_img, label, query=None):
        if self.optimizer is None:
            self._create_optimizer()

        self.optimizer.zero_grad()
        output = self(input_img, query)

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

        correct = pred.eq(label.data).cpu()
        accuracy = correct.sum() * 100. / len(label)

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

        per_query_results = defaultdict(list)
        for q, c in zip(np.argmax(query.cpu().numpy(), 1), correct.numpy()):
            per_query_results[q].append(c)

        results = dict(
            accuracy=accuracy,
            loss=loss.item(),
            auc=auc,
            pred=pred,
            per_query_results=per_query_results
        )

        if self.compute_correct_rank:
            answer_indices = self.num_classes - np.argsort(np.argsort(output.data.cpu().numpy(), 1), 1)
            results['correct_rank'] = answer_indices[np.arange(output.shape[0]), np_labels]

        return results

    def test_(self, input_img, label, query=None):
        output = self(input_img, query)

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

        per_query_results = defaultdict(list)
        for q, c in zip(np.argmax(query.cpu().numpy(), 1), correct.numpy()):
            per_query_results[q].append(c)

        results = dict(
            accuracy=accuracy,
            loss=loss.item(),
            auc=auc,
            pred=pred,
            per_query_results=per_query_results
        )

        if self.compute_correct_rank:
            answer_indices = self.num_classes - np.argsort(np.argsort(output.data.cpu().numpy(), 1), 1)
            results['correct_rank'] = answer_indices[np.arange(output.shape[0]), np_labels]

        return results

    def save_model(self, epoch=None, save_results=True, **kwargs):
        if epoch is None:
            epoch = len(self.results['train_accuracies'])

        torch.save(self.state_dict(), self._save_path(epoch))
        self.results['epoch'] = epoch
        self.results.update(kwargs)

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

        if epoch is None:
            if 'epoch' in self.results:
                epoch = self.results['epoch']
            else:
                epoch = len(self.results['train_accuracies'])

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
        # print(os.system(f'ls -laR {self._save_dir()}'))

    def _create_optimizer(self):
        raise NotImplementedError()

    def post_test(self, test_loss, epoch):
        pass


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def train_epoch(model, dataloader, cuda=True, device=None,
                num_batches_to_print=DEFAULT_NUM_BATCHES_TO_PRINT):
    epoch_results = defaultdict(list)
    epoch_results['per_query_results'] = defaultdict(list)

    for batch_index, batch in enumerate(dataloader):
        if model.use_query:
            X, y, Q = batch

        else:
            X, y = batch
            Q = None

        if cuda:
            X = X.to(device)
            y = y.to(device)
            if Q is not None: Q = Q.to(device)

        images = Variable(X)
        labels = Variable(y).long()
        if Q is not None:
            queries = Variable(Q).float()
            results = model.train_(images, labels, queries)
        else:
            results = model.train_(images, labels)

        epoch_results['accuracies'].append(results['accuracy'])
        epoch_results['losses'].append(results['loss'])
        if 'auc' in results and results['auc'] is not None: epoch_results['aucs'].append(results['auc'])
        if model.compute_correct_rank: epoch_results['correct_rank'].extend(results['correct_rank'])

        for query in results['per_query_results']:
            epoch_results['per_query_results'][query].extend(results['per_query_results'][query])

        if (batch_index + 1) % num_batches_to_print == 0:
            print(
                f'{now()}: After batch {batch_index + 1}, average acc is {np.mean(accuracies):.3f} and average loss is {np.mean(losses):.3f}')

    model.results['train_accuracies'].append(np.mean(epoch_results['accuracies']))
    model.results['train_losses'].append(np.mean(epoch_results['losses']))
    model.results['train_aucs'].append(np.mean(epoch_results['aucs']))
    if model.compute_correct_rank: model.results['train_correct_rank'].append(np.mean(epoch_results['correct_rank']))
    model.results['train_per_query_accuracies'].append(
        {query: np.mean(values) for query, values in epoch_results['per_query_results'].items()})

    return epoch_results


def test(model, dataloader, cuda=True, device=None, training=False):
    test_results = defaultdict(list)
    test_results['per_query_results'] = defaultdict(list)

    for batch in dataloader:
        if model.use_query:
            X, y, Q = batch

        else:
            X, y = batch
            Q = None

        if cuda:
            X = X.to(device)
            y = y.to(device)
            if Q is not None: Q = Q.to(device)

        images = Variable(X)
        labels = Variable(y).long()
        if Q is not None:
            queries = Variable(Q).float()
            results = model.test_(images, labels, queries)
        else:
            results = model.test_(images, labels)

        test_results['accuracies'].append(results['accuracy'])
        test_results['losses'].append(results['loss'])
        if 'auc' in results and results['auc'] is not None: test_results['aucs'].append(results['auc'])
        if model.compute_correct_rank: test_results['correct_rank'].extend(results['correct_rank'])

        for query in results['per_query_results']:
            test_results['per_query_results'][query].extend(results['per_query_results'][query])

    model.results['test_accuracies'].append(np.mean(test_results['accuracies']))
    mean_loss = np.mean(test_results['losses'])
    model.results['test_losses'].append(mean_loss)
    model.results['test_aucs'].append(np.mean(test_results['aucs']))
    if model.compute_correct_rank: model.results['test_correct_rank'].append(np.mean(test_results['correct_rank']))
    model.results['test_per_query_accuracies'].append(
        {query: np.mean(values) for query, values in test_results['per_query_results'].items()})

    if training: model.post_test(mean_loss, len(model.results['test_losses']))

    return test_results


def mid_train_plot(model, epochs_to_test):
    num_plots = 4
    plt.figure(figsize=(16, num_plots))
    epoch = len(model.results['train_losses'])
    plt.suptitle(f'After epoch {epoch}')
    train_x_values = np.arange(1, epoch + 1)
    test_x_values = np.arange(1, epoch // epochs_to_test + 1) * epochs_to_test

    loss_ax = plt.subplot(1, num_plots, 1)
    loss_ax.set_title('Loss')
    # print(train_x_values, model.results['train_losses'])
    # print(test_x_values, model.results['test_losses'])
    loss_ax.plot(train_x_values, model.results['train_losses'], label='Train')
    loss_ax.plot(test_x_values, model.results['test_losses'], label='Test')
    loss_ax.legend(loc='best')

    acc_ax = plt.subplot(1, num_plots, 2)
    acc_ax.set_title('Accuracy')
    acc_ax.plot(train_x_values, model.results['train_accuracies'], label='Train')
    acc_ax.plot(test_x_values, model.results['test_accuracies'], label='Test')
    acc_ax.legend(loc='best')

    auc_ax = plt.subplot(1, num_plots, 3)
    auc_ax.set_title('AUC')
    auc_ax.plot(train_x_values, model.results['train_aucs'], label='Train')
    auc_ax.plot(test_x_values, model.results['test_aucs'], label='Test')
    auc_ax.legend(loc='best')

    per_query_accuracies = defaultdict(list)
    for epoch_accuracies in model.results['test_per_query_accuracies']:
        for query, acc in epoch_accuracies.items():
            per_query_accuracies[query].append(acc)

    per_query_accuracy_ax = plt.subplot(1, num_plots, 4)
    per_query_accuracy_ax.set_title('Average Per-Query Accuracy (test)')
    for query, accuracies in per_query_accuracies.items():
        per_query_accuracy_ax.plot(test_x_values[-len(accuracies):], accuracies, label=str(query))
    per_query_accuracy_ax.legend(loc='best')

    # if model.compute_correct_rank:
    #     rank_ax = plt.subplot(1, num_plots, 4)
    #     rank_ax.set_title('Average Correct Rank')
    #     rank_ax.plot(train_x_values, model.results['train_correct_rank'], label='Train')
    #     rank_ax.plot(test_x_values, model.results['test_correct_rank'], label='Test')
    #     rank_ax.legend(loc='best')

    plt.show()


def print_status(model, epoch, status_type, results):
    status_lines = [
        f'{now()}: After epoch {epoch}, {status_type} acc is {np.mean(results["accuracies"]):.4f},',
        f'loss is {np.mean(results["losses"]):.4f}, AUC is {np.mean(results["aucs"]):.4f}'
    ]

    if model.compute_correct_rank:
        status_lines.append(f'and correct rank is {np.mean(results["correct_rank"]):.4f}')

    print(' '.join(status_lines))


def train(model, train_dataloader, test_dataloader, num_epochs=100,
          epochs_to_test=5, epochs_to_graph=None, cuda=True,
          num_batches_to_print=DEFAULT_NUM_BATCHES_TO_PRINT, save=True,
          start_epoch=0, watch=True):

    if epochs_to_graph is None:
        epochs_to_graph = epochs_to_test

    if cuda:
        device = next(model.parameters()).device

    if watch:
        wandb.watch(model)

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_results = train_epoch(model, train_dataloader, cuda, device, num_batches_to_print)
        print_status(model, epoch, 'TRAIN', train_results, )

        if save:
            model.save_model()

        if epoch % epochs_to_test == 0:
            test_results = test(model, test_dataloader, cuda, device, True)
            print_status(model, epoch, 'TEST', test_results)

            log_results = {
                'Train Accuracy': np.mean(train_results['accuracies']),
                'Train Loss': np.mean(train_results['losses']),
                'Train AUC': np.mean(train_results['aucs']),
                'Train Per-Query Accuracy (dict)': {str(query): np.mean(values) for query, values in
                                                    train_results['per_query_results'].items()},
                'Test Accuracy': np.mean(test_results['accuracies']),
                'Test Loss': np.mean(test_results['losses']),
                'Test AUC': np.mean(test_results['aucs']),
                'Test Per-Query Accuracy (dict)': {str(query): np.mean(values) for query, values in
                                                   test_results['per_query_results'].items()},
            }
            if model.compute_correct_rank:
                log_results['Train Correct Rank'] = np.mean(train_results['correct_rank'])
                log_results['Test Correct Rank'] = np.mean(test_results['correct_rank'])

            wandb.log(log_results, step=epoch)

        if epoch % epochs_to_graph == 0:
            mid_train_plot(model, epochs_to_test)


