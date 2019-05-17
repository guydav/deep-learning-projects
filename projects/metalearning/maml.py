from .base_model import *
from .cnnmlp import *

import torch

import copy


class MamlModel(BasicModel):
    """
    A base neural network module implementing widely useful behavior, such selecting
    the right loss function, running training and testing and computing the requisite
    metrics, and so on. This model does not actually specify the structure of the neural
    networks examined; subclasses implement that behavior.
    """
    def __init__(self, fast_weight_lr, name, should_super_init=True, use_mse=False,
                 save_dir=DEFAULT_SAVE_DIR, mse_threshold=0.5, num_classes=2,
                 loss=None, use_query=True, compute_correct_rank=False):

        super(MamlModel, self).__init__(name, should_super_init, use_mse, save_dir,
                                        mse_threshold, num_classes, loss, use_query,
                                        compute_correct_rank)

        # self.optimizer is actually the meta-optimizer in MAML parlance
        self.fast_weight_lr = fast_weight_lr

    def train_(self, input_img, label, query=None):
        raise NotImplemented('This method should never be called on the MAML model')

    def _create_fast_weight_optimizer(self):
        self.fast_weight_optimizer = optim.SGD(self.parameters(), self.fast_weight_lr)

    def maml_train_(self, X_train, Q_train, y_train, X_meta_train, Q_meta_train, y_meta_train,
                    active_tasks, debug=False):
        """
        """
        if self.optimizer is None:
            self._create_optimizer()
            self._create_fast_weight_optimizer()

        meta_objective_loss = 0
        meta_train_correct = []
        meta_train_preds_for_auc = []
        meta_train_labels_for_auc = []

        _, train_task_indices = Q_train.max(1)
        _, meta_train_task_indices = Q_meta_train.max(1)

        pre_training_weights = copy.deepcopy(self.state_dict())
        need_load = False

        for task in active_tasks:
            # No need to load the first time around
            if need_load:
                self.load_state_dict(pre_training_weights)
            need_load = True

            # extract training and meta-training data for task
            train_task_examples = train_task_indices == task
            X_train_task = X_train[train_task_examples]
            Q_train_task = Q_train[train_task_examples]
            y_train_task = y_train[train_task_examples]

            meta_train_task_examples = meta_train_task_indices == task
            X_meta_train_task = X_meta_train[meta_train_task_examples]
            Q_meta_train_task = Q_meta_train[meta_train_task_examples]
            y_meta_train_task = y_meta_train[meta_train_task_examples]

            if debug: print(task, torch.sum(train_task_examples), torch.sum(meta_train_task_examples))

            # train on task
            self.fast_weight_optimizer.zero_grad()
            task_output = self(X_train_task, Q_train_task)
            task_loss = self.loss(task_output, y_train_task)
            task_loss.backward()
            self.fast_weight_optimizer.step()
            # task_grad = autograd.grad(task_loss, self.parameters())
            # task_fast_weights = list(map(lambda p, g: p - self.fast_weight_lr * g, self.parameters(), task_grad))

            # meta-train on task
            # meta_train_model_state = meta_train_model_copy.state_dict()
            # param_names = meta_train_model_state.keys()
            # for name, fast_weight in zip(param_names, task_fast_weights):
            #     meta_train_model_state[name] = fast_weight
            #
            # meta_train_model_copy.load_state_dict(meta_train_model_state)

            # meta_task_output = meta_train_model_copy(X_meta_train_task, Q_meta_train_task)
            # meta_task_loss = meta_train_model_copy.loss(meta_task_output, y_meta_train_task)
            # meta_objective_losses.append(meta_task_loss)

            meta_task_output = self(X_meta_train_task, Q_meta_train_task)
            meta_task_loss = self.loss(meta_task_output, y_meta_train_task)
            meta_objective_loss += meta_task_loss

            meta_task_pred = meta_task_output.data.max(1)[1]
            meta_train_preds_for_auc.append(meta_task_pred.data.cpu().numpy())
            meta_task_correct = meta_task_pred.eq(y_meta_train_task.data).cpu()
            meta_train_correct.append(meta_task_correct.numpy())
            meta_train_labels_for_auc.append(y_meta_train_task.data.cpu().numpy())

        self.load_state_dict(pre_training_weights)
        meta_loss = meta_objective_loss / len(active_tasks)
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            meta_train_accuracy = np.concatenate(meta_train_correct).ravel().mean() * 100

            try:
                all_meta_train_labels = np.concatenate(meta_train_labels_for_auc).ravel()
                all_meta_train_preds = np.concatenate(meta_train_preds_for_auc).ravel()
                auc = roc_auc_score(all_meta_train_labels, all_meta_train_preds)
            except ValueError:
                auc = None

            per_query_results = {task: list(correct) for task, correct in zip(active_tasks, meta_train_correct)}

        return dict(accuracy=meta_train_accuracy, loss=meta_loss.item(),
                    auc=auc, per_query_results=per_query_results)


class MamlPoolingDropoutCNNMLP(CNNMLPMixIn, MamlModel):
    """
    A full model combining the more advanced versions of the convolutional input module (with pooling and dropout
    support) and fully-connected module (with dropout support). Also adding a learning rate scheduler, support for
    using weight decay, and alternative loss functions.
    """

    def __init__(self, fast_weight_lr, query_length=30, conv_filter_sizes=(16, 24, 32, 40),
                 conv_dropout=True, conv_p_dropout=0.2,
                 mlp_layer_sizes=(256, 256, 256, 256),
                 mlp_dropout=True, mlp_p_dropout=0.5, use_lr_scheduler=True, lr_scheduler_patience=5,
                 conv_output_size=1920, lr=1e-4, weight_decay=0, num_classes=2,
                 name='Pooling_Dropout_CNN_MLP', save_dir=DEFAULT_SAVE_DIR):
        """
        :param query_length: What length of query to expect; defaults to 30
        :param conv_filter_sizes: How many filters to use in each convolutional filter group
        :param conv_dropout: Should spatial dropout be used on the convolutional layers
        :param conv_p_dropout: If using spatial dropout, what dropout proabability?
        :param mlp_layer_sizes: What sizes to use for the fully-connected (MLP, multilayer perceptron)
        :param mlp_dropout: Should dropout be used in the MLP?
        :param mlp_p_dropout: If using dropout, what dropout probability to use
        :param use_lr_scheduler: Should a learning rate scheduler to reduce the learning rate on plateau be used?
        :param lr_scheduler_patience: If using a learning rate scheduler, how long a plateau to move before?
        :param conv_output_size: What output size (once flattened) should be expected from the CNN?
        :param lr: Learning rate to use
        :param weight_decay: Weight decay to use
        :param num_classes: How many classes to support
        :param use_mse: Whether or not to use the mean squared error (MSE) loss function
        :param loss: Which loss function to use, if not the defaul one
        :param compute_correct_rank: Whether or not to compute the correct rank of every mistaken classification
        :param name: Which name to save checkpoints under
        :param save_dir: Which directory to save checkpoints to
        """
        CNNMLPMixIn.__init__(self)
        MamlModel.__init__(self, fast_weight_lr=fast_weight_lr, name=name, save_dir=save_dir,
                           num_classes=num_classes, use_mse=False, loss=None, use_query=query_length != 0,
                           compute_correct_rank=False)

        self.query_length = query_length
        self.conv = self._create_conv_module(conv_filter_sizes, conv_dropout, conv_p_dropout)
        self.fc1 = nn.Linear(conv_output_size + query_length, mlp_layer_sizes[0])  # query concatenated to all

        fc_output_func = lambda x: F.log_softmax(x, dim=1)
        self.fcout = self._create_fc_module(fc_output_func, mlp_dropout, mlp_layer_sizes, mlp_p_dropout, num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_patience = lr_scheduler_patience


def maml_train_epoch(model, dataloader, cuda=True, device=None,
                     num_batches_to_print=DEFAULT_NUM_BATCHES_TO_PRINT, debug=False):

    epoch_results = defaultdict(list)
    epoch_results['per_query_results'] = defaultdict(list)

    dataloader_iter = iter(dataloader)

    for batch_index, train_batch in enumerate(dataloader_iter):
        X_train, Q_train, y_train = split_batch(train_batch, cuda, device, model)
        meta_train_batch = next(dataloader_iter)
        X_meta_train, Q_meta_train, y_meta_train = split_batch(meta_train_batch, cuda, device, model)

        results = model.maml_train_(X_train, Q_train, y_train, X_meta_train, Q_meta_train, y_meta_train,
                                    dataloader.dataset.query_order[:dataloader.dataset.current_query_index + 1],
                                    debug=debug)

        epoch_results['accuracies'].append(results['accuracy'])
        epoch_results['losses'].append(results['loss'])
        if 'auc' in results and results['auc'] is not None: epoch_results['aucs'].append(results['auc'])

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


def split_batch(batch, cuda, device, model):
    if model.use_query:
        if len(batch) == 4:
            X, y, Q, index = batch
            print(index[:10])
            print(index[-10:])

        else:
            X, y, Q = batch

    else:
        X, y = batch
        Q = None

    if cuda:
        X = X.to(device)
        y = y.to(device)
        if Q is not None: Q = Q.to(device)

    X = Variable(X)
    y = Variable(y).long()
    if Q is not None: Q = Variable(Q).float()

    return X, Q, y
