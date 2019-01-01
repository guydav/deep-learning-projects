from ..metalearning.cnn_mlp import PoolingDropoutCNNMLP

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


DEFAULT_SAVE_DIR = 'drive/Research Projects/DendrogramLoss/models'
NUM_CIFAR_10_CLASSES = 10


class DendrogramPoolingDropoutCNNMLP(PoolingDropoutCNNMLP):
    def __init__(self,
                 conv_filter_sizes=(16, 32, 48, 64),
                 conv_dropout=True, conv_p_dropout=0.2,
                 mlp_layer_sizes=(512, 512, 512, 512),
                 mlp_dropout=True, mlp_p_dropout=0.5,
                 conv_output_size=256,
                 lr=1e-3, weight_decay=1e-4, num_classes=NUM_CIFAR_10_CLASSES,
                 use_mse=False, loss=None, compute_correct_rank=True,
                 name='Pooling_Dropout_CNN_MLP', save_dir=DEFAULT_SAVE_DIR):
        super(DendrogramPoolingDropoutCNNMLP, self).__init__(
            conv_filter_sizes=conv_filter_sizes, conv_dropout=conv_dropout, conv_p_dropout=conv_p_dropout,
            mlp_layer_sizes=mlp_layer_sizes, mlp_dropout=mlp_dropout, mlp_p_dropout=mlp_p_dropout,
            conv_output_size=conv_output_size, lr=lr, weight_decay=weight_decay, num_classes=num_classes,
            use_mse=use_mse, loss=loss, compute_correct_rank=compute_correct_rank, name=name, save_dir=save_dir)

