from .base_model import BasicModel, DEFAULT_SAVE_DIR

import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class PoolingDropoutConvInputModel(nn.Module):
    def __init__(self, filter_sizes=(16, 24, 32, 40),
                 dropout=True, p_dropout=0.2):
        super(PoolingDropoutConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, filter_sizes[0], 3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(filter_sizes[0])
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], 3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(filter_sizes[1])
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], 3, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(filter_sizes[2])
        self.conv4 = nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, stride=1, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(filter_sizes[3])

        self.dropout = dropout
        self.p_dropout = p_dropout


    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        return x


class SmallerDropoutFCOutputModel(nn.Module):
    def __init__(self, layer_sizes=(256, 256, 256, 256),
                 dropout=True, p_dropout=0.5, output_func=F.log_softmax, output_size=2):
        super(SmallerDropoutFCOutputModel, self).__init__()

        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc5 = nn.Linear(layer_sizes[3], output_size)

        self.dropout = dropout
        self.p_dropout = p_dropout
        self.output_func = output_func

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, self.p_dropout, self.training)

        x = self.fc3(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, self.p_dropout, self.training)

        x = self.fc4(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, self.p_dropout, self.training)

        x = self.fc5(x)

        if self.output_func is not None:
            return self.output_func(x)
        else:
            return x


class PoolingDropoutCNN_MLP(BasicModel):
    def __init__(self, conv_filter_sizes=(16, 24, 32, 40),
                 conv_dropout=True, conv_p_dropout=0.2,
                 mlp_layer_sizes=(256, 256, 256, 256),
                 mlp_dropout=True, mlp_p_dropout=0.5,
                 conv_output_size=1920, lr=1e-4, weight_decay=0, num_classes=10,
                 use_mse=False, loss=None,
                 name='Pooling_Dropout_CNN_MLP', save_dir=DEFAULT_SAVE_DIR):
        super(PoolingDropoutCNN_MLP, self).__init__(name=name, save_dir=save_dir, num_classes=num_classes,
                                                    use_mse=use_mse, loss=loss)

        self.conv  = PoolingDropoutConvInputModel(conv_filter_sizes,
                                                  conv_dropout,
                                                  conv_p_dropout)
        self.fc1   = nn.Linear(conv_output_size, mlp_layer_sizes[0])  # query concatenated to all

        if use_mse:
            fc_output_func = lambda x: F.softmax(x, dim=1)
        else:
            fc_output_func = lambda x: F.log_softmax(x, dim=1)

        self.fcout = SmallerDropoutFCOutputModel(mlp_layer_sizes,
                                                 mlp_dropout,
                                                 mlp_p_dropout,
                                                 output_func=fc_output_func,
                                                 output_size=num_classes)
        self.lr = lr
        self.weight_decay = weight_decay

    def _create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              factor=0.5, patience=10, verbose=True)

    def post_test(self, test_loss):
        epoch = len(self.test_losses)
        if epoch > 100:
            self.scheduler.step(test_loss)

    def forward(self, img):
        x = self.conv(img)  # x = (16 x 24 x 15 x 20)
        # fully connected layers
        x = x.view(x.size(0), -1)
        x_ = self.fc1(x)
        x_ = F.relu(x_)

        return self.fcout(x_)
