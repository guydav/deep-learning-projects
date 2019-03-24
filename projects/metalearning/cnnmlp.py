from .base_model import *


class ConvInputModel(nn.Module):
    """
    A simple convolutional input model: performs four convolutional layers with the provided filter sizes,
    all using 3x3 convolutions with stride 2 and padding 1, and ReLU activation functions and batch normalization.
    """
    def __init__(self, filter_sizes=DEFAULT_CONV_FILTER_SIZES):
        super(ConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, filter_sizes[0], 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(filter_sizes[0])
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(filter_sizes[1])
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(filter_sizes[2])
        self.conv4 = nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(filter_sizes[3])

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


class FCOutputModel(nn.Module):
    """
    Simple fully-connected module: performs four fully-conencted layers with the given layer sizes,
    using ReLU activation functions, and either a softmax or log-softmax on the output function
    """
    def __init__(self, layer_sizes=DEFAULT_MLP_LAYER_SIZES, log_softmax=True):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc5 = nn.Linear(layer_sizes[3], 2)
        self.log_softmax = log_softmax

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        if self.log_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)


class CNNMLP(BasicModel):
    """
    The basic architecture for this task, using a convolutional neural networks followed by a fully connected one,
    concatenating the query in between the two modules.
    """
    def __init__(self, query_length, conv_filter_sizes=DEFAULT_CONV_FILTER_SIZES,
                 mlp_layer_sizes=DEFAULT_MLP_LAYER_SIZES, conv_output_size=7200,
                 lr=1e-4,
                 name='CNN_MLP', save_dir=DEFAULT_SAVE_DIR):
        super(CNNMLP, self).__init__(name, save_dir)

        self.query_length = query_length
        self.conv = ConvInputModel(conv_filter_sizes)
        self.fc1 = nn.Linear(conv_output_size + query_length, mlp_layer_sizes[0])  # query concatenated to all
        self.fcout = FCOutputModel(mlp_layer_sizes)
        self.lr = lr

    def _create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, img, query):
        x = self.conv(img)  ## x = (16 x 24 x 15 x 20)
        """fully connected layers"""
        x = x.view(x.size(0), -1)

        x_ = torch.cat((x, query), 1)  # Concat query - as a float?

        x_ = self.fc1(x_)
        x_ = F.relu(x_)

        return self.fcout(x_)


class PoolingDropoutConvInputModel(nn.Module):
    """
    A more advanced version of the convolutional input model, offering support for spatial convolutions after
    all layers, and performing 2x2 max-pooling after every convolutional layer group.
    """
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
    """
    A modified version of the fully-connected output model, supporting dropout after each layer.
    """
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


#         if self.log_softmax:
#             return F.log_softmax(x, dim=1)
#         else:
#             return F.softmax(x, dim=1)


class PoolingDropoutCNNMLP(BasicModel):
    """
    A full model combining the more advanced versions of the convolutional input module (with pooling and dropout
    support) and fully-connected module (with dropout support). Also adding a learning rate scheduler, support for
    using weight decay, and alternative loss functions.
    """
    def __init__(self, query_length=30, conv_filter_sizes=(16, 24, 32, 40),
                 conv_dropout=True, conv_p_dropout=0.2,
                 mlp_layer_sizes=(256, 256, 256, 256),
                 mlp_dropout=True, mlp_p_dropout=0.5, use_lr_scheduler=True, lr_scheduler_patience=5,
                 conv_output_size=1920, lr=1e-4, weight_decay=0, num_classes=2,
                 use_mse=False, loss=None, compute_correct_rank=False,
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
        super(PoolingDropoutCNNMLP, self).__init__(name=name, save_dir=save_dir, num_classes=num_classes,
                                                   use_mse=use_mse, loss=loss,
                                                   use_query=query_length != 0,
                                                   compute_correct_rank=compute_correct_rank)

        
        self.query_length = query_length
        self.conv = self._create_conv_module(conv_filter_sizes, conv_dropout, conv_p_dropout)
        self.fc1 = nn.Linear(conv_output_size + query_length, mlp_layer_sizes[0])  # query concatenated to all
        output_size = num_classes
        
        if use_mse:
            if num_classes == 2:
                output_size = 1
                fc_output_func = lambda x: torch.sigmoid(x)
            else:
                fc_output_func = lambda x: F.softmax(x, dim=1)
        else:
            fc_output_func = lambda x: F.log_softmax(x, dim=1)

        self.fcout = self._create_fc_module(fc_output_func, mlp_dropout, mlp_layer_sizes, mlp_p_dropout, output_size)
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_patience = lr_scheduler_patience

    def _create_conv_module(self, conv_filter_sizes, conv_dropout, conv_p_dropout):
        return PoolingDropoutConvInputModel(conv_filter_sizes,
                                            conv_dropout,
                                            conv_p_dropout)

    def _create_fc_module(self, fc_output_func, mlp_dropout, mlp_layer_sizes, mlp_p_dropout, output_size):
        return SmallerDropoutFCOutputModel(mlp_layer_sizes,
                                           mlp_dropout,
                                           mlp_p_dropout,
                                           output_func=fc_output_func,
                                           output_size=output_size)

    def _create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  factor=0.5,
                                                                  patience=self.lr_scheduler_patience,
                                                                  verbose=True)

    def post_test(self, test_loss, epoch):
        if self.use_lr_scheduler:
            self.scheduler.step(test_loss)

    def forward(self, img, query=None):
        x = self.conv(img)  ## x = (16 x 24 x 15 x 20)
        """fully connected layers"""
        x = x.view(x.size(0), -1)

        x_ = x
        if self.query_length > 0:
            x_ = torch.cat((x_, query), 1)  # Concat query - as a float?

        x_ = self.fc1(x_)
        x_ = F.relu(x_)

        return self.fcout(x_)


class QueryModulatingPoolingDropoutConvInputModel(nn.Module):
    """
    A query-modulating convolutional model. Receives the query as well, passes tit hrough a single
    fully-connected layer, and adds the output for each filter at the appropriate level before the activation function.
    Setting mod_level to zero corresponds to no query modulation, aka the baseline model.
    """
    def __init__(self, mod_level, query_length=30, filter_sizes=(16, 24, 32, 40),
                 dropout=True, p_dropout=0.2):
        super(QueryModulatingPoolingDropoutConvInputModel, self).__init__()

        if mod_level < 0 or mod_level > 4:
            raise ValueError('Query modulation level should be between 0 and 4 inclusive')

        self.mod_level = mod_level
        self.query_mod_layer = nn.Linear(query_length, filter_sizes[self.mod_level - 1])

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

    def forward(self, img, query):
        # adding two fake dimensions for the spatial ones => [b, c, w, h]
        # checking for > 0 such that mod_level = 0 is the baseline model
        if self.mod_level > 0:
            query_mod = self.query_mod_layer(query)[:, :, None, None]

        """convolution"""
        x = self.conv1(img)
        if self.mod_level == 1:
            x = x + query_mod
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        x = self.conv2(x)
        if self.mod_level == 2:
            x = x + query_mod
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        x = self.conv3(x)
        if self.mod_level == 3:
            x = x + query_mod
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        x = self.conv4(x)
        if self.mod_level == 4:
            x = x + query_mod
        x = F.relu(x)
        x = self.batchNorm4(x)
        x = F.max_pool2d(x, 2)
        if self.dropout:
            x = F.dropout2d(x, self.p_dropout, self.training)

        return x


class QueryModulatingCNNMLP(PoolingDropoutCNNMLP):
    """
    A full model using the query-modulating convolutional model; see documentation above.m
    """
    def __init__(self, mod_level, query_length=30, conv_filter_sizes=(16, 24, 32, 40),
                 conv_dropout=True, conv_p_dropout=0.2,
                 mlp_layer_sizes=(256, 256, 256, 256),
                 mlp_dropout=True, mlp_p_dropout=0.5, use_lr_scheduler=True, lr_scheduler_patience=5,
                 conv_output_size=1920, lr=1e-4, weight_decay=0, num_classes=2,
                 use_mse=False, loss=None, compute_correct_rank=False,
                 name='Pooling_Dropout_CNN_MLP', save_dir=DEFAULT_SAVE_DIR):

        self.mod_level = mod_level

        super(QueryModulatingCNNMLP, self).__init__(
            query_length, conv_filter_sizes, conv_dropout, conv_p_dropout,
            mlp_layer_sizes, mlp_dropout, mlp_p_dropout, use_lr_scheduler, lr_scheduler_patience,
            conv_output_size, lr, weight_decay, num_classes, use_mse, loss,
            compute_correct_rank, name, save_dir
        )

    def _create_conv_module(self, conv_filter_sizes, conv_dropout, conv_p_dropout):
        return QueryModulatingPoolingDropoutConvInputModel(self.mod_level, self.query_length,
                                                          conv_filter_sizes, conv_dropout, conv_p_dropout)

    def forward(self, img, query):
        x = self.conv(img, query)  # adding the query to be modulated
        # x = (16 x 24 x 15 x 20)
        """fully connected layers"""
        x = x.view(x.size(0), -1)

        x_ = x
        if self.query_length > 0:
            x_ = torch.cat((x_, query), 1)  # Concat query - as a float?

        x_ = self.fc1(x_)
        x_ = F.relu(x_)

        return self.fcout(x_)
