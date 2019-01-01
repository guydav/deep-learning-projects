from .cnn_model import DendrogramPoolingDropoutCNNMLP
from .dendrogram import DendrogramLoss, HingeDendrogramLoss, HingeDendrogramMarginLoss, \
                        DEFAULT_CLASSES, DEFAULT_EDGE_DICTS, ALPHABETICAL_EDGE_DICTS

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def load_relevant_models():
    test_models = []
    name_to_label = {}

    def load_saved_model(name, label, loss=None, use_mse=False):
        if loss is not None:
            loss = loss.cuda()

        model = DendrogramPoolingDropoutCNNMLP(
            conv_filter_sizes=(16, 32, 48, 64),
            conv_output_size=256,
            mlp_layer_sizes=(512, 512, 512, 512),
            lr=1e-3,
            weight_decay=1e-4,
            use_mse=use_mse,
            loss=loss,
            name=name
        ).cuda()

        test_models.append(model)
        name_to_label[name] = label
        return model

    # Standard cross-entropy
    cnn_mlp_standard_conv_model = load_saved_model('cnn_mlp_standard_conv_model', 'Cross Entropy')

    # MSE-based models

    mse_loss_fixed_cnn_mlp = load_saved_model('mse_loss_fixed_cnn_mlp', 'MSE', use_mse=True)

    dendrogram_loss = DendrogramLoss(DEFAULT_EDGE_DICTS, DEFAULT_CLASSES)
    dendrogram_loss_cnn_mlp = load_saved_model('dendrogram_loss_cnn_mlp', 'MSE Dendrogram', dendrogram_loss,
                                               use_mse=True)

    alphabetical_dendrogram_loss = DendrogramLoss(ALPHABETICAL_EDGE_DICTS, DEFAULT_CLASSES)
    alphabetical_dendrogram_loss_cnn_mlp = load_saved_model(
        'alphabetical_dendrogram_loss_cnn_mlp', 'MSE Dendrogram (Alphabetical)', alphabetical_dendrogram_loss,
        use_mse=True)

    # Hinge-based models
    hinge_loss = torch.nn.MultiMarginLoss()
    hinge_loss_fixed_cnn_mlp = load_saved_model('hinge_loss_fixed_cnn_mlp', 'Hinge (L1 SVM)', hinge_loss)

    hinge_squared_loss = torch.nn.MultiMarginLoss(p=2)
    hinge_squared_loss_fixed_cnn_mlp = load_saved_model('hinge_squared_loss_fixed_cnn_mlp', 'Hinge Squared (L2 SVM)',
                                                        hinge_squared_loss)

    hinge_dendrogram_loss = HingeDendrogramMarginLoss(DEFAULT_EDGE_DICTS, DEFAULT_CLASSES)
    hinge_dendrogram_loss_cnn_mlp = load_saved_model('hinge_dendrogram_loss_cnn_mlp', 'Hinge Dendrogram',
                                                     hinge_dendrogram_loss)

    large_margin_hinge_squared_loss = torch.nn.MultiMarginLoss(p=2, margin=4.5)
    large_margin_hinge_squared_loss_cnn_mlp = load_saved_model(
        'large_margin_hinge_squared_loss_cnn_mlp', 'Hinge Squared (L2 SVM), Large Margin',
        large_margin_hinge_squared_loss)

    large_margin_hinge_squared_dendrogram_loss = HingeDendrogramLoss(DEFAULT_EDGE_DICTS, DEFAULT_CLASSES, p=2, margin=4.5)
    large_margin_hinge_squared_dendrogram_loss_cnn_mlp = load_saved_model(
        'large_margin_hinge_squared_dendrogram_loss_cnn_mlp', 'Hinge Squared Dendrogram, Large Margin',
        large_margin_hinge_squared_dendrogram_loss)

    alphabetical_hinge_dendrogram_loss = HingeDendrogramLoss(ALPHABETICAL_EDGE_DICTS, DEFAULT_CLASSES, p=2).cuda()
    alphabetical_hinge_squared_dendrogram_loss_cnn_mlp = load_saved_model(
        'alphabetical_hinge_squared_dendrogram_loss_cnn_mlp', 'Hinge Squared Dendrogram (Alphabetical)',
        alphabetical_hinge_dendrogram_loss)

    hinge_squared_dendrogram_margin_loss = HingeDendrogramMarginLoss(DEFAULT_EDGE_DICTS, DEFAULT_CLASSES, p=2,
                                                                     distance_scale=2.0)
    hinge_squared_dendrogram_margin_loss_cnn_mlp = load_saved_model(
        'hinge_squared_dendrogram_margin_loss_cnn_mlp', 'Hinge Squared Dendrogram Margin',
        hinge_squared_dendrogram_margin_loss)

    return test_models, name_to_label


DEFAULT_PLOT_SAVE_DIR = r'drive/Research Projects/DendrogramLoss/plots'


def plot_model_results(models, model_labels, results_key, results_name=None, figsize=(12, 9), colors=None,
                       x_label='Epoch', x_range=None, max_epoch=200, font_size=16, ylim=None,
                       result_extractor=None, result_extractor_params=None, model_specific_limits=None,
                       save_name=None, save_dir=DEFAULT_PLOT_SAVE_DIR):
    plt.figure(figsize=figsize)

    if results_name is None:
        results_name = results_key.replace('_', ' ').replace('ies', 'y').title()

    if x_range is None:
        x_range = np.arange(1, max_epoch + 1)

    for i, model in enumerate(models):
        if len(model.results.keys()) == 0:
            load_epoch = max_epoch
            if model_specific_limits is not None and model.name in model_specific_limits:
                load_epoch = model_specific_limits[model.name]
            model.load_model(load_epoch)

        model_results = model.results[results_key]
        if result_extractor is not None:
            if result_extractor_params is None:
                result_extractor_params = dict()

            model_results = result_extractor(model_results, **result_extractor_params)

        plot_length = min(len(x_range), len(model_results))
        if model_specific_limits is not None and model.name in model_specific_limits:
            plot_length = min(plot_length, model_specific_limits[model.name])

        c = None
        if colors is not None:
            c = colors[i]

        plt.plot(x_range[:plot_length], model_results[:plot_length], label=model_labels[model.name], c=c)

    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(results_name, fontsize=font_size)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend(loc='best', fontsize=font_size)

    if save_name is not None:
        plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')

    plt.show()




