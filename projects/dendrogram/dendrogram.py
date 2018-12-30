import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from itertools import combinations_with_replacement

import torch
import torch.nn as nn
import numpy as np


class DendrogramLoss(nn.Module):
    def __init__(self, dendrogram, classes, distances=None, distance_increment=1.0):
        super(DendrogramLoss, self).__init__()

        self.register_buffer('distance_increment', torch.tensor(distance_increment).float())

        if type(dendrogram) == dict:
            key = next(iter(dendrogram))
            val = dendrogram[key]
            if type(val) == dict:
                self.dendrogram = nx.from_dict_of_dicts(dendrogram)
            elif type(val) in (list, tuple):
                self.dendrogram = nx.from_dict_of_lists(dendrogram)
            else:
                raise ValueError(f'Received a dendrogram dictionary with unknown value type: {type(val)}')

        elif type(dendrogram) == nx.classes.graph.Graph:
            self.dendrogram = dendrogram

        else:
            raise ValueError(f'Received a dendrogram of unknown type: {type(dendrogram)}')

        self.classes = classes
        if distances is not None:
            self.distances = distances
        else:
            self._dendrogram_to_distances()

    def _dendrogram_to_distances(self):
        all_distances = dict(nx.floyd_warshall(self.dendrogram))
        num_classes = len(self.actual_classes)
        distances = np.zeros((num_classes, num_classes))

        for i, j in combinations_with_replacement(np.arange(num_classes), 2):
            distances[i, j] = distances[j, i] = all_distances[self.classes[i]][self.classes[j]]

        # adding 1 since this is a multiplicative factor, and the factor for a node * itself should be one
        # this creates ones on the diagonal and > 1 everywhere else
        # TODO: do I need to move this to a device?
        distances = torch.from_numpy(distances).float() + self.distance_increment
        self.register_buffer('distances', distances)

    """
    def plot_dendrogram(self):
        plt.figure(figsize=(12, 8))
        layout = graphviz_layout(self.dendrogram)
        node_colors = [0.8 - 0.4 * float(n in self.classes) for n in graph.nodes()]
        labels = nx.get_edge_attributes(self.dendrogram, 'weight')
        nx.draw_networkx_edge_labels(self.dendrogram, layout, edge_labels=labels)
        nx.draw(self.dendrogram, layout, with_labels=True, node_color=node_colors, node_size=2000,
                cmap=plt.cm.spectral, linewidths=3, font_color='c', font_size=14)
    """

    def forward(self, output, labels):
        # output are the [batch size] x [num classes] softmax of predictions
        # labels are the [batch size] x [num classes] binarized labels
        square_errors = torch.pow(output - labels, 2)
        weighted_errors = torch.mul(square_errors, torch.matmul(labels, self.distances))
        return weighted_errors.mean()


class HingeDendrogramLoss(DendrogramLoss):
    def __init__(self, dendrogram, classes, distances=None, distance_increment=1.0, p=1, margin=1.0):
        super(HingeDendrogramLoss, self).__init__(dendrogram, classes, distances, distance_increment)
        self.p = p
        self.margin = margin

    def forward(self, output, labels):
        # output are the [batch size] x [num classes] softmax of predictions
        # labels are the [batch size] x [1] label indices
        correct_class_scores = output.gather(1, labels.view(-1, 1))
        # weigh each score by the correct distances
        margin_scores = torch.pow(torch.clamp(self.margin + output - correct_class_scores, min=0), self.p) * \
                                  self.distances[labels]
        # we don't want to index through the correct class, which will always contribute a penalty of self.margin
        # Thus, before averaging for each example, we subtract out the determinstic score for the correct class
        # Note that since for the correct class the distance is always one, we dont have to account for it when subtracting
        per_example_loss = (margin_scores.sum(1) - self.margin ** self.p) / output.shape[1]
        # assuming mean reduction, as is PyTorch's deafult
        return per_example_loss.mean()


class HingeDendrogramMarginLoss(DendrogramLoss):
    def __init__(self, dendrogram, classes, distances=None, distance_increment=0.0, p=1, distance_scale=1.0):
        super(HingeDendrogramMarginLoss, self).__init__(dendrogram, classes, distances, distance_increment)
        self.p = p
        self.distances = self.distances / distance_scale

    def forward(self, output, labels):
        # output are the [batch size] x [num classes] softmax of predictions
        # labels are the [batch size] x [1] label indices
        correct_class_scores = output.gather(1, labels.view(-1, 1))
        # weigh each score by the correct distances
        margin = self.distances[labels]
        margin_scores = torch.pow(torch.clamp(margin + output - correct_class_scores, min=0), self.p)
        # we don't want to index through the correct class, which will always contribute a penalty of self.margin
        # In this case, it's not a problem, since the margin for the correct class is zero,
        # and the output is equal the correct class score, so the correct class never contributes to the loss
        per_example_loss = margin_scores.sum(1) / output.shape[1]
        # assuming mean reduction, as is PyTorch's deafult
        return per_example_loss.mean()


DEFAULT_EDGE_DICTS = {
    'root': {'transport': {'weight': 1}, 'animal': {'weight': 1} },
    'transport': {'air': {'weight': 1}, 'land': {'weight': 1}, 'sea': {'weight': 1} },
    'air': {'airplane': {'weight': 1}},
    'sea': {'ship': {'weight': 1}},
    'land': {'auto': {'weight': 0.5}, 'truck': {'weight': 0.5}},
    'animal': {'mammal': {'weight': 1}, 'bird': {'weight': 2}, 'frog': {'weight': 2}},
    'mammal': {'domsetic': {'weight': 0.5}, 'deer': {'weight': 1}, 'horse': {'weight': 1}},
    'domsetic': {'cat': {'weight': 0.5}, 'dog': {'weight': 0.5}}
}

ALPHABETICAL_EDGE_DICTS = {
    'root': {'a-d': {'weight': 1}, 'f-z': {'weight': 1} },
    'a-d': {'a-c': {'weight': 1}, 'd': {'weight': 1}},
    'a-c': {'bird': {'weight': 1}, 'auto': {'weight': 1}, 'cat': {'weight': 1}},
    'd': {'deer': {'weight': 1}, 'dog': {'weight': 1}},
    'f-z': {'f-p': {'weight': 1}, 'q-z': {'weight': 1}},
    'f-p': {'frog': {'weight': 1}, 'horse': {'weight': 1}, 'airplane': {'weight': 1}},
    'q-z': {'ship': {'weight': 1}, 'truck': {'weight': 1}}
}


DEFAULT_CLASSES = ('airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dendrogram_loss = DendrogramLoss(DEFAULT_EDGE_DICTS, DEFAULT_CLASSES).cuda()
alphabetical_dendrogram_loss = DendrogramLoss(ALPHABETICAL_EDGE_DICTS, DEFAULT_CLASSES).cuda()
