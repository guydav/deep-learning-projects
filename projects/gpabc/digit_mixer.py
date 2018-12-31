from sklearn import datasets as sk_datasets
from torchvision import datasets, transforms
import numpy as np
import torch
from .vae import BATCH_SIZE, kwargs


class TorchDigitMixer:
    def __init__(self, digits, weights, noise_std=0):
        self.digits = digits
        self.weights = np.array(weights)

        self.train_mnist = datasets.MNIST('../data', train=True, download=True,
                                          transform=transforms.ToTensor())
        self.train_indices = self._digit_indices(self.train_mnist.train_labels)

        self.test_mnist = datasets.MNIST('../data', train=False, download=True,
                                         transform=transforms.ToTensor())
        self.test_indices = self._digit_indices(self.test_mnist.test_labels)

        self.noise_std = noise_std

    def _digit_indices(self, target):
        return {digit: np.where(target == digit)[0] for digit in range(10)}

    def __call__(self, n=1, train=True):
        if train:
            data = self.train_mnist
            indices = self.train_indices
        else:
            data = self.test_mnist
            indices = self.test_indices

        image_indices = [np.random.choice(indices[d], n) for d in self.digits]
        # i = 0; j = 0; print(torch.squeeze(data[image_indices[i][j]]))

        images_per_digit = [torch.stack([torch.squeeze(data[image_indices[i][j]][0]) for j in range(n)])
                            for i in range(len(self.digits))]
        weighted_images = np.array([images_per_digit[i].numpy() * self.weights[i]
                                    for i in range(self.weights.shape[0])])
        images = np.sum(weighted_images, axis=0)
        if self.noise_std != 0:
            images += np.random.normal(0, self.noise_std, images.shape)
        return images


SKLEARN_MNIST = sk_datasets.fetch_mldata('MNIST original')
MNIST_TRAIN_END = 60000


class SklearnDigitMixer:
    def __init__(self, digits, weights, noise_std=0):
        self.digits = digits
        self.weights = np.array(weights)

        self.mnist = SKLEARN_MNIST
        self.train_data = self.mnist.data[:MNIST_TRAIN_END]
        self.train_indices = self._digit_indices(self.mnist.target[:MNIST_TRAIN_END])

        self.test_data = self.mnist.data[MNIST_TRAIN_END:]
        self.test_indices = self._digit_indices(self.mnist.target[MNIST_TRAIN_END:])

        self.noise_std = noise_std

    def _digit_indices(self, target):
        return {digit: np.where(target == digit)[0] for digit in range(10)}

    def __call__(self, n=1, train=True):
        if train:
            data = self.train_data
            indices = self.train_indices
        else:
            data = self.test_data
            indices = self.test_indices

        image_indices = [np.random.choice(indices[d], n) for d in self.digits]
        weighted_images = np.array([data[image_indices[i]].astype(np.float32) / 255.0 * self.weights[i]
                                    for i in range(self.weights.shape[0])])
        images = np.sum(weighted_images, axis=0)
        if self.noise_std != 0:
            images += np.random.normal(0, self.noise_std, images.shape)
        return images


from torch.utils.data import Dataset, DataLoader


class DigitMixerDataset(Dataset):
    def __init__(self, data, target, noise_std=0, num_digits=2,
                 p_min=0.1, p_max=0.9,
                 epoch_length=60000, valid_digits=None, seed=0):
        super(DigitMixerDataset, self).__init__()

        self.data = data
        self.digit_indices = self._digit_indices(target)
        self.noise_std = 0
        self.num_digits = num_digits
        self.p_min = p_min
        self.p_max = p_max
        self.epoch_length = epoch_length
        if valid_digits is None:
            valid_digits = np.arange(10)
        self.valid_digits = valid_digits
        self.seed = 0
        self.epoch = 0

    def _digit_indices(self, target):
        return {digit: np.where(target == digit)[0] for digit in range(10)}

    def epoch_end(self):
        self.epoch += 1

    def __getitem__(self, index):
        # Seeding to make sure the combination of item x epoch is repeatable
        np.random.seed(self.seed + index + self.epoch * self.epoch_length)
        digits = np.random.choice(self.valid_digits, self.num_digits, replace=False)
        # TODO: if need be, extend probability sampling for self.num_digits > 2
        p = np.random.uniform(self.p_min, self.p_max)
        probs = (p, 1 - p)

        image_indices = [np.random.choice(self.digit_indices[d]) for d in digits]
        weighted_images = np.array([self.data[image_indices[i]].astype(np.float32) / 255.0 * probs[i]
                                    for i in range(self.num_digits)])
        image = np.sum(weighted_images, axis=0)
        if self.noise_std != 0:
            image += np.random.normal(0, self.noise_std, image.shape)

        image_tensor = torch.tensor(image, dtype=torch.float)
        return image_tensor, index

    def __len__(self):
        return self.epoch_length


train_data = SKLEARN_MNIST.data[:MNIST_TRAIN_END]
train_target = SKLEARN_MNIST.target[:MNIST_TRAIN_END]
test_data = SKLEARN_MNIST.data[MNIST_TRAIN_END:]
test_target = SKLEARN_MNIST.target[MNIST_TRAIN_END:]

mixed_digit_train_dataset = DigitMixerDataset(train_data, train_target)
mixed_digit_train_loader = torch.utils.data.DataLoader(mixed_digit_train_dataset,
                                                       batch_size=BATCH_SIZE, shuffle=True, **kwargs)

mixed_digit_test_dataset = DigitMixerDataset(test_data, test_target)
mixed_digit_test_loader = torch.utils.data.DataLoader(mixed_digit_test_dataset,
                                                      batch_size=BATCH_SIZE, shuffle=True, **kwargs)
