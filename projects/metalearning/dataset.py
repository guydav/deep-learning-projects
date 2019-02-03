import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import h5py
import os
import pickle


META_LEARNING_DATA = 'drive/Research Projects/Meta-Learning/v1/CLEVR_meta_learning_uint8_desc.h5'
META_LEARNING_DATA_SMALL = 'drive/Research Projects/Meta-Learning/v1/CLEVR_meta_learning_small_uint8.h5'
BATCH_SIZE = 512  # 64
NUM_WORKERS = 1

DOWNSAMPLE_SIZE = (96, 128)
DEFAULT_TRAIN_PROPORTION = 0.9

NORMALIZATION_CACHE_FILE = 'normalization_cache.pickle'


class MetaLearningH5Dataset(Dataset):
    def __init__(self, in_file, transform=None, start_index=0,
                 end_index=None, query_subset=None, return_indices=True):
        super(MetaLearningH5Dataset, self).__init__()
        self.file = h5py.File(in_file, 'r')
        self.transform = transform

        self.start_index = start_index
        self.end_index = end_index
        if self.end_index is None:
            self.end_index = self.file['X'].shape[0]

        self.num_images = self.end_index - self.start_index
        self.query_length = self.file['Q'].shape[2]
        self.total_queries_per_image = self.file['Q'].shape[1]

        if query_subset is None:
            query_subset = np.arange(self.total_queries_per_image)

        self.query_subset = query_subset
        self.active_queries_per_image = len(self.query_subset)

        self.return_indices = return_indices

    def _compute_indices(self, index):
        image_index = self.start_index + (index // self.active_queries_per_image)
        query_index = index % self.active_queries_per_image
        # index from the query to the subset
        actual_query_index = self.query_subset[query_index]
        return image_index, actual_query_index

    def __getitem__(self, index):
        image_index, query_index = self._compute_indices(index)

        x = self.file['X'][image_index, ...]
        q = self.file['Q'][image_index, query_index, ...]
        y = self.file['y'][image_index, query_index]

        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)

        if self.return_indices:
            return (x, y, q), index

        return x, y, q

    def __len__(self):
        return self.num_images * self.active_queries_per_image


class MetaLearningH5DatasetFromDescription(MetaLearningH5Dataset):
    def __init__(self, in_file, transform=None, start_index=0,
                 end_index=None, query_subset=None, return_indices=True,
                 num_dimensions=2, features_per_dimension=(10, 11)):
        super(MetaLearningH5DatasetFromDescription, self).__init__(
            in_file, transform, start_index, end_index, query_subset, return_indices)

        self.num_dimensions = num_dimensions
        self.features_per_dimension = features_per_dimension

    def __getitem__(self, index):
        image_index, query_index = self._compute_indices(index)

        x = self.file['X'][image_index, ...]
        # TODO: for simple queries I can use y, but I will compute y, because I'll need to later
        # y = self.file['y'][image_index, actual_query_index]

        desc = self.file['D'][image_index]
        y = int(np.any(desc == query_index))
        q = np.zeros((self.total_queries_per_image,))
        q[query_index] = 1

        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)

        if self.return_indices:
            return (x, y, q), index

        return x, y, q


def create_normalized_datasets(dataset_path=META_LEARNING_DATA, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                               dataset_train_prop=DEFAULT_TRAIN_PROPORTION,
                               pin_memory=True, downsample_size=DOWNSAMPLE_SIZE,
                               should_flip=True,
                               shuffle=True, query_subset=None, return_indices=False,
                               dataset_class=MetaLearningH5DatasetFromDescription,
                               dataset_class_kwargs=None):

    full_dataset = dataset_class(dataset_path, query_subset=query_subset, return_indices=return_indices)
    test_train_split_index = int(full_dataset.num_images * dataset_train_prop)
    print(f'Splitting test-train at {test_train_split_index}')
    del full_dataset

    if dataset_class_kwargs is None:
        dataset_class_kwargs = {}

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    cache_path = os.path.join(__location__, NORMALIZATION_CACHE_FILE)

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as cache_file:
            cache = pickle.load(cache_file)

    else:
        cache = {}

    cache_key = (dataset_path, dataset_train_prop, downsample_size)

    if cache_key in cache:
        print('Loaded normalization from cache')
        channel_means, channel_stds = cache[cache_key]

    else:

        to_tensor = transforms.ToTensor()

        # TODO: why did I have to_tensor.float()? Add it back in later?
        if downsample_size is not None:
            to_pil = transforms.ToPILImage()
            resize = transforms.Resize(downsample_size)
            unnormalized_transformer = transforms.Compose([
                to_pil,
                resize,
                to_tensor
            ])
        else:
            unnormalized_transformer = to_tensor

        unnormalized_train_dataset = dataset_class(dataset_path, unnormalized_transformer,
                                                   end_index=test_train_split_index,
                                                   query_subset=query_subset,
                                                   return_indices=return_indices)

        transformed_images = np.stack([unnormalized_transformer(image).numpy() for image in
                                       unnormalized_train_dataset.file['X']])
        channel_means = np.mean(transformed_images, (0, 2, 3))
        channel_stds = np.std(transformed_images, (0, 2, 3))
        del unnormalized_train_dataset

        cache[cache_key] = channel_means, channel_stds
        with open(cache_path, 'w') as cache_file:
            pickle.dump(cache, cache_file)

    print(channel_means)
    print(channel_stds)

    normalizer = transforms.Normalize(torch.from_numpy(channel_means),
                                      torch.from_numpy(channel_stds))

    train_transforms = []
    test_transforms = []

    if downsample_size is not None:
        train_transforms.extend([to_pil, resize])
        test_transforms.extend([to_pil, resize])

    if should_flip:
        train_transforms.extend(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip()])

    train_transforms.extend([to_tensor, normalizer])
    test_transforms.extend([to_tensor, normalizer])

    train_transformer = transforms.Compose(train_transforms)
    test_transformer = transforms.Compose(test_transforms)

    normalized_train_dataset = dataset_class(dataset_path, train_transformer,
                                             end_index=test_train_split_index,
                                             query_subset=query_subset,
                                             return_indices=return_indices, **dataset_class_kwargs)
    train_dataloader = DataLoader(normalized_train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    normalized_test_dataset = dataset_class(dataset_path, test_transformer,  # augment only in train
                                            start_index=test_train_split_index,
                                            query_subset=query_subset,
                                            return_indices=return_indices, **dataset_class_kwargs)
    test_dataloader = DataLoader(normalized_test_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return normalized_train_dataset, train_dataloader, normalized_test_dataset, test_dataloader
