import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import h5py
import os
import pickle
from collections import defaultdict
import itertools
from datetime import datetime


META_LEARNING_DATA = 'drive/Research Projects/Meta-Learning/v1/CLEVR_meta_learning_uint8_desc.h5'
META_LEARNING_DATA_SMALL = 'drive/Research Projects/Meta-Learning/v1/CLEVR_meta_learning_small_uint8.h5'
BATCH_SIZE = 512  # 64
NUM_WORKERS = 1

DOWNSAMPLE_SIZE = (96, 128)
DEFAULT_TRAIN_PROPORTION = 0.9

DATASET_CACHE_FILE = 'dataset_cache.pickle'


class MetaLearningH5Dataset(Dataset):
    def __init__(self, in_file, transform=None, start_index=0,
                 end_index=None, query_subset=None, return_indices=True):
        super(MetaLearningH5Dataset, self).__init__()

        self.in_file = in_file
        self.file = None
        self.transform = transform
        self.start_index = start_index
        self.end_index = end_index

        with h5py.File(in_file, 'r') as file:
            if self.end_index is None:
                self.end_index = file['X'].shape[0]

            self.num_images = self.end_index - self.start_index
            self.query_length = file['Q'].shape[2]
            self.total_queries_per_image = file['Q'].shape[1]

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

        if self.file is None:
            self.file = h5py.File(self.in_file, 'r')

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
                 num_dimensions=3, features_per_dimension=(10, 10, 10)):
        super(MetaLearningH5DatasetFromDescription, self).__init__(
            in_file, transform, start_index, end_index, query_subset, return_indices)

        self.num_dimensions = num_dimensions
        self.features_per_dimension = features_per_dimension

    def __getitem__(self, index):
        image_index, query_index = self._compute_indices(index)

        if self.file is None:
            self.file = h5py.File(self.in_file, 'r')

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


def debug_print(message):
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {message}')


class SequentialBenchmarkMetaLearningDataset(MetaLearningH5DatasetFromDescription):
    def __init__(self, in_file, benchmark_dimension, random_seed,
                 previous_query_coreset_size, query_order, coreset_size_per_query=False,
                 transform=None, start_index=0, end_index=None, return_indices=True,
                 num_dimensions=3, features_per_dimension=(10, 10, 10),
                 imbalance_threshold=0.25, num_sampling_attempts=10):
        """
        Important API-wise:
        Call start_epoch whenever, well, you're starting a new epoch
        Call next_query whenever criterion is reached for the current query
        :param in_file:
        :param benchmark_dimension:
        :param random_seed:
        :param previous_query_coreset_size:
        :param query_order:
        :param transform:
        :param start_index:
        :param end_index:
        :param return_indices:
        :param num_dimensions:
        :param features_per_dimension:
        """
        super(SequentialBenchmarkMetaLearningDataset, self).__init__(
            in_file, transform, start_index, end_index, None, return_indices,
            num_dimensions, features_per_dimension)

        if benchmark_dimension >= num_dimensions:
            raise ValueError(f'Benchmark dimension ({benchmark_dimension}) must be smaller than the number of dimensions ({num_dimensions})')

        self.benchmark_dimension = benchmark_dimension
        self.previous_query_coreset_size = previous_query_coreset_size
        self.coreset_size_per_query = coreset_size_per_query
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.imbalance_threshold = imbalance_threshold
        self.num_sampling_attempts = num_sampling_attempts

        if len(query_order) != features_per_dimension[benchmark_dimension]:
            raise ValueError(f'The length of the query order {query_order} => {len(query_order)} must be equal to the number of features in that dimension ({features_per_dimension[benchmark_dimension]})')

        dimension_sizes = list(np.cumsum(features_per_dimension))
        dimension_sizes.insert(0, 0)

        if np.any(np.array(query_order) < dimension_sizes[benchmark_dimension]) or \
                np.any(np.array(query_order) >= dimension_sizes[benchmark_dimension + 1]):
            raise ValueError(
                f'The query order {query_order} for dimension {benchmark_dimension} must be between {dimension_sizes[benchmark_dimension]} and {dimension_sizes[benchmark_dimension + 1] - 1}')

        self.query_order = query_order
        self.current_query_index = 0
        self._cache_images_by_query()

        self.current_epoch_queries = []
        self.start_epoch()

    def _cache_images_by_query(self):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        cache_path = os.path.join(__location__, DATASET_CACHE_FILE)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                cache = pickle.load(cache_file)

        else:
            cache = {}

        positive_cache_key = (self.in_file, 'per_query_positive')
        negative_cache_key = (self.in_file, 'per_query_negative')

        if positive_cache_key in cache and negative_cache_key in cache:
            print('Loading positive and negative images from cache')
            self.positive_images = cache[positive_cache_key]
            self.negative_images = cache[negative_cache_key]

        else:
            self.positive_images = defaultdict(list)
            self.negative_images = defaultdict(list)

            with h5py.File(self.in_file, 'r') as file:
                y = file['y']
                for i in range(y.shape[0]):
                    for q in range(y.shape[1]):
                        if y[i, q] == 1:
                            self.positive_images[q].append(i)
                        else:
                            self.negative_images[q].append(i)

            cache[positive_cache_key] = self.positive_images
            cache[negative_cache_key] = self.negative_images

            with open(cache_path, 'wb') as cache_file:
                pickle.dump(cache, cache_file)

    def __len__(self):
        # if self.coreset_size_per_query:
        #     return self.current_query_index * self.previous_query_coreset_size + self.num_images
        #
        # # After the first task, we work with the entire set
        # if self.current_query_index > 0:
        #     return self.num_images
        #
        # # For the first task, there's no coreset
        # return self.num_images - self.previous_query_coreset_size
        return len(self.current_epoch_queries)

    def next_query(self):
        self.current_query_index += 1

    def start_epoch(self, debug=False, depth=0):
        """
        Sample the images for each coreset query to be used for the current epoch
        """
        self.current_epoch_queries = []

        if debug: debug_print('Starting start_epoch')

        if depth >= self.num_sampling_attempts:
            print('Warning, exceeded maximum number of sampling attempts, this is not great')
            # TODO: should I abort here?

        if not self.coreset_size_per_query:
            current_task_set = set(np.random.choice(self.num_images,
                                                    self.num_images - self.previous_query_coreset_size,
                                                    False))
            coreset = set(range(self.num_images)).difference(current_task_set)

            if self.current_query_index > 0:
                query_coreset_sizes = np.array([int(self.previous_query_coreset_size * i / self.current_query_index)
                                                for i in range(self.current_query_index + 1)])
                query_coreset_sizes = query_coreset_sizes[1:] - query_coreset_sizes[:-1]

        for previous_query_index in range(self.current_query_index):
            previous_query = self.query_order[previous_query_index]

            if debug: debug_print(f'Starting previous query #{previous_query_index} = {previous_query}')

            # This would happen in our test loader:
            if self.previous_query_coreset_size == self.num_images:
                self.current_epoch_queries.extend(list(zip(range(self.num_images),
                                                           itertools.cycle([previous_query]))))

            else:
                if self.coreset_size_per_query:
                    positive_size = self.previous_query_coreset_size // 2
                    negative_size = positive_size
                    positive_queries = np.random.choice(self.positive_images[previous_query], positive_size, False)
                    negative_queries = np.random.choice(self.negative_images[previous_query], negative_size, False)
                    current_task_coreset = np.concatenate((positive_queries, negative_queries))

                else:  # shared coreset among all queries
                    current_coreset_size = query_coreset_sizes[previous_query_index]

                    smaller_proportion = 0
                    attempt_count = 0

                    while smaller_proportion < self.imbalance_threshold \
                            and attempt_count < self.num_sampling_attempts:

                        attempt_count += 1
                        if debug: debug_print(f'Starting attempt #{attempt_count}')
                        coreset_list = list(coreset)
                        if debug: debug_print(f'Casted coreset to a list')
                        current_task_coreset = np.random.choice(coreset_list, current_coreset_size, False)
                        if debug: debug_print(f'Sampled current task coreset')
                        positive_count = sum([x in self.positive_images[previous_query] for x in current_task_coreset])
                        negative_count = sum([x in self.negative_images[previous_query] for x in current_task_coreset])
                        smaller_proportion = min(positive_count / current_coreset_size,
                                                 negative_count / current_coreset_size)
                        if debug: debug_print(f'Computed counts and proportions')

                    if attempt_count >= self.num_sampling_attempts:
                        print(f'Warning, failed to balance query #{previous_query_index + 1}, restarting...')
                        return self.start_epoch(debug, depth + 1)

                    if debug: debug_print(f'Successfully generated coreset for current task')
                    coreset = coreset.difference(set(current_task_coreset))

                self.current_epoch_queries.extend(list(zip(current_task_coreset, itertools.cycle([previous_query]))))

        if self.coreset_size_per_query:  # use the entire training set for the previous query
            self.current_epoch_queries.extend(list(zip(range(self.num_images),
                                                   itertools.cycle([self.query_order[self.current_query_index]]))))
        else:
            self.current_epoch_queries.extend(list(zip(current_task_set,
                                                       itertools.cycle([self.query_order[self.current_query_index]]))))

        # print(self.current_query_index, len(self), len(self.current_epoch_queries))

    def _compute_indices(self, index):
        return self.current_epoch_queries[index]


def create_normalized_datasets(dataset_path=META_LEARNING_DATA, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                               dataset_train_prop=DEFAULT_TRAIN_PROPORTION,
                               pin_memory=True, downsample_size=DOWNSAMPLE_SIZE,
                               should_flip=True,
                               shuffle=True, return_indices=False,
                               dataset_class=MetaLearningH5DatasetFromDescription,
                               dataset_class_kwargs=None, train_dataset_kwargs=None, test_dataset_kwargs=None,
                               normalization_dataset_class=MetaLearningH5DatasetFromDescription):

    full_dataset = normalization_dataset_class(dataset_path,return_indices=return_indices)
    test_train_split_index = int(full_dataset.num_images * dataset_train_prop)
    print(f'Splitting test-train at {test_train_split_index}')
    del full_dataset

    if dataset_class_kwargs is None:
        dataset_class_kwargs = {}

    if train_dataset_kwargs is None:
        train_dataset_kwargs = {}

    if test_dataset_kwargs is None:
        test_dataset_kwargs = {}

    # Using this instead of dictionary.update to make sure values in the specific kwargs take precedence
    for key in dataset_class_kwargs:
        if key not in train_dataset_kwargs:
            train_dataset_kwargs[key] = dataset_class_kwargs[key]

        if key not in test_dataset_kwargs:
            test_dataset_kwargs[key] = dataset_class_kwargs[key]

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    cache_path = os.path.join(__location__, DATASET_CACHE_FILE)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            cache = pickle.load(cache_file)

    else:
        cache = {}

    cache_key = (dataset_path, dataset_train_prop, downsample_size)
    to_tensor = transforms.ToTensor()

    if cache_key in cache:
        print('Loaded normalization from cache')
        channel_means, channel_stds = cache[cache_key]

    else:
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

        unnormalized_train_dataset = normalization_dataset_class(dataset_path, unnormalized_transformer,
                                                                 end_index=test_train_split_index,
                                                                 return_indices=return_indices)

        transformed_images = np.stack([unnormalized_transformer(image).numpy() for image in
                                       unnormalized_train_dataset.file['X']])
        channel_means = np.mean(transformed_images, (0, 2, 3))
        channel_stds = np.std(transformed_images, (0, 2, 3))
        del unnormalized_train_dataset

        cache[cache_key] = channel_means, channel_stds
        with open(cache_path, 'wb') as cache_file:
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

    normalized_train_dataset = dataset_class(dataset_path, transform=train_transformer,
                                             end_index=test_train_split_index,
                                             return_indices=return_indices, **train_dataset_kwargs)
    train_dataloader = DataLoader(normalized_train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    normalized_test_dataset = dataset_class(dataset_path, transform=test_transformer,  # augment only in train
                                            start_index=test_train_split_index,
                                            return_indices=return_indices, **test_dataset_kwargs)
    test_dataloader = DataLoader(normalized_test_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return normalized_train_dataset, train_dataloader, normalized_test_dataset, test_dataloader
