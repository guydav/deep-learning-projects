import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import h5py
import os
import pickle
from collections import defaultdict, OrderedDict
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
    """
    The default dataset class for the meta-learning dataset we created.
    Loads the data from an HDF5 file, computes which effective indices are valid
    (as the number of total valid indices is [number of images] x [number of queries used],
    and serves images accordingly (see the __getitem__ method)
    """
    def __init__(self, in_file, transform=None, start_index=0,
                 end_index=None, query_subset=None, return_indices=True):
        """
        Initialize a new dataset.
        :param in_file: The path to read the dataset from
        :param transform: Whether or not to apply any transformations to the images before returning them
        :param start_index: Which image to start reading from; used for test-train splits; default 0
        :param end_index: Which image to stop reading from; used for test-train splits;
            default None meaning "end of the file"
        :param query_subset: Which subset of queries to use, if not using all queries;
            default None which means "all queries"
        :param return_indices: Whether or not to return the requested indices along with the image; default True
        """
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
        """
        Compute the image and query index from the requested index. We treat the image index as $index // num_queries$
        and the query index as $index \mod num_queries$, thus every contiguous set of num_query indices correspond to
        the same image.
        :param index: The index to retrieve
        :return: The real indices of the image and query
        """
        image_index = self.start_index + (index // self.active_queries_per_image)
        query_index = index % self.active_queries_per_image
        # index from the query to the subset
        actual_query_index = self.query_subset[query_index]
        return image_index, actual_query_index

    def __getitem__(self, index):
        """
        Return a transformed image, query, and ground truth answer corresponding to an index.
        :param index: Which index to return from
        :return: The input image, transformed if a transformer was set during initialization,
            The query corresponding to this index,
            The ground truth answer for this query on this image
            The input index, if initalized with this option
        """
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
            return x, y, q, index

        return x, y, q

    def __len__(self):
        return self.num_images * self.active_queries_per_image


class MetaLearningH5DatasetFromDescription(MetaLearningH5Dataset):
    """
    Entirely the same as its super class, but computing the corect answer from the image descriptions saved in the
    dataset, rather than from the hard-coded query answer. This is setup for the compostional benchmark (and other
    two-feature queries), where we wouldn't want to save the answers for all 300 possible two-item queries.
    """
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
            return x, y, q, index

        return x, y, q


def debug_print(message):
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {message}')


class SequentialBenchmarkMetaLearningDataset(MetaLearningH5DatasetFromDescription):
    def __init__(self, in_file, benchmark_dimension, random_seed,
                 previous_query_coreset_size, query_order, single_dimension=True,
                 coreset_size_per_query=False, transform=None,
                 start_index=0, end_index=None, return_indices=True,
                 num_dimensions=3, features_per_dimension=(10, 10, 10),
                 imbalance_threshold=0.2, num_sampling_attempts=20):
        """
        Dataset class for the sequential benchmark. Samples coreset images according to the description in the paper.
        During the first episode, returns 22,500 images for the current task. During every subsequent episodes, returns
        22,500 images for the current task (or query), and splits the rest evenly between previous tasks as the coreset.

        Important API-wise:
        Call start_epoch whenever, well, you're starting a new epoch
        Call next_query whenever criterion is reached for the current query and you're starting a new episode
        :param in_file: Which HDF5 file to load the dataset from
        :param benchmark_dimension: Which dimension the benchmark is running on, in the sequential, homogeneous
            dimension condition; None if it is teh sequential, heterogeneous dimensions condition
        :param random_seed: Which random seed to use for sampling purposes.
        :param previous_query_coreset_size: How many images to allocate to the coreset used for previous queries.
        :param query_order: Which query to introduce at which point in the benchmark
        :param transform: Whether or not to apply any transformations to the images before returning them
        :param start_index: Which image to start reading from; used for test-train splits; default 0
        :param end_index: Which image to stop reading from; used for test-train splits;
            default None meaning "end of the file"
        :param query_subset: Which subset of queries to use, if not using all queries;
            default None which means "all queries"
        :param return_indices: Whether or not to return the requested indices along with the image; default True
        :param num_dimensions: how many dimensions exist; default 3
        :param features_per_dimension: how many features exist in each dimension; default 10 each
        """
        super(SequentialBenchmarkMetaLearningDataset, self).__init__(
            in_file, transform, start_index, end_index, None, return_indices,
            num_dimensions, features_per_dimension)

        # In the case it's a single-dimension example, validate parameters
        if single_dimension:
            if benchmark_dimension >= num_dimensions:
                raise ValueError(f'Benchmark dimension ({benchmark_dimension}) must be smaller than the number of dimensions ({num_dimensions})')

            if len(query_order) != features_per_dimension[benchmark_dimension]:
                raise ValueError(
                    f'The length of the query order {query_order} => {len(query_order)} must be equal to the number of features in that dimension ({features_per_dimension[benchmark_dimension]})')

            dimension_sizes = list(np.cumsum(features_per_dimension))
            dimension_sizes.insert(0, 0)

            if np.any(np.array(query_order) < dimension_sizes[benchmark_dimension]) or \
                    np.any(np.array(query_order) >= dimension_sizes[benchmark_dimension + 1]):
                raise ValueError(
                    f'The query order {query_order} for dimension {benchmark_dimension} must be between {dimension_sizes[benchmark_dimension]} and {dimension_sizes[benchmark_dimension + 1] - 1}')

        self.previous_query_coreset_size = previous_query_coreset_size
        self.coreset_size_per_query = coreset_size_per_query
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.imbalance_threshold = imbalance_threshold
        self.num_sampling_attempts = num_sampling_attempts
        self.query_order = query_order
        self.current_query_index = 0
        self._cache_images_by_query()
        self.current_epoch_queries = []

        self.start_epoch()

    def _cache_images_by_query(self):
        """
        Cache which images are positive and which are negative for each query, to allow for balanced coresets.
        This computation should happen once per dataset, and then be loaded from the cache.
        :return:
        """
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
            self.positive_images = defaultdict(set)
            self.negative_images = defaultdict(set)

            with h5py.File(self.in_file, 'r') as file:
                y = file['y']
                for i in range(y.shape[0]):
                    for q in range(y.shape[1]):
                        if y[i, q] == 1:
                            self.positive_images[q].add(i)
                        else:
                            self.negative_images[q].add(i)

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
        """
        This does not actualyl do much, other than increment the current_query_index. The reason is that start_epoch
        reads that variable and will use this new value
        """
        self.current_query_index += 1

    def _allocate_images_to_tasks(self, depth=0):
        task_to_images = OrderedDict()

        if depth >= self.num_sampling_attempts:
            raise ValueError('Warning, exceeded maximum number of sampling attempts, this is not great')

        if not self.coreset_size_per_query:
            image_set = set(range(self.num_images))

            if self.current_query_index > 0:
                query_coreset_sizes = np.array([int(self.previous_query_coreset_size * i / self.current_query_index)
                                                for i in range(self.current_query_index + 1)])
                query_coreset_sizes = query_coreset_sizes[1:] - query_coreset_sizes[:-1]

        for previous_query_index in range(self.current_query_index):
            previous_query = self.query_order[previous_query_index]

            # This would happen in our test loader:
            if self.previous_query_coreset_size == self.num_images:
                task_to_images[previous_query] = range(self.num_images)

            else:
                if self.coreset_size_per_query:
                    positive_size = self.previous_query_coreset_size // 2
                    negative_size = positive_size
                    positive_queries = np.random.choice(list(self.positive_images[previous_query]), positive_size,
                                                        False)
                    negative_queries = np.random.choice(list(self.negative_images[previous_query]), negative_size,
                                                        False)
                    current_task_coreset = np.concatenate((positive_queries, negative_queries))

                else:  # shared coreset among all queries
                    current_coreset_size = query_coreset_sizes[previous_query_index]

                    smaller_proportion = 0
                    attempt_count = 0

                    while smaller_proportion < self.imbalance_threshold \
                            and attempt_count < self.num_sampling_attempts:
                        attempt_count += 1
                        image_list = list(image_set)
                        current_task_coreset = np.random.choice(image_list, current_coreset_size, False)

                        # negative_count = sum([x in self.negative_images[previous_query]
                        # for x in current_task_coreset])
                        positive_count = sum([x in self.positive_images[previous_query] for x in current_task_coreset])
                        smaller_proportion = min(positive_count / current_coreset_size,
                                                 1 - (positive_count / current_coreset_size))

                    if attempt_count >= self.num_sampling_attempts:
                        print(f'Warning, failed to balance query #{previous_query_index + 1}, restarting...')
                        return self._allocate_images_to_tasks(depth + 1)

                    image_set = image_set.difference(set(current_task_coreset))

                task_to_images[previous_query] = current_task_coreset

        current_query = self.query_order[self.current_query_index]

        if self.coreset_size_per_query:  # use the entire training set for the previous query
            task_to_images[current_query] = range(self.num_images)

        else:
            if self.current_query_index == 0:
                image_set = set(np.random.choice(self.num_images,
                                                 self.num_images - self.previous_query_coreset_size,
                                                 False))

            task_to_images[current_query] = image_set

        return task_to_images

    def start_epoch(self, debug=False):
        """
        Sample the images for each coreset query to be used for the current epoch. This supports a number of
        different variations:

        if coreset_size_per_query is True, we supplied a coreset size to be used for all queries, rather than divided
        between them -- in this case, sample a coreset for each query and move on with our life. This mode is used in
        our test-set data loader, with all 5000 images assigned to each query - that is, no actual randomization.

        If coreset_size_per_query is False, we divide the coreset evenly between the previous tasks. We then sample an
        appropriately sized coreset, making sure it is balanced, and after we finish sampling the coresets, we
        assign the remaining images to the current task.
        """
        task_to_images = self._allocate_images_to_tasks()

        self.current_epoch_queries = []
        for task, images in task_to_images.items():
            self.current_epoch_queries.extend(list(zip(images, itertools.cycle([task]))))

    def _compute_indices(self, index):
        return self.current_epoch_queries[index]


class BalancedBatchesMetaLearningDataset(SequentialBenchmarkMetaLearningDataset):
    def __init__(self, in_file, batch_size, benchmark_dimension, random_seed,
                 previous_query_coreset_size, query_order, single_dimension=True,
                 coreset_size_per_query=False, transform=None,
                 start_index=0, end_index=None, return_indices=True,
                 num_dimensions=3, features_per_dimension=(10, 10, 10),
                 imbalance_threshold=0.2, num_sampling_attempts=20):

        super(BalancedBatchesMetaLearningDataset, self).__init__(
            in_file, benchmark_dimension, random_seed, previous_query_coreset_size,
            query_order, single_dimension, coreset_size_per_query, transform,
            start_index, end_index, return_indices, num_dimensions, features_per_dimension,
            imbalance_threshold, num_sampling_attempts)

        self.batch_size = batch_size
        self.num_batches_per_epoch = self.num_images // self.batch_size

    def start_epoch(self, debug=False):
        """
        Sample the images for each coreset query to be used for the current epoch. This supports a number of
        different variations:

        if coreset_size_per_query is True, we supplied a coreset size to be used for all queries, rather than divided
        between them -- in this case, sample a coreset for each query and move on with our life. This mode is used in
        our test-set data loader, with all 5000 images assigned to each query - that is, no actual randomization.

        If coreset_size_per_query is False, we divide the coreset evenly between the previous tasks. We then sample an
        appropriately sized coreset, making sure it is balanced, and after we finish sampling the coresets, we
        assign the remaining images to the current task.
        """
        task_to_images = self._allocate_images_to_tasks()
        for task in task_to_images:
            image_list = list(task_to_images[task])
            np.random.shuffle(image_list)
            task_to_images[task] = image_list

        # if only one task, shuffle its examples, call it a day
        if self.current_query_index == 0:
            first_task = self.query_order[0]
            first_task_images = task_to_images[first_task]
            self.current_epoch_queries = list(zip(first_task_images, itertools.cycle([first_task])))
            return

        # more than one task -- we can deal with the current task first
        # since it occupies half of every epoch
        current_task = self.query_order[self.current_query_index]
        current_task_images = task_to_images[current_task]
        current_task_per_batch = self.batch_size // 2

        batches = [list(zip(current_task_images[i * current_task_per_batch:(i + 1) * current_task_per_batch],
                            itertools.cycle([current_task])))
                   for i in range(self.num_batches_per_epoch)]

        # move onto the previous tasks
        prev_task_per_batch = current_task_per_batch // self.current_query_index
        num_to_round_up = current_task_per_batch % self.current_query_index

        for batch_index in range(self.num_batches_per_epoch):
            tasks_rounding_up = sorted(self.query_order[:self.current_query_index],
                                       key=lambda x: len(task_to_images[x]),
                                       reverse=True)[:num_to_round_up]

            for task in self.query_order[:self.current_query_index]:
                num_task_examples = prev_task_per_batch + 1 * (task in tasks_rounding_up)
                batches[batch_index].extend(list(zip(task_to_images[task][:num_task_examples],
                                                     itertools.cycle([task]))))
                task_to_images[task] = task_to_images[task][num_task_examples:]

        self.current_epoch_queries = [pair for batch in batches for pair in batch]

    def _compute_indices(self, index):
        return self.current_epoch_queries[index]


class ForgettingExperimentMetaLearningDataset(MetaLearningH5DatasetFromDescription):
    def __init__(self, in_file, benchmark_dimension, random_seed,
                 sub_epoch_size, query_order, single_dimension=True,
                 transform=None,
                 start_index=0, end_index=None, return_indices=True,
                 num_dimensions=3, features_per_dimension=(10, 10, 10)):
        """
        Dataset class for the sequential benchmark. Samples coreset images according to the description in the paper.
        During the first episode, returns 22,500 images for the current task. During every subsequent episodes, returns
        22,500 images for the current task (or query), and splits the rest evenly between previous tasks as the coreset.

        Important API-wise:
        Call start_epoch whenever, well, you're starting a new epoch
        Call next_query whenever criterion is reached for the current query and you're starting a new episode
        :param in_file: Which HDF5 file to load the dataset from
        :param benchmark_dimension: Which dimension the benchmark is running on, in the sequential, homogeneous
            dimension condition; None if it is teh sequential, heterogeneous dimensions condition
        :param random_seed: Which random seed to use for sampling purposes.
        :param sub_epoch_size: The size of each 'small' epoch to run
        :param query_order: Which query to introduce at which point in the benchmark
        :param transform: Whether or not to apply any transformations to the images before returning them
        :param start_index: Which image to start reading from; used for test-train splits; default 0
        :param end_index: Which image to stop reading from; used for test-train splits;
            default None meaning "end of the file"
        :param query_subset: Which subset of queries to use, if not using all queries;
            default None which means "all queries"
        :param return_indices: Whether or not to return the requested indices along with the image; default True
        :param num_dimensions: how many dimensions exist; default 3
        :param features_per_dimension: how many features exist in each dimension; default 10 each
        """
        super(ForgettingExperimentMetaLearningDataset, self).__init__(
            in_file, transform, start_index, end_index, None, return_indices,
            num_dimensions, features_per_dimension)

        # In the case it's a single-dimension example, validate parameters
        if single_dimension:
            if benchmark_dimension >= num_dimensions:
                raise ValueError(f'Benchmark dimension ({benchmark_dimension}) must be smaller than the number of dimensions ({num_dimensions})')

            if len(query_order) != features_per_dimension[benchmark_dimension]:
                raise ValueError(
                    f'The length of the query order {query_order} => {len(query_order)} must be equal to the number of features in that dimension ({features_per_dimension[benchmark_dimension]})')

            dimension_sizes = list(np.cumsum(features_per_dimension))
            dimension_sizes.insert(0, 0)

            if np.any(np.array(query_order) < dimension_sizes[benchmark_dimension]) or \
                    np.any(np.array(query_order) >= dimension_sizes[benchmark_dimension + 1]):
                raise ValueError(
                    f'The query order {query_order} for dimension {benchmark_dimension} must be between {dimension_sizes[benchmark_dimension]} and {dimension_sizes[benchmark_dimension + 1] - 1}')

        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.sub_epoch_size = sub_epoch_size
        self.num_sub_epochs = self.num_images // self.sub_epoch_size
        self.sub_epoch_index = -1
        self.sub_epochs = []
        self.query_order = query_order
        self.current_query_index = 1  # we start from one, since we do not actually train on the 1st query

        self.current_epoch_queries = []
        self.start_epoch()

    def __len__(self):
        return self.sub_epoch_size

    def next_query(self):
        """
        This does not actually do much, other than increment the current_query_index. The reason is that start_epoch
        reads that variable and will use this new value
        """
        self.current_query_index += 1
        self.sub_epoch_index = -1

    def assign_images_to_sub_epochs(self):
        perm = np.random.permutation(self.num_images)
        sub_epochs_without_task = [perm[i * self.sub_epoch_size:(i + 1) * self.sub_epoch_size]
                                   for i in range(self.num_sub_epochs)]

        self.sub_epochs = [list(zip(sub_epoch,
                                    itertools.cycle([self.query_order[self.current_query_index]])))
                           for sub_epoch in sub_epochs_without_task]

    def _compute_indices(self, index):
        return self.sub_epochs[self.sub_epoch_index][index]

    def start_epoch(self):
        """
        Start a new "small" epoch - either resample the dataset into sub-epochs
        or simply increment to the next one
        """
        self.sub_epoch_index += 1

        if self.sub_epoch_index % self.num_sub_epochs == 0:
            self.assign_images_to_sub_epochs()
            self.sub_epoch_index = 0


def create_normalized_datasets(dataset_path=META_LEARNING_DATA, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                               dataset_train_prop=DEFAULT_TRAIN_PROPORTION,
                               pin_memory=True, downsample_size=DOWNSAMPLE_SIZE,
                               should_flip=True, shuffle=True, return_indices=False,
                               dataset_class=MetaLearningH5DatasetFromDescription,
                               dataset_class_kwargs=None, train_dataset_kwargs=None, test_dataset_kwargs=None,
                               normalization_dataset_class=MetaLearningH5DatasetFromDescription,
                               train_dataset_class=None, test_dataset_class=None,
                               train_shuffle=None, test_shuffle=None,
                               train_batch_size=None, test_batch_size=None):
    """
    Helper function to create both the train and test normalized datasets.
    :param dataset_path: Which HDF5 file to load the dataset from
    :param batch_size: What batch size to use
    :param num_workers: How many workers to use in the PyTorch dataloaders
    :param dataset_train_prop: What proportion of the dataset to assign to train; the remainder goes to test
    :param pin_memory: Whether or not to pin GPU memory; PyTorch optimizations
    :param downsample_size: If downsampling, how much to downsample by
    :param should_flip: Should the training set dataloader introduce flipping data augmentation
    :param shuffle: Should the dataloaders shuffle the data
    :param return_indices: Whether or not to return indices with the examples
    :param dataset_class: Which of the dataset classes to use
    :param dataset_class_kwargs: Keyword arguments to pass to both train and test dataset
    :param train_dataset_kwargs: Keyword arguments to pass only to the training dataset
    :param test_dataset_kwargs: Keyword argmetns to pass only to the test dataset
    :param normalization_dataset_class: If we need to load the dataset to normalize, if it's not cached - which class
        to use.
    :return: The datasets and dataloaders for both train and test.
    """
    if train_dataset_class is None:
        train_dataset_class = dataset_class

    if test_dataset_class is None:
        test_dataset_class = dataset_class

    full_dataset = normalization_dataset_class(dataset_path, return_indices=return_indices)
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

        # To get it to load the file
        _ = unnormalized_train_dataset[0]

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

    normalized_train_dataset = train_dataset_class(dataset_path, transform=train_transformer,
                                             end_index=test_train_split_index,
                                             return_indices=return_indices, **train_dataset_kwargs)
    if train_shuffle is None:
        train_shuffle = shuffle
    if train_batch_size is None:
        train_batch_size = batch_size
    train_dataloader = DataLoader(normalized_train_dataset, batch_size=train_batch_size,
                                  shuffle=train_shuffle, num_workers=num_workers, pin_memory=pin_memory)

    normalized_test_dataset = test_dataset_class(dataset_path, transform=test_transformer,  # augment only in train
                                            start_index=test_train_split_index,
                                            return_indices=return_indices, **test_dataset_kwargs)
    if test_shuffle is None:
        test_shuffle = shuffle
    if test_batch_size is None:
        test_batch_size = batch_size
    test_dataloader = DataLoader(normalized_test_dataset, batch_size=test_batch_size,
                                 shuffle=test_shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return normalized_train_dataset, train_dataloader, normalized_test_dataset, test_dataloader



