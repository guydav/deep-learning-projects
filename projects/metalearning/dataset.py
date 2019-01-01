import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import h5py


META_LEARNING_DATA = 'drive/Research Projects/Meta-Learning/v1/CLEVR_meta_learning_uint8_desc.h5'
META_LEARNING_DATA_SMALL = 'drive/Research Projects/Meta-Learning/v1/CLEVR_meta_learning_small_uint8.h5'
BATCH_SIZE = 512  # 64
NUM_WORKERS = 1

DOWNSAMPLE_SIZE = (96, 128)
TEST_TRAIN_SPLIT_INDEX = 4096 * 7 // 8  # 512 * 7 // 8 #


class MetaLearningH5Dataset(Dataset):
    def __init__(self, in_file, transform=None, start_index=0,
                 end_index=None, query_subset=None):
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

        return (x, y, q), index

    def __len__(self):
        return self.num_images * self.active_queries_per_image


class MetaLearningH5DatasetFromDescription(MetaLearningH5Dataset):
    def __init__(self, in_file, transform=None, start_index=0,
                 end_index=None, query_subset=None,
                 num_dimensions=2, features_per_dimension=(10, 11)):
        super(MetaLearningH5DatasetFromDescription, self).__init__(
            in_file, transform, start_index, end_index, query_subset)

        self.num_dimensions = num_dimensions
        self.features_per_dimension = features_per_dimension

    def __getitem__(self, index):
        image_index, query_index = self._compute_indices(index)

        x = self.file['X'][image_index, ...]
        # TODO: for simple queries I can use y, but I will compute y, because I'll need to later
        # y = self.file['y'][image_index, actual_query_index]

        desc = self.file['D'][image_index]
        y = int(np.any(desc == query_index))
        q = np.zeros((self.num_dimensions + self.total_queries_per_image,))
        # TODO: generalize for 2-item queries and 3rd dimensions
        q[0] = query_index < self.features_per_dimension[0]
        q[1] = 1 - q[0]
        q[self.num_dimensions + query_index] = 1

        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)

        return (x, y, q), index

def create_normalized_datasets(dataset_path=META_LEARNING_DATA, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                               pin_memory=True, downsample_size=DOWNSAMPLE_SIZE,
                               test_train_split_index=TEST_TRAIN_SPLIT_INDEX, shuffle=True):

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(downsample_size)
    to_pil = transforms.ToPILImage()

    # TODO: why did I have to_tensor.float()? Add it back in later?
    unnormalized_transformer = transforms.Compose([
        to_pil,
        resize,
        to_tensor
    ])

    unnormalized_train_dataset = MetaLearningH5DatasetFromDescription(dataset_path,  # META_LEARNING_DATA_SMALL,
                                                                      unnormalized_transformer,
                                                                      end_index=test_train_split_index)

    transformed_images = np.stack([unnormalized_transformer(image).numpy() for image in
                                   unnormalized_train_dataset.file['X']])
    channel_means = np.mean(transformed_images, (0, 2, 3))
    channel_stds = np.std(transformed_images, (0, 2, 3))
    print(channel_means)
    print(channel_stds)

    normalizer = transforms.Normalize(torch.from_numpy(channel_means),
                                      torch.from_numpy(channel_stds))

    normalized_transformer = transforms.Compose([
        to_pil,
        resize,
        to_tensor,
        normalizer,
    ])

    normalized_augmenting_transformer = transforms.Compose([
        to_pil,
        resize,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        to_tensor,
        normalizer,
    ])

    normalized_train_dataset = MetaLearningH5DatasetFromDescription(dataset_path,
                                                                    normalized_augmenting_transformer,
                                                                    end_index=test_train_split_index)
    train_dataloader = DataLoader(normalized_train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    normalized_test_dataset = MetaLearningH5DatasetFromDescription(dataset_path,
                                                                   normalized_transformer,  # augment only in train
                                                                   start_index=test_train_split_index)
    test_dataloader = DataLoader(normalized_test_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return normalized_train_dataset, train_dataloader, normalized_test_dataset, test_dataloader
