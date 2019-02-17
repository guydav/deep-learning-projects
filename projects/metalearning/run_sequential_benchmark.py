import sys
import torch

torch.multiprocessing.set_start_method("spawn")
sys.path.extend(('/home/cc/deep-learning-projects', '/home/cc/src/tqdm'))

import projects
from projects.metalearning import *
import argparse


parser = argparse.ArgumentParser()

ML_50K = '/home/cc/meta_learning_50k.h5'
parser.add_argument('--path_dataset', default=ML_50K)
DEFAULT_BATCH_SIZE = 1500
parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
DEFAULT_NUM_WORKERS = 4
parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
DEFAULT_PIN_MEMORY = 1
parser.add_argument('--pin_memory', type=int, default=DEFAULT_PIN_MEMORY)

parser.add_argument('--script_random_seed', type=int, default=None)
parser.add_argument('--benchmark_dimension', type=int, default=None)
parser.add_argument('--dataset_random_seed', type=int, default=None)
parser.add_argument('--train_coreset_size', type=int)
DEFAULT_TEST_CORESET_SIZE = 5000
parser.add_argument('--test_coreset_size', type=int, default=DEFAULT_TEST_CORESET_SIZE)
parser.add_argument('--coreset_per_query', type=int, default=0)
parser.add_argument('--query_order', default=None)
DEFAULT_ACCURACY_THRESHOLD = 0.95
parser.add_argument('--accuracy_threshold', type=float, default=DEFAULT_ACCURACY_THRESHOLD)

parser.add_argument('--description', default='')
DEFAULT_SAVE_DIR = '/home/cc/checkpoints'
parser.add_argument('--save_dir', default=DEFAULT_SAVE_DIR)
DEFAULT_MAX_EPOCHS = 200
parser.add_argument('--max_epochs', default=DEFAULT_MAX_EPOCHS)
parser.add_argument('--threshold_all_queries', type=int, default=1)


if __name__ == '__main__':
    args = parser.parse_args()
    
    dataset_path = args.path_dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = bool(args.pin_memory)

    if args.script_random_seed is not None:
        np.random.seed(args.script_random_seed)

    benchmark_dimension = args.benchmark_dimension
    if benchmark_dimension is None:
        benchmark_dimension = np.random.randint(3)
    dataset_random_seed = args.dataset_random_seed
    if dataset_random_seed is None:
        dataset_random_seed = np.random.randint(2 ** 32)

    train_coreset_size = args.train_coreset_size
    test_coreset_size = args.test_coreset_size
    train_coreset_size_per_query = bool(args.coreset_per_query)

    if args.query_order is not None:
        query_order = np.array([int(x) for x in args.query_order.split(' ')])
    else:
        query_order = np.arange(10) + benchmark_dimension * 10
        np.random.shuffle(query_order)

    accuracy_threshold = args.accuracy_threshold

    save_dir = args.save_dir
    current_epoch = 0
    total_epochs = args.max_epochs
    threshold_all_queries = bool(args.threshold_all_queries)

    normalized_train_dataset, train_dataloader, normalized_test_dataset, test_dataloader = \
        create_normalized_datasets(dataset_path=dataset_path,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   downsample_size=None,
                                   should_flip=False,
                                   return_indices=False,
                                   dataset_class=SequentialBenchmarkMetaLearningDataset,
                                   dataset_class_kwargs=dict(
                                       benchmark_dimension=benchmark_dimension,
                                       random_seed=dataset_random_seed,
                                       query_order=query_order
                                   ),
                                   train_dataset_kwargs=dict(
                                       previous_query_coreset_size=train_coreset_size,
                                       coreset_size_per_query=train_coreset_size_per_query,
                                   ),
                                   test_dataset_kwargs=dict(previous_query_coreset_size=test_coreset_size))

    sequential_benchmark_test_model = PoolingDropoutCNNMLP(
        query_length=30,
        conv_filter_sizes=(16, 32, 48, 64),
        conv_output_size=4480,
        mlp_layer_sizes=(512, 512, 512, 512),
        lr=5e-4,
        weight_decay=0,  # 1e-4,
        lr_scheduler_patience=100,
        conv_dropout=False,
        mlp_dropout=False,
        name='no_dropout_no_decay_sequential_benchmark_v1_coreset_15k_second_dimension',
        save_dir=save_dir)

    sequential_benchmark_test_model.load_model(current_epoch)
    sequential_benchmark_test_model = sequential_benchmark_test_model.cuda()

    # os.environ['WANDB_RUN_ID'] ='98w3kzlw'
    # os.environ['WANDB_RESUME'] = 'must'
    wandb.init(entity='meta-learning-scaling', project='sequential-benchmark')

    description = args.description
    if len(description) > 0:
        description += '\n'

    description += f'coreset size: {train_coreset_size}, benchmark dimension: {benchmark_dimension}, dataset random seed: {dataset_random_seed}, query order: {list(query_order)}, threshold all queries: {threshold_all_queries}'
    wandb.run.description = description
    wandb.run.save()

    current_model = sequential_benchmark_test_model

    wandb.config.lr = current_model.lr
    wandb.config.decay = current_model.weight_decay
    wandb.config.loss = 'CE'
    wandb.config.batch_size = train_dataloader.batch_size
    wandb.config.benchmark_dimension = benchmark_dimension
    wandb.config.dataset_random_seed = dataset_random_seed
    wandb.config.train_coreset_size = train_coreset_size
    wandb.config.test_coreset_size = test_coreset_size
    wandb.config.query_order = [int(x) for x in query_order]
    wandb.config.accuracy_threshold = accuracy_threshold
    wandb.config.epochs=total_epochs

    sequential_benchmark(sequential_benchmark_test_model, train_dataloader, test_dataloader, accuracy_threshold,
                         threshold_all_queries=threshold_all_queries,
                         num_epochs=total_epochs - current_epoch, epochs_to_graph=10, start_epoch=current_epoch)
