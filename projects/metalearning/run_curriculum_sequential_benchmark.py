import sys
import torch

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

WILLIAMS_SQUARE_TEN = np.array(
    [[ 1,  2, 10,  3,  9,  4,  8,  5,  7,  6],
     [ 2,  3,  1,  4, 10,  5,  9,  6,  8,  7],
     [ 3,  4,  2,  5,  1,  6, 10,  7,  9,  8],
     [ 4,  5,  3,  6,  2,  7,  1,  8, 10,  9],
     [ 5,  6,  4,  7,  3,  8,  2,  9,  1, 10],
     [ 6,  7,  5,  8,  4,  9,  3, 10,  2,  1],
     [ 7,  8,  6,  9,  5, 10,  4,  1,  3,  2],
     [ 8,  9,  7, 10,  6,  1,  5,  2,  4,  3],
     [ 9, 10,  8,  1,  7,  2,  6,  3,  5,  4],
     [10,  1,  9,  2,  8,  3,  7,  4,  6,  5]])
parser.add_argument('--use_latin_square', action='store_true')
parser.add_argument('--latin_square_random_seed', type=int, default=None)
parser.add_argument('--latin_square_index', type=int, default=None)

parser.add_argument('--script_random_seed', type=int, default=None)
parser.add_argument('--benchmark_dimension', type=int, default=None)
parser.add_argument('--dataset_random_seed', type=int, default=None)

TOTAL_DATASET_SIZE = 50000

DEFAULT_TEST_CORESET_SIZE = 5000
parser.add_argument('--test_coreset_size', type=int, default=DEFAULT_TEST_CORESET_SIZE)

parser.add_argument('--query_order', default=None)
DEFAULT_ACCURACY_THRESHOLD = 0.95
parser.add_argument('--accuracy_threshold', type=float, default=DEFAULT_ACCURACY_THRESHOLD)

DEFAULT_LEARNING_RATE = 5e-4
parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
DEFAULT_WEIGHT_DECAY = 1e-4
parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)

parser.add_argument('--name')
parser.add_argument('--description', default='')
DEFAULT_SAVE_DIR = '/home/cc/checkpoints'
parser.add_argument('--save_dir', default=DEFAULT_SAVE_DIR)
DEFAULT_MAX_EPOCHS = 4000
parser.add_argument('--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS)
parser.add_argument('--threshold_all_queries', type=int, default=1)

DEFAULT_WANDB_PROJECT = 'sequential-benchmark'
parser.add_argument('--wandb_project', default=DEFAULT_WANDB_PROJECT)

DEFAULT_LATEST_TASK_NUM_EXAMPLES = 22500
parser.add_argument('--latest_task_num_examples', type=int, default=DEFAULT_LATEST_TASK_NUM_EXAMPLES)
DEFAULT_PREVIOUS_TASKS_EXAMPLE_RATIO = 1.25
parser.add_argument('--previous_tasks_example_ratio', type=float, default=DEFAULT_PREVIOUS_TASKS_EXAMPLE_RATIO)

# parser.add_argument('--maml', action='store_true')
# DEFAULT_FAST_WEIGHT_LEARNING_RATE = 5e-4
# parser.add_argument('--fast_weight_learning_rate', type=float, default=DEFAULT_FAST_WEIGHT_LEARNING_RATE)
# parser.add_argument('--return_indices', action='store_true')
#
parser.add_argument('--balanced_batches', action='store_true')
# parser.add_argument('--maml_meta_test', action='store_true')

parser.add_argument('--debug', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    dataset_path = args.path_dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = bool(args.pin_memory)

    if num_workers > 1:
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    if args.script_random_seed is not None:
        np.random.seed(args.script_random_seed)

    benchmark_dimension = args.benchmark_dimension
    if benchmark_dimension is None:
        benchmark_dimension = np.random.randint(3)

    dataset_random_seed = args.dataset_random_seed
    if dataset_random_seed is None:
        dataset_random_seed = np.random.randint(2 ** 32)

    torch.manual_seed(dataset_random_seed)
    torch.cuda.manual_seed_all(dataset_random_seed)

    test_coreset_size = args.test_coreset_size
    # train_coreset_size = args.train_coreset_size
    # train_coreset_size_per_query = bool(args.coreset_size_per_query)

    # if args.maml_meta_test and not args.maml:
    #     print('maml_meta_test can only be set to true if maml is. Aborting...')
    #     sys.exit(1)

    if args.query_order is not None:
        query_order = np.array([int(x) for x in args.query_order.split(' ')])

    else:
        if args.use_latin_square:
            latin_square_random = np.random.RandomState(args.latin_square_random_seed)
            latin_square = np.copy(WILLIAMS_SQUARE_TEN)
            latin_square_random.shuffle(latin_square)  # permute rows
            latin_square_random.shuffle(latin_square.T)  # permute cols
            query_order = latin_square[args.latin_square_index % latin_square.shape[0]]
            query_order -= 1

        else:
            query_order = np.arange(10)
            np.random.shuffle(query_order)

        query_order += benchmark_dimension * 10

    accuracy_threshold = args.accuracy_threshold

    save_dir = args.save_dir
    current_epoch = 0
    total_epochs = args.max_epochs
    threshold_all_queries = bool(args.threshold_all_queries)

    def curriculum_function(episode_number, task_number):
        if task_number > episode_number:
            raise ValueError(
                f'Cannot compute examples for a higher task number ({task_number}) than the episode number ({episode_number})')

        if episode_number == task_number:
            return args.latest_task_num_examples

        total_previous_task_examples = TOTAL_DATASET_SIZE - test_coreset_size - args.latest_task_num_examples
        last_previous_task_number = episode_number - 1
        first_task_examples = total_previous_task_examples / np.sum(np.power(args.previous_tasks_example_ratio, range(last_previous_task_number)))

        return first_task_examples * (args.previous_tasks_example_ratio ** (task_number - 1))


    train_dataset_class = CustomCurriculumSequentialBenchmarkMetaLearningDataset
    train_shuffle = True
    train_dataset_kwargs = dict(
        curriculum_function=curriculum_function,
    )

    test_dataset_kwargs = dict(
        previous_query_coreset_size=test_coreset_size,
        coreset_size_per_query=True,
    )

    if args.balanced_batches:
        print('Using balanced batches')
        train_dataset_class = BalancedBatchesCustomCurriculumSequentialBenchmarkMetaLearningDataset
        train_shuffle = False

        if 'batch_size' not in train_dataset_kwargs:
            train_dataset_kwargs['batch_size'] = batch_size

    normalized_train_dataset, train_dataloader, normalized_test_dataset, test_dataloader = \
        create_normalized_datasets(dataset_path=dataset_path,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   downsample_size=None,
                                   should_flip=False,
                                   return_indices=args.return_indices,
                                   dataset_class=SequentialBenchmarkMetaLearningDataset,
                                   dataset_class_kwargs=dict(
                                       benchmark_dimension=benchmark_dimension,
                                       random_seed=dataset_random_seed,
                                       query_order=query_order
                                   ),
                                   train_dataset_class=train_dataset_class,
                                   train_dataset_kwargs=train_dataset_kwargs,
                                   test_dataset_class=SequentialBenchmarkMetaLearningDataset,
                                   test_dataset_kwargs=test_dataset_kwargs,
                                   train_shuffle=train_shuffle,
                                   test_shuffle=True,
                                   train_batch_size=None,
                                   test_batch_size=None)

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    model_kwargs = dict(query_length=30,
                        conv_filter_sizes=(16, 32, 48, 64),
                        conv_output_size=4480,
                        mlp_layer_sizes=(512, 512, 512, 512),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        use_lr_scheduler=False,
                        conv_dropout=False,
                        mlp_dropout=False,
                        name=f'{args.name}-{dataset_random_seed}',
                        save_dir=save_dir)

    # if args.maml:
    #     model_kwargs['fast_weight_lr'] = args.fast_weight_learning_rate
    #     model = MamlPoolingDropoutCNNMLP(**model_kwargs)
    #
    # else:
    model = PoolingDropoutCNNMLP(**model_kwargs)

    model.load_model(current_epoch)
    model = model.cuda()

    if args.debug: print('After model.cuda()')

    if args.debug: print(f'wandb project is {args.wandb_project}')

    description = args.description
    if len(description) > 0:
        description += '\n'

    description += f'coreset size: {train_coreset_size}, benchmark dimension: {benchmark_dimension}, dataset random seed: {dataset_random_seed}, query order: {list(query_order)}, threshold all queries: {threshold_all_queries}'

    # os.environ['WANDB_RUN_ID'] ='98w3kzlw'
    # os.environ['WANDB_RESUME'] = 'must'
    wandb.init(entity='meta-learning-scaling', project=args.wandb_project,
               name=f'{args.name}-{dataset_random_seed}', notes=description)

    wandb.run.save()

    current_model = model

    wandb.config.lr = current_model.lr
    wandb.config.decay = current_model.weight_decay
    wandb.config.loss = 'CE'
    wandb.config.batch_size = train_dataloader.batch_size
    wandb.config.benchmark_dimension = benchmark_dimension
    wandb.config.dataset_random_seed = dataset_random_seed
    # wandb.config.train_coreset_size = train_coreset_size
    wandb.config.test_coreset_size = test_coreset_size
    wandb.config.query_order = [int(x) for x in query_order]
    wandb.config.accuracy_threshold = accuracy_threshold
    wandb.config.epochs = total_epochs
    wandb.config.latest_task_num_examples = args.latest_task_num_examples
    wandb.config.previous_tasks_example_ratio = args.previous_tasks_example_ratio
    wandb.config.balanced_batches = args.balanced_batches

    # if args.maml:
    #     wandb.config.fast_weight_lr = args.fast_weight_learning_rate

    if args.use_latin_square:
        wandb.config.latin_square_random_seed = args.latin_square_random_seed
        wandb.config.latin_square_index = args.latin_square_index

    if args.debug: print('After wandb init')

    train_epoch_func = train_epoch
    test_epoch_func = test

    # if args.maml:
    #     train_epoch_func = maml_train_epoch
    #
    #     if args.maml_meta_test:
    #         test_epoch_func = maml_test_epoch

    if args.debug: print('Calling sequential benchmark')

    sequential_benchmark(model, train_dataloader, test_dataloader, accuracy_threshold,
                         threshold_all_queries=threshold_all_queries,
                         num_epochs=total_epochs - current_epoch,
                         epochs_to_graph=total_epochs + 1,
                         start_epoch=current_epoch,
                         debug=args.debug,
                         save_name=f'{args.name}-{dataset_random_seed}',
                         train_epoch_func=train_epoch_func, test_epoch_func=test_epoch_func)
