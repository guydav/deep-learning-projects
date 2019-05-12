import sys
import torch
import wandb
import json

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

RUN_ID_FILE = './forgetting_experiment_run_id.txt'
parser.add_argument('--run_id_file_path', type=str, default=RUN_ID_FILE)
parser.add_argument('--run_id_line_number', type=int)

RUN_PATTERN = 'meta-learning-scaling/sequential-benchmark-baseline/{run_id}'
parser.add_argument('--run_path_pattern', type=str, default=RUN_PATTERN)

CHECKPOINT_FILE_PATTERN = 'Baseline-{seed}-query-{{query}}.pth'
parser.add_argument('--checkpoint_file_pattern', type=str, default=CHECKPOINT_FILE_PATTERN)

DEFAULT_TRAIN_SUB_EPOCH_SIZE = 4500
parser.add_argument('--train_sub_epoch_size', type=int, default=DEFAULT_TRAIN_SUB_EPOCH_SIZE)

##### verified up to there


DEFAULT_TEST_CORESET_SIZE = 5000
parser.add_argument('--test_coreset_size', type=int, default=DEFAULT_TEST_CORESET_SIZE)
parser.add_argument('--coreset_size_per_query', type=int, default=0)

parser.add_argument('--accuracy_threshold', type=float, default=DEFAULT_ACCURACY_THRESHOLD)

DEFAULT_LEARNING_RATE = 5e-4
parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE)
DEFAULT_WEIGHT_DECAY = 1e-4
parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)

parser.add_argument('--name')
parser.add_argument('--description', default='')
DEFAULT_SAVE_DIR = '/home/cc/checkpoints'
parser.add_argument('--save_dir', default=DEFAULT_SAVE_DIR)
DEFAULT_MAX_EPOCHS = 1000
parser.add_argument('--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS)
parser.add_argument('--threshold_all_queries', type=int, default=1)

DEFAULT_WANDB_PROJECT = 'sequential-benchmark'
parser.add_argument('--wandb_project', default=DEFAULT_WANDB_PROJECT)

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

    with open(args.run_id_file_path, 'r') as run_id_file:
        run_ids = run_id_file.readlines()
        run_id = run_ids[args.run_id_line_number]

    print(f'For line {args.run_id_line_number} the id is {run_id}')

    wandb_api = wandb.Api()
    wandb_run = wandb_api.run(args.run_path_pattern.format(run_id=run_id))
    run_config = json.loads(wandb_run.json_config)

    dataset_random_seed = run_config['dataset_random_seed']['value']
    benchmark_dimension = run_config['benchmark_dimension']['value']['value']
    query_order = run_config['query_order']['value']

    files = wandb_run.files()
    for f in files:
        if f.name.endswith('.pth'):
            f.download()

    checkpoint_file_pattern = args.checkpoint_file_pattern.format(seed=dataset_random_seed)

    np.random.seed(dataset_random_seed)
    torch.manual_seed(dataset_random_seed)
    torch.cuda.manual_seed_all(dataset_random_seed)

    train_sub_epoch_size = args.train_sub_epoch_size
    test_coreset_size = args.test_coreset_size
    train_coreset_size_per_query = bool(args.coreset_size_per_query)

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
                                   train_dataset_class=ForgettingExperimentMetaLearningDataset,
                                   train_dataset_kwargs=dict(
                                       sub_epoch_size=train_sub_epoch_size,
                                   ),
                                   test_dataset_kwargs=dict(
                                       previous_query_coreset_size=test_coreset_size,
                                       coreset_size_per_query=True,
                                   ))

    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    sequential_benchmark_test_model = PoolingDropoutCNNMLP(
        query_length=30,
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

    sequential_benchmark_test_model.load_model(current_epoch)
    sequential_benchmark_test_model = sequential_benchmark_test_model.cuda()

    wandb.init(entity='meta-learning-scaling', project=args.wandb_project)

    description = args.description
    if len(description) > 0:
        description += '\n'

    description += f'{args.name}-{dataset_random_seed}\ncoreset size: {train_coreset_size}, benchmark dimension: {benchmark_dimension}, dataset random seed: {dataset_random_seed}, query order: {list(query_order)}, threshold all queries: {threshold_all_queries}'
    wandb.run.description = description
    wandb.run.save()

    current_model = sequential_benchmark_test_model

    wandb.config.lr = current_model.lr
    wandb.config.decay = current_model.weight_decay
    wandb.config.loss = 'CE'
    wandb.config.batch_size = train_dataloader.batch_size
    wandb.config.benchmark_dimension = benchmark_dimension
    wandb.config.dataset_random_seed = dataset_random_seed
    wandb.config.train_sub_epoch_size = train_sub_epoch_size
    wandb.config.test_coreset_size = test_coreset_size
    wandb.config.query_order = [int(x) for x in query_order]
    wandb.config.accuracy_threshold = accuracy_threshold
    wandb.config.epochs = total_epochs
    wandb.config.run_id_line_number = args.run_id_line_number
    wandb.config.original_run_id = run_id

    forgetting_experiment(sequential_benchmark_test_model, checkpoint_file_pattern,
                          train_dataloader, test_dataloader, accuracy_threshold,
                         threshold_all_queries=threshold_all_queries,
                         num_epochs=total_epochs - current_epoch,
                         epochs_to_graph=total_epochs + 1,
                         start_epoch=current_epoch,
                         debug=args.debug,
                         save_name=f'{args.name}-{dataset_random_seed}')
