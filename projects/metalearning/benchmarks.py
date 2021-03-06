from .base_model import *


DEFAULT_ACCURACY_THRESHOLD = 0.95


def simultaneous_training(model, train_dataloader, test_dataloader, accuracy_threshold=DEFAULT_ACCURACY_THRESHOLD,
                          num_epochs=1000, cuda=True, save=True, start_epoch=0,
                          watch=True, debug=False, save_name='model',
                          train_epoch_func=train_epoch, test_epoch_func=test):
    """
    Execute the sequential benchmark as described in the paper.
    :param model: Which model to train and test according to the benchmark
    :param train_dataloader: The dataloader to use for the training set; should be using a dataset form the appropriate
        (SequentialBenchmarkMetaLearningDataset) class
    :param test_dataloader: The dataloader to use for the test set; should be using a dataset form the appropriate
        (SequentialBenchmarkMetaLearningDataset) class
    :param accuracy_threshold: What accuracy criterion to use; defaults to 95%
    :param num_epochs: How many epochs to stop after; previously defaulted to 100, currently defaults to 1000
    :param cuda: Whether or not to use CUDA (GPU acceleration)
    :param save: Whether or not to save the model
    :param start_epoch: Which epoch we're starting from; useful for restarting failed runs from the middle
    :param watch: Whether or not to watch the model
    :param debug: Whether or not to print debug prints
    :param save_name: A save name for saving results to weights & biases
    :param train_epoch_func: Allowing support to using a second train_epoch function, for MAML purposes
    :return:
    """
    device = None
    if cuda:
        device = next(model.parameters()).device

    if watch:
        wandb.watch(model)

    total_training_size = 0

    if hasattr(train_dataloader.dataset, 'query_subset'):
        print(f'Starting simultaneous training on the following tasks: {train_dataloader.dataset.query_subset}')
    else:
        print('Starting simultaneous training')

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        print(f'At epoch #{epoch}, len(train) = {len(train_dataloader.dataset)}, len(test) = {len(test_dataloader.dataset)}')

        train_results = train_epoch_func(model, train_dataloader, cuda, device, debug=debug)
        print_status(model, epoch, 'TRAIN', train_results)

        if save:
            model.save_model()

        test_results = test_epoch_func(model, test_dataloader, cuda, device, True)
        print_status(model, epoch, 'TEST', test_results)

        total_training_size += len(train_dataloader.dataset)

        log_results = create_log_results_dict(total_training_size, train_results, test_results,
                                              query_order=train_dataloader.dataset.query_subset, all_queries=True)

        wandb.log(log_results, step=epoch)

        criterion_met = np.all(np.array(log_results['Test Per-Query Accuracy (list)']) > accuracy_threshold)

        if criterion_met:
            print(f'On epoch #{epoch}, reached criterion on all queries {train_dataloader.dataset.query_subset}), done')
            # Added saving on every time we hit criterion
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{save_name}-final.pth'))
            return


def sequential_benchmark(model, train_dataloader, test_dataloader, accuracy_threshold=DEFAULT_ACCURACY_THRESHOLD,
                         threshold_all_queries=True,
                         num_epochs=1000, epochs_to_graph=None, cuda=True, save=True, start_epoch=0,
                         watch=True, debug=False, save_name='model',
                         train_epoch_func=train_epoch, test_epoch_func=test):
    """
    Execute the sequential benchmark as described in the paper.
    :param model: Which model to train and test according to the benchmark
    :param train_dataloader: The dataloader to use for the training set; should be using a dataset form the appropriate
        (SequentialBenchmarkMetaLearningDataset) class
    :param test_dataloader: The dataloader to use for the test set; should be using a dataset form the appropriate
        (SequentialBenchmarkMetaLearningDataset) class
    :param accuracy_threshold: What accuracy criterion to use; defaults to 95%
    :param threshold_all_queries: Should the benchmark proceed only when all queries pass the 95% accuracy criterion;
        defaults to true
    :param num_epochs: How many epochs to stop after; previously defaulted to 100, currently defaults to 1000
    :param epochs_to_graph: How many epochs to locally graph results using matplotlib; defualt None => never
    :param cuda: Whether or not to use CUDA (GPU acceleration)
    :param save: Whether or not to save the model
    :param start_epoch: Which epoch we're starting from; useful for restarting failed runs from the middle
    :param watch: Whether or not to watch the model
    :param debug: Whether or not to print debug prints
    :param save_name: A save name for saving results to weights & biases
    :param train_epoch_func: Allowing support to using a second train_epoch function, for MAML purposes
    :return:
    """

    if epochs_to_graph is None:
        epochs_to_graph = 10

    device = None
    if cuda:
        device = next(model.parameters()).device

    if watch:
        wandb.watch(model)

    total_training_size = 0
    query_order = train_dataloader.dataset.query_order
    print(f'Working in query order {query_order}, starting from query #0 ({query_order[0]})')

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_dataloader.dataset.start_epoch(debug)
        test_dataloader.dataset.start_epoch(debug)
        print(f'At epoch #{epoch}, len(train) = {len(train_dataloader.dataset)}, len(test) = {len(test_dataloader.dataset)}')

        train_results = train_epoch_func(model, train_dataloader, cuda, device, debug=debug)
        print_status(model, epoch, 'TRAIN', train_results)

        if save:
            model.save_model()

        test_results = test_epoch_func(model, test_dataloader, cuda, device, True)
        print_status(model, epoch, 'TEST', test_results)

        current_query_index = train_dataloader.dataset.current_query_index

        total_training_size += len(train_dataloader.dataset)

        log_results = create_log_results_dict(total_training_size, train_results, test_results, query_order,
                                              current_query_index)

        # for k, v in log_results.items():
        #     print(f'{k}: {v}')

        wandb.log(log_results, step=epoch)

        if threshold_all_queries:
            criterion_met = np.all(np.array(log_results['Test Per-Query Accuracy (list)']) > accuracy_threshold)

        else:
            criterion_met = log_results['Test Per-Query Accuracy (list)'][current_query_index] > accuracy_threshold

        if criterion_met:
            print(f'On epoch #{epoch}, reached criterion on query #{current_query_index} ({query_order[current_query_index]}), moving to the next query')
            train_dataloader.dataset.next_query()
            test_dataloader.dataset.next_query()

            # Added saving on every time we hit criterion
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{save_name}-query-{current_query_index}.pth'))

        if epoch % epochs_to_graph == 0:
            mid_train_plot(model, 1)

        if train_dataloader.dataset.current_query_index == len(query_order):
            return


def forgetting_experiment(model, checkpoint_file_pattern, train_dataloader, test_dataloader,
                          accuracy_threshold=DEFAULT_ACCURACY_THRESHOLD,
                          num_epochs=1000, epochs_to_graph=None, cuda=True, save=True, start_epoch=0,
                          watch=True, save_name='model', start_task=2, per_task_epoch_limit=1000):
    """
    Execute the sequential benchmark as described in the paper.
    :param model: Which model to train and test according to the benchmark
    :param train_dataloader: The dataloader to use for the training set; should be using a dataset form the appropriate
        (SequentialBenchmarkMetaLearningDataset) class
    :param test_dataloader: The dataloader to use for the test set; should be using a dataset form the appropriate
        (SequentialBenchmarkMetaLearningDataset) class
    :param accuracy_threshold: What accuracy criterion to use; defaults to 95%
    :param threshold_all_queries: Should the benchmark proceed only when all queries pass the 95% accuracy criterion;
        defaults to true
    :param num_epochs: How many epochs to stop after; previously defaulted to 100, currently defaults to 1000
    :param epochs_to_graph: How many epochs to locally graph results using matplotlib; defualt None => never
    :param cuda: Whether or not to use CUDA (GPU acceleration)
    :param save: Whether or not to save the model
    :param start_epoch: Which epoch we're starting from; useful for restarting failed runs from the middle
    :param watch: Whether or not to watch the model
    :param debug: Whether or not to print debug prints
    :param save_name: A save name for saving results to weights & biases
    :return:
    """

    if epochs_to_graph is None:
        epochs_to_graph = 10

    device = None
    if cuda:
        device = next(model.parameters()).device

    if watch:
        wandb.watch(model)

    total_training_size = 0
    query_order = train_dataloader.dataset.query_order
    print(f'Working in query order {query_order}, starting from query #1 ({query_order[1]})')

    for skipped_task in range(2, start_task):
        train_dataloader.dataset.next_query()
        test_dataloader.dataset.next_query()

    # Bug-fix -- forgot to do this previously
    test_dataloader.dataset.start_epoch()

    print('Loading checkpoint for end of previous query')
    # Subtracting 2: 1 for zero-based vs. one-based, one because we want to load at the end of the previous task
    model.load_state(checkpoint_file_pattern.format(query=start_task - 2))

    # Test the model to get a baseline for the forgetting curves
    test_results = test(model, test_dataloader, cuda, device, True)
    print_status(model, 0, f'PRE-TASK {start_task - 1}', test_results)
    log_results = create_log_results_dict(total_training_size, test_results, query_order,
                                          train_dataloader.dataset.current_query_index)
    wandb.log(log_results)

    test_dataloader.dataset.next_query()  # we need to make sure the next query us active from the first moment

    current_task_start_epoch = start_epoch

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_dataloader.dataset.start_epoch()
        test_dataloader.dataset.start_epoch()
        print(f'At epoch #{epoch}, len(train) = {len(train_dataloader.dataset)}, len(test) = {len(test_dataloader.dataset)}')

        train_results = train_epoch(model, train_dataloader, cuda, device)
        print_status(model, epoch, 'TRAIN', train_results)

        if save:
            model.save_model()

        test_results = test(model, test_dataloader, cuda, device, True)
        print_status(model, epoch, 'TEST', test_results)

        current_query_index = train_dataloader.dataset.current_query_index

        total_training_size += len(train_dataloader.dataset)

        log_results = create_log_results_dict(total_training_size, train_results, test_results, query_order,
                                              current_query_index)

        wandb.log(log_results)

        criterion_met = log_results['Test Per-Query Accuracy (list)'][current_query_index] > accuracy_threshold
        out_of_epochs = (epoch - current_task_start_epoch) > per_task_epoch_limit

        print(epoch, current_query_index, log_results['Test Per-Query Accuracy (list)'], criterion_met, out_of_epochs)

        if criterion_met or out_of_epochs:
            if criterion_met:
                print(f'On epoch #{epoch}, reached criterion on query #{current_query_index} ({query_order[current_query_index]}), moving to the next query')

            if out_of_epochs:
                print(
                    f'On epoch #{epoch}, reached {per_task_epoch_limit} from the previous task start ({current_task_start_epoch}), moving to the next query')
            train_dataloader.dataset.next_query()
            test_dataloader.dataset.next_query()

            # Added saving on every time we hit criterion
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{save_name}-query-{current_query_index}.pth'))

            # Load the model from the previous checkpoint after learning the new query
            print('Loading checkpoint for end of previous query')
            model.load_state(checkpoint_file_pattern.format(query=current_query_index))

            # Test the model to get a baseline for the forgetting curves
            test_results = test(model, test_dataloader, cuda, device, True)
            print_status(model, epoch, f'PRE-TASK {current_query_index + 1}', test_results)
            log_results = create_log_results_dict(total_training_size, test_results, query_order, current_query_index)
            wandb.log(log_results)

            current_task_start_epoch = epoch

        if epoch % epochs_to_graph == 0:
            mid_train_plot(model, 1)

        if train_dataloader.dataset.current_query_index == len(query_order):
            return


def create_log_results_dict(total_training_size, train_results=None, test_results=None, query_order=None,
                            current_query_index=0, all_queries=False):
    log_results = {
        'Total Train Size': total_training_size,
    }

    if train_results is not None:
        train_log = epoch_results_to_log_dict('Train', train_results, query_order, current_query_index, all_queries)
        log_results.update(train_log)

    if test_results is not None:
        test_log = epoch_results_to_log_dict('Test', test_results, query_order, current_query_index, all_queries)
        log_results.update(test_log)

    return log_results


def epoch_results_to_log_dict(name, epoch_results, query_order=None, current_query_index=0, all_queries=False):
    name = name.capitalize()
    log_dict = {
        f'{name} Accuracy': np.mean(epoch_results['accuracies']),
        f'{name} Loss': np.mean(epoch_results['losses']),
        f'{name} AUC': np.mean(epoch_results['aucs'])
    }

    if query_order is not None:
        if not all_queries:
            log_dict.update({
                f'{name} Per-Query Accuracy (dict)': {str(index): np.mean(epoch_results['per_query_results'][query])
                                                      for index, query in enumerate(query_order[:current_query_index + 1])},
                f'{name} Per-Query Accuracy (list)': np.array([np.mean(epoch_results['per_query_results'][query])
                                                               for query in query_order[:current_query_index + 1]]),
            })

            log_dict.update({f'{name} Accuracy, Query #{index + 1}': np.mean(epoch_results['per_query_results'][query])
                             for index, query in enumerate(query_order[:current_query_index + 1])})

        else:
            log_dict.update({
                f'{name} Per-Query Accuracy (dict)': {str(query): np.mean(epoch_results['per_query_results'][query])
                                                      for query in query_order},
                f'{name} Per-Query Accuracy (list)': np.array([np.mean(epoch_results['per_query_results'][query])
                                                               for query in query_order]),
            })

    if query_order is not None and current_query_index > 0:
        log_dict[f'{name} Mean Previous-Query Accuracy'] = np.mean(
            [np.mean(epoch_results['per_query_results'][query])
             for query in query_order[:current_query_index]])

    return log_dict


def forgetting_experiment_resume_pre_test_fix(model, checkpoint_file_pattern, test_dataloader,
                                              task_to_pre_test=2, cuda=True):
    device = None
    if cuda:
        device = next(model.parameters()).device

    query_order = test_dataloader.dataset.query_order
    print(f'Working in query order {query_order}, starting from query #1 ({query_order[1]})')

    for skipped_task in range(2, task_to_pre_test):
        test_dataloader.dataset.next_query()

    test_dataloader.start_epoch()

    print('Loading checkpoint for end of previous query')
    # Subtracting 2: 1 for zero-based vs. one-based, one because we want to load at the end of the previous task
    model.load_state(checkpoint_file_pattern.format(query=task_to_pre_test - 2))

    # Test the model to get a baseline for the forgetting curves
    test_results = test(model, test_dataloader, cuda, device, True)
    return test_results['Test Per-Query Accuracy (list)']


