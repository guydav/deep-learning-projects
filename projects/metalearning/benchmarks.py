from .base_model import *


def sequential_benchmark(model, train_dataloader, test_dataloader, accuracy_threshold, threshold_all_queries=False,
                         num_epochs=100, epochs_to_graph=None, cuda=True, save=True, start_epoch=0, watch=True):
    if epochs_to_graph is None:
        epochs_to_graph = 10

    if cuda:
        device = next(model.parameters()).device

    if watch:
        wandb.watch(model)

    total_training_size = 0

    for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
        train_dataloader.dataset.start_epoch()
        test_dataloader.dataset.start_epoch()

        train_results = train_epoch(model, train_dataloader, cuda, device)
        print_status(model, epoch, 'TRAIN', train_results)

        if save:
            model.save_model()

        if save:
            model.save_model()

        test_results = test(model, test_dataloader, cuda, device, True)
        print_status(model, epoch, 'TEST', test_results)

        query_order = train_dataloader.dataset.query_order
        current_query_index = train_dataloader.dataset.current_query_index

        total_training_size += len(train_dataloader.dataset)

        log_results = {
            'Total Train Size': total_training_size,
            'Train Accuracy': np.mean(train_results['accuracies']),
            'Train Loss': np.mean(train_results['losses']),
            'Train AUC': np.mean(train_results['aucs']),
            'Train Per-Query Accuracy (dict)': {index: np.mean(train_results['per_query_results'][query])
                                                for index, query in enumerate(query_order[:current_query_index + 1])},
            'Train Per-Query Accuracy (list)': np.array([np.mean(train_results['per_query_results'][query])
                                                         for query in query_order[:current_query_index + 1]]),
            'Test Accuracy': np.mean(test_results['accuracies']),
            'Test Loss': np.mean(test_results['losses']),
            'Test AUC': np.mean(test_results['aucs']),
            'Train Per-Query Accuracy (dict)': {index: np.mean(test_results['per_query_results'][query])
                                                for index, query in enumerate(query_order[:current_query_index + 1])},
            'Test Per-Query Accuracy (list)': np.array([np.mean(test_results['per_query_results'][query])
                                                        for query in query_order[:current_query_index + 1]]),
        }

        log_results.update({f'Train Accuracy, Query #{index + 1}': np.mean(train_results['per_query_results'][query])
                            for index, query in enumerate(query_order[:current_query_index + 1])})

        log_results.update({f'Test Accuracy, Query #{index + 1}': np.mean(test_results['per_query_results'][query])
                            for index, query in enumerate(query_order[:current_query_index + 1])})

        if current_query_index > 0:
            log_results['Train Mean Previous-Query Accuracy'] = np.mean(
                [np.mean(train_results['per_query_results'][query])
                 for query in query_order[:current_query_index]])

            log_results['Test Mean Previous-Query Accuracy'] = np.mean(
                [np.mean(test_results['per_query_results'][query])
                 for query in query_order[:current_query_index]])


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

        if epoch % epochs_to_graph == 0:
            mid_train_plot(model, 1)

        if train_dataloader.dataset.current_query_index == len(query_order):
            return
