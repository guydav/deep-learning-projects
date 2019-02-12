from .base_model import *


def sequential_benchmark(model, train_dataloader, test_dataloader, accuracy_threshold, num_epochs=100,
                         epochs_to_graph=None, cuda=True, save=True, start_epoch=0, watch=True):
    if epochs_to_graph is None:
        epochs_to_graph = 10

    if cuda:
        device = next(model.parameters()).device

    if watch:
        wandb.watch(model)

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

        log_results = {
            'Train Accuracy': np.mean(train_results['accuracies']),
            'Train Loss': np.mean(train_results['losses']),
            'Train AUC': np.mean(train_results['aucs']),
            # 'Train Per-Query Accuracy (dict)': {query: np.mean(values) for query, values in
            #                                     train_results['per_query_results'].items()},
            # 'Train Per-Query Accuracy (list)': np.array([np.mean(train_results['per_query_results'][query])
            #                                     for query in query_order[:current_query_index + 1]]),
            'Test Accuracy': np.mean(test_results['accuracies']),
            'Test Loss': np.mean(test_results['losses']),
            'Test AUC': np.mean(test_results['aucs']),
            # 'Test Per-Query Accuracy (dict)': {query: np.mean(values) for query, values in
            #                                    test_results['per_query_results'].items()},
            # 'Test Per-Query Accuracy (list)': np.array([np.mean(test_results['per_query_results'][query])
            #                                    for query in query_order[:current_query_index + 1]]),
        }

        log_results.update({f'Train accuracy, query = {query}': np.mean(values)
                            for query, values in train_results['per_query_results'].items()})

        log_results.update({f'Test accuracy, query = {query}': np.mean(values)
                            for query, values in test_results['per_query_results'].items()})

        for k, v in log_results.items():
            print(f'{k}: {v}')

        wandb.log(log_results, step=epoch)

        current_query = query_order[current_query_index]
        if log_results[f'Train accuracy, query = {current_query}'] > accuracy_threshold and \
            log_results[f'Test accuracy, query = {current_query}'] > accuracy_threshold:

            print(f'On epoch #{epoch}, reached criterion on query #{current_query_index} ({current_query}), moving to the next query')
            train_dataloader.dataset.next_query()
            test_dataloader.dataset.next_query()

        if epoch % epochs_to_graph == 0:
            mid_train_plot(model, 1)
