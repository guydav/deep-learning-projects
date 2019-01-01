import numpy as np
import matplotlib.pyplot as plt
import torch
import tabulate
import sklearn


def compare_results(ground_truth, first_results, second_results,
                    first_title='ABC', second_title='ABC_GP', top_k_values=None):
    ground_truth = np.array(ground_truth)
    max_size = min(len(first_results), len(second_results))
    max_power = int(np.floor(np.log10(max_size)))

    if type(second_results[0][1]) == torch.Tensor:
        second_results = [(tup[0], tup[1].numpy()) for tup in second_results]

    if top_k_values is None:
        top_k_values = [10 ** pow for pow in range(max_power + 1)]

    rows = []
    for top_k in top_k_values:
        first_score = np.mean([tup[0] for tup in first_results[:top_k]])
        first_mean_mad = np.mean([np.sum(np.abs(tup[1] - ground_truth)) for tup in first_results[:top_k]])

        second_score = np.mean([tup[0] for tup in second_results[:top_k]])
        second_mean_mad = np.mean([np.sum(np.abs(tup[1] - ground_truth)) for tup in second_results[:top_k]])

        rows.append((top_k, first_score, first_mean_mad, second_score, second_mean_mad,
                     first_score - second_score, first_mean_mad - second_mean_mad))

    print(tabulate.tabulate(rows, ('Top K', f'{first_title} score', f'{first_title} digit MAD',
                                   f'{second_title} score', f'{second_title} digit MAD',
                                   'Score diff', 'Mean MAD diff'), tablefmt='fancy_grid'))


def single_digit_result_histogram(result_sets, labels, digit=0, ground_truth=0.3, top_k=None,
                                  figsize=(13, 6), font_size=16, title=None):
    if top_k is None:
        top_k = len(result_sets[0])

    plt.figure(figsize=figsize)

    for result, label in zip(result_sets, labels):
        if type(result[0][1]) == torch.Tensor:
            result = [(tup[0], np.squeeze(tup[1].numpy())) for tup in result]

        relevant_results = [x[1][digit] for x in result][:top_k]
        plt.hist(relevant_results, alpha=0.66, edgecolor='black', label=label, density=True)

    plt.vlines(ground_truth, 0, plt.ylim()[1] * 0.8, label='Ground Truth')
    plt.legend(loc='best', fontsize=font_size)
    plt.xlabel(f'Probability assigned to digit: {digit}', fontsize=font_size)
    plt.ylabel(f'Density', fontsize=font_size)
    if title is not None:
        plt.title(title, fontsize=font_size)
    plt.show()


def two_digit_result_density(result_sets, labels, suptitle=None, ground_truth=(0.3, 0.7), top_k=None,
                             figsize=(15, 6), font_size=16, step=0.05, cmap='YlOrRd'):
    if top_k is None:
        top_k = len(result_sets[0])

    plt.figure(figsize=figsize)

    for i, (result, label) in enumerate(zip(result_sets, labels)):
        if type(result[0][1]) == torch.Tensor:
            result = [(tup[0], np.squeeze(tup[1].numpy())) for tup in result]

        relevant_results = np.array([x[1][:2] for x in result][:top_k])
        kde = sklearn.neighbors.KernelDensity().fit(relevant_results)

        x = np.arange(0, 1, step)
        y = np.arange(0, 1, step)
        xx, yy = np.meshgrid(x, y)

        points = np.array(list(zip(xx.flat, yy.flat)))
        z = kde.score_samples(points).reshape((x.shape[0], y.shape[0]))

        ax = plt.subplot(1, len(result_sets), i + 1)
        ax.contourf(x, y, z, cmap=cmap)
        ax.scatter(*ground_truth, c='black', s=100)
        ax.text(ground_truth[0] + 0.01, ground_truth[1] + 0.01, 'Ground Truth', fontsize=font_size)
        ax.set_xlabel(f'Probability assigned to digit 0', fontsize=font_size)
        ax.set_ylabel(f'Probability assigned to digit 1', fontsize=font_size)
        ax.set_title(label, fontsize=font_size)

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=font_size + 4)
    plt.show()