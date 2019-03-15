from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import path as mpath

import numpy as np

from meta_learning_data_analysis import *

DEFAULT_COLORMAP = 'tab10'


def examples_by_times_trained_on(ax, results, colors, ylim=None, log_x=False, log_y=False, shade_error=False, sem_n=1,
                                 font_dict=None, x_label=None, y_label=None, title=None):
    if font_dict is None:
        font_dict = {}
        
    num_points = results.mean.shape[0]
    nonzero_rows, nonzero_cols = np.nonzero(results.mean)
    means = [results.mean[r, c] for (r, c) in zip(nonzero_rows, nonzero_cols)]
    
    
    ax.scatter(nonzero_rows + 1, means, 
                                 color=[colors(x / num_points) for x 
                                        in abs(nonzero_cols - nonzero_rows)])
    
    # TODO: verify this does the right thing for the accuracies. It might not
    for task in range(num_points):
        x_values = np.arange(1, num_points - task + 1)
        y_means = np.diag(results.mean, task)
        y_stds = np.diag(results.std, task) / (sem_n ** 0.5)
        
        ax.plot(x_values, y_means, color=colors(task / 10))
        if shade_error:
            ax.fill_between(x_values, y_means - y_stds, y_means + y_stds,
                            color=colors(task / 10), alpha=0.25)
        
    if ylim is not None:
        ax.set_ylim(ylim)

    if log_x:
        ax.set_xscale("log", nonposx='clip')
    
    if log_y:
#         ax.set_yscale("log", nonposy='clip')
        y_min, y_max = ax.get_ylim()
        y_min_pow_10 = np.ceil(y_min * np.log10(np.e))
        y_max_pow_10 = np.ceil(y_max * np.log10(np.e))
        
        y_powers_10 = np.arange(y_min_pow_10, y_max_pow_10)
        y_ticks = np.log(10) * y_powers_10
        y_tick_labels = [f'$10^{{ {int(y_tick)} }}$' for y_tick in y_powers_10]
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    
    if x_label is None:
        x_label = f'{log_x and "log(" or ""}Number of times trained{log_x and ")" or ""}'
    ax.set_xlabel(x_label, **font_dict)
        
    if y_label is None:
        y_label = f'{results.name}'
    ax.set_ylabel(y_label, **font_dict)
    
    if title is None:
        title = f'{results.name} vs. number of tasks trained'
    ax.set_title(title, **font_dict)
    
    
def examples_by_num_tasks_trained(ax, results, colors, ylim=None, log_x=False, log_y=False, shade_error=False, sem_n=1,
                                 font_dict=None, x_label=None, y_label=None, title=None):
    if font_dict is None:
        font_dict = {}
        
    num_points = results.mean.shape[0]
    nonzero_rows, nonzero_cols = np.nonzero(results.mean)
    means = [results.mean[r, c] for (r, c) in zip(nonzero_rows, nonzero_cols)]
    
#     ax.scatter(nonzero_cols + 1, results.mean, 
#                                  color=[colors(x / 10) for x in abs(nonzero_cols - nonzero_rows)])
    for task in range(num_points):
        x_values = np.arange(task + 1, num_points + 1)
#         y_means = np.diag(results.mean, task)
#         y_stds = np.diag(results.std, task) / (sem_n ** 0.5)
        y_means = results.mean[task, task:]
        y_stds = results.std[task, task:] / (sem_n ** 0.5)
        ax.scatter(x_values, y_means, color=colors(task / 10))
        ax.plot(x_values, y_means, color=colors(task / 10))
        if shade_error:
            ax.fill_between(x_values, y_means - y_stds, y_means + y_stds,
                            color=colors(task / 10), alpha=0.25)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if log_x:
        ax.set_xscale("log", nonposx='clip')
        
    if log_y:
#         ax.set_yscale("log", nonposy='clip')
        y_min, y_max = ax.get_ylim()
        y_min_pow_10 = np.ceil(y_min * np.log10(np.e))
        y_max_pow_10 = np.ceil(y_max * np.log10(np.e))
        
        y_powers_10 = np.arange(y_min_pow_10, y_max_pow_10)
        y_ticks = np.log(10) * y_powers_10
        y_tick_labels = [f'$10^{{ {int(y_tick)} }}$' for y_tick in y_powers_10]
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        
    if x_label is None:
        x_label = f'{log_x and "log(" or ""}Number of tasks trained{log_x and ")" or ""}'    
    ax.set_xlabel(x_label, **font_dict)
        
    if y_label is None:
        y_label = f'{results.name}'
    ax.set_ylabel(y_label, **font_dict)
    
    if title is None:
        title = f'{results.name} vs. tasks trained on'
    ax.set_title(title, **font_dict)


def plot_processed_results(results, title, ylim=None, log_x=False, log_y=None,
                           sem_n=1, shade_error=False, plot_bottom=False, colormap=DEFAULT_COLORMAP):
    figure = plt.figure(figsize=(12, 6 + 6 * plot_bottom))
    NROWS = 1 + plot_bottom
    NCOLS = 2
    
    figure.suptitle(title)
    colors = plt.get_cmap(colormap)
    
    if hasattr(log_x, '__len__'):
        if len(log_x) == 1:
            log_x_num_times_trained = log_x[0]
            log_x_num_current_tasks = log_x[0]
        else:
            log_x_num_times_trained = log_x[0]
            log_x_num_current_tasks = log_x[1]
            
    else:
        log_x_num_times_trained = log_x
        log_x_num_current_tasks = log_x
        
    if log_y is None:
        log_y = 'log' in results.name
        
    num_times_trained_ax = plt.subplot(NROWS, NCOLS, 1)
    examples_by_times_trained_on(num_times_trained_ax, results, colors, ylim, 
                                 log_x_num_times_trained, log_y, shade_error, sem_n)
    
    num_current_tasks_ax = plt.subplot(NROWS, NCOLS, 2)
    examples_by_num_tasks_trained(num_current_tasks_ax, results, colors, ylim, 
                                  log_x_num_current_tasks, log_y, shade_error, sem_n)
            
#     if plot_bottom:
#         # Accuracy as a function of both
#         both_ax = plt.subplot(NROWS, NCOLS, 3, projection='3d')
#         both_ax.scatter(nonzero_rows, nonzero_cols, values)

#         both_ax.set_zscale("log")
#         both_ax.set_xlabel('Number of times trained on')
#         both_ax.set_ylabel('Number of tasks trained on')
#         both_ax.set_zlabel('log(examples to criterion)')
#         both_ax.set_title('# examples to criterion vs. both')

#         # Accuracy as a function of both in a heatmap
#         heatmap_ax = plt.subplot(NROWS, NCOLS, 4)
#         heatmap_ax.imshow(np.log(examples_to_criterion + 1), cmap='YlOrRd')
#         for i in range(10):
#             for j in range(i, 10):
#                 text = heatmap_ax.text(j, i, examples_to_criterion[i, j],
#                                ha="center", va="center", color="w", fontsize=8)


#         heatmap_ax.set_xlabel('Number of times trained on')
#         heatmap_ax.set_ylabel('Number of tasks trained on')
#         heatmap_ax.set_title('Heatmap of # examples to criterion')
    
    plt.show()

    
PER_MODEL_NROWS = 5
PER_MODEL_NCOLS = 4

PER_MODEL_COL_WIDTH = 5
PER_MODEL_ROW_HEIGHT = 6
    
    
def plot_per_model_per_dimension(baseline, per_query_replication, plot_func, super_title,
                                 font_dict=None, colormap=DEFAULT_COLORMAP, 
                                 ylim=None, log_x=True, log_y=True, shade_error=True, 
                                 sem_n=1, baseline_sem_n=1):
    
    plt.figure(figsize=(PER_MODEL_NCOLS * (PER_MODEL_COL_WIDTH + 1), 
                        PER_MODEL_NROWS * PER_MODEL_ROW_HEIGHT))
    plt.subplots_adjust(top=0.925, hspace=0.25, wspace=0.15)
    
    if font_dict is None:
        font_dict = {}

    if not hasattr(sem_n, '__len__'):
        sem_n = [sem_n] * len(CONDITION_ANALYSES_FIELDS)
        
    if not hasattr(baseline_sem_n, '__len__'):
        baseline_sem_n = [baseline_sem_n] * len(CONDITION_ANALYSES_FIELDS)
        
    colors = plt.get_cmap(colormap)
    plt.suptitle(super_title, fontsize=font_dict['fontsize'] * 1.5)
    
    # plot the baseline
    for dimension_index, dimension_name in enumerate(CONDITION_ANALYSES_FIELDS):
        ax = plt.subplot(PER_MODEL_NROWS, PER_MODEL_NCOLS, dimension_index + 1)
            
        if log_y:
            results = baseline[dimension_index].log_examples
        else:
            results = baseline[dimension_index].examples

        title = dimension_name.capitalize()

        x_label = ''

        y_label = ''
        if dimension_index == 0:
            y_label = f'No query\nmodulation'

        plot_func(ax, results, colors, ylim=ylim, log_x=log_x, log_y=log_y, shade_error=shade_error, 
                  sem_n=baseline_sem_n[dimension_index], font_dict=font_dict, x_label=x_label, y_label=y_label, title=title)
    
    # plot per query
    for replication_level, replication_analyses in per_query_replication.items():
        for dimension_index, dimension_name in enumerate(CONDITION_ANALYSES_FIELDS):
            ax = plt.subplot(PER_MODEL_NROWS, PER_MODEL_NCOLS, 
                             replication_level * PER_MODEL_NCOLS + dimension_index + 1)
            
            if log_y:
                results = replication_analyses[dimension_index].log_examples
            else:
                results = replication_analyses[dimension_index].examples

            title = ''
#             if replication_level == 1:
#                 title = dimension_name.capitalize()
                
            x_label = ''
            if replication_level + 1 == PER_MODEL_NROWS:
                x_label = None
                
            y_label = ''
            if dimension_index == 0:
                y_label = f'Query modulation\nat conv-{replication_level}'
    
            plot_func(ax, results, colors, ylim=ylim, log_x=log_x, log_y=log_y, shade_error=shade_error, 
                      sem_n=sem_n[dimension_index], font_dict=font_dict, x_label=x_label, y_label=y_label, title=title)
        
    plt.show()
    
    
def comparison_plot_per_model(baseline, per_query_replication, plot_func, super_title,
                              comparison_mod_level, conditions=None, comparison_func=np.subtract,
                              font_dict=None, colormap=DEFAULT_COLORMAP, comparison_first=False,
                              ylim=None, data_index=None, log_x=True, log_y=True, shade_error=True, 
                              sem_n=1, baseline_sem_n=1):
    
    if comparison_mod_level < 0 or comparison_mod_level > 4:
        raise ValueError('Comparison model level should be between 0 and 4 inclusive')
    
    if conditions is None:
        conditions = list(range(len(CONDITION_ANALYSES_FIELDS)))
        
    if not hasattr(conditions, '__len__'):
        conditions = (conditions, )
        
    COMPARISON_NROWS = (PER_MODEL_NROWS - 1)
    COMPARISON_NCOLS = len(conditions)
    
    plt.figure(figsize=(COMPARISON_NCOLS * (PER_MODEL_COL_WIDTH + 5 - len(conditions)), 
                        COMPARISON_NROWS * PER_MODEL_ROW_HEIGHT))
    plt.subplots_adjust(top=0.925, hspace=0.25, wspace=0.15)
    
    if font_dict is None:
        font_dict = {}
        
    if data_index is None:
        data_index = int(log_y)

    if not hasattr(sem_n, '__len__'):
        sem_n = [sem_n] * len(CONDITION_ANALYSES_FIELDS)
        
    if not hasattr(baseline_sem_n, '__len__'):
        baseline_sem_n = [baseline_sem_n] * len(CONDITION_ANALYSES_FIELDS)
        
    colors = plt.get_cmap(colormap)
    plt.suptitle(super_title, fontsize=font_dict['fontsize'] * 1.5)
    
    if comparison_mod_level == 0:
        comparison_set = baseline
    else:
        comparison_set = per_query_replication[comparison_mod_level]
    
    # plot the baseline
    if comparison_mod_level != 0:
        for index, (dimension_index, dimension_name) in enumerate([(c, CONDITION_ANALYSES_FIELDS[c]) for c in conditions]):
            ax = plt.subplot(COMPARISON_NROWS, COMPARISON_NCOLS, index + 1)
            
            if comparison_first:
                results_mean = comparison_func(comparison_set[dimension_index][data_index].mean,
                                               baseline[dimension_index][data_index].mean,)
            else:
                results_mean = comparison_func(comparison_set[dimension_index][data_index].mean,
                                               baseline[dimension_index][data_index].mean,)
            # TODO: fix the stddev computation
            results_std = comparison_set[dimension_index][data_index].std
            results = ResultSet(name='diff', mean=results_mean, std=results_std)

            title = dimension_name.capitalize()

            x_label = ''

            y_label = ''
            if index == 0:
                y_label = f'No query\nmodulation'

            plot_func(ax, results, colors, ylim=ylim, log_x=log_x, log_y=log_y, shade_error=shade_error, 
                      sem_n=baseline_sem_n[dimension_index], font_dict=font_dict, 
                      x_label=x_label, y_label=y_label, title=title)
    
    # plot per query
    for replication_level, replication_analyses in per_query_replication.items():
        if replication_level == comparison_mod_level:
            continue
        
        for index, (dimension_index, dimension_name) in enumerate([(c, CONDITION_ANALYSES_FIELDS[c]) for c in conditions]):
            replication_level_for_axes = replication_level
            if replication_level > comparison_mod_level:
                replication_level_for_axes -= 1
            
            ax = plt.subplot(COMPARISON_NROWS, COMPARISON_NCOLS, 
                             replication_level_for_axes * COMPARISON_NCOLS + index + 1)


            if comparison_first:
                results_mean = comparison_func(comparison_set[dimension_index][data_index].mean,
                                               replication_analyses[dimension_index][data_index].mean,)
            else:
                results_mean = comparison_func(replication_analyses[dimension_index][data_index].mean,
                                               comparison_set[dimension_index][data_index].mean,
                                               )
            
            # TODO: fix the stddev computation
            results_std = comparison_set[dimension_index][data_index].std
            results = ResultSet(name='diff', mean=results_mean, std=results_std)
                
            title = ''
            if replication_level == 1 and comparison_mod_level == 0:
                title = dimension_name.capitalize()
                
            x_label = ''
            if replication_level_for_axes + 1 == COMPARISON_NROWS:
                x_label = None
                
            y_label = ''
            if index == 0:
                y_label = f'Query modulation\nat conv-{replication_level}'
    
            plot_func(ax, results, colors, ylim=ylim, log_x=log_x, log_y=log_y, shade_error=shade_error, 
                      sem_n=sem_n[dimension_index], font_dict=font_dict, x_label=x_label, y_label=y_label, title=title)
        
    plt.show()
    