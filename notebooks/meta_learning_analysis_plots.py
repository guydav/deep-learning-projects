from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import path as mpath
from matplotlib import ticker
import numpy as np

from meta_learning_data_analysis import *


DEFAULT_COLORMAP = 'tab10'
DEFAULT_LOG_SCALE_CUSTOM_TICKS = np.power(2, np.arange(8)) * 4000  # previously: (4500, 9000, 22500, 45000, 90000, 225000, 450000)


def examples_by_times_trained_on(ax, results, colors, ylim=None, log_x=False, log_y=False, shade_error=False, sem_n=1,
                                 font_dict=None, x_label=None, y_label=None, y_label_right=False, 
                                 title=None, hline_y=None, hline_style=None, log_y_custom_ticks=DEFAULT_LOG_SCALE_CUSTOM_TICKS):
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
        
        ax.plot(x_values, y_means, color=colors(task / num_points))
        if shade_error:
            ax.fill_between(x_values, y_means - y_stds, y_means + y_stds,
                            color=colors(task / num_points), alpha=0.25)
            
    if hline_y is not None:
        if hline_style is None:
            hline_style = {}
        
        ax.axhline(hline_y, **hline_style)
        
    if ylim is not None:
        ax.set_ylim(ylim)

    if log_x:
        ax.set_xscale("log", nonposx='clip')
    
    ax.set_xticks(np.arange(num_points) + 1)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    if log_y:
#         ax.set_yscale("log", nonposy='clip')
        y_min, y_max = ax.get_ylim()
#        y_min_pow_10 = np.ceil(y_min * np.log10(np.e))
#        y_max_pow_10 = np.ceil(y_max * np.log10(np.e))
        
#        y_powers_10 = np.arange(y_min_pow_10, y_max_pow_10)
#        y_ticks = np.log(10) * y_powers_10
#        y_tick_labels = [f'$10^{{ {int(y_tick)} }}$' for y_tick in y_powers_10]

        # Trying y-ticks at fixed intervals
        # real_y_min = np.exp(y_min)
        # real_y_max = np.exp(y_max)
        
        # scaled_y_min = np.ceil(real_y_min / log_y_tick_interval) * log_y_tick_interval
        # scaled_y_max = np.ceil(real_y_max / log_y_tick_interval) * log_y_tick_interval
        
        real_y_ticks = log_y_custom_ticks
        y_ticks = np.log(real_y_ticks)
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:,}' for y in real_y_ticks])
    
    if x_label is None:
        x_label = f'{log_x and "Log(" or ""}Number of times trained{log_x and ")" or ""}'
    ax.set_xlabel(x_label, **font_dict)
        
    if y_label is None:
        y_label = f'{results.name}'
    ax.set_ylabel(y_label, **font_dict)
    
    if y_label_right:
        ax.yaxis.set_label_position("right")
    
    if title is None:
        # title = f'{results.name} vs. number of tasks trained'
        title = 'Number of times trained on'
    ax.set_title(title, **font_dict)
    
    
def examples_by_num_tasks_trained(ax, results, colors, ylim=None, log_x=False, log_y=False, shade_error=False, sem_n=1,
                                 font_dict=None, x_label=None, y_label=None, y_label_right=False, 
                                  title=None, hline_y=None, hline_style=None, log_y_custom_ticks=DEFAULT_LOG_SCALE_CUSTOM_TICKS):
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
        ax.scatter(x_values, y_means, color=colors(task / num_points))
        ax.plot(x_values, y_means, color=colors(task / num_points))
        if shade_error:
            ax.fill_between(x_values, y_means - y_stds, y_means + y_stds,
                            color=colors(task / num_points), alpha=0.25)
            
    if hline_y is not None:
        if hline_style is None:
            hline_style = {}
        
        ax.axhline(hline_y, **hline_style)
    
    if ylim is not None:
        ax.set_ylim(ylim)

    if log_x:
        ax.set_xscale("log", nonposx='clip')
    
    ax.set_xticks(np.arange(num_points) + 1)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        
    if log_y:
#         ax.set_yscale("log", nonposy='clip')
        y_min, y_max = ax.get_ylim()
#        y_min_pow_10 = np.ceil(y_min * np.log10(np.e))
#        y_max_pow_10 = np.ceil(y_max * np.log10(np.e))
        
#        y_powers_10 = np.arange(y_min_pow_10, y_max_pow_10)
#        y_ticks = np.log(10) * y_powers_10
#        y_tick_labels = [f'$10^{{ {int(y_tick)} }}$' for y_tick in y_powers_10]

        # Trying y-ticks at fixed intervals
        # real_y_min = np.exp(y_min)
        # real_y_max = np.exp(y_max)
        
        # scaled_y_min = np.ceil(real_y_min / log_y_tick_interval) * log_y_tick_interval
        # scaled_y_max = np.ceil(real_y_max / log_y_tick_interval) * log_y_tick_interval
        
        real_y_ticks = log_y_custom_ticks
        y_ticks = np.log(real_y_ticks)
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:,}' for y in real_y_ticks])
        
    if x_label is None:
        x_label = f'{log_x and "Log(" or ""}Number of tasks trained{log_x and ")" or ""}'    
    ax.set_xlabel(x_label, **font_dict)
        
    if y_label is None:
        y_label = f'{results.name}'
    ax.set_ylabel(y_label, **font_dict)
    
    if y_label_right:
        ax.yaxis.set_label_position("right")
    
    if title is None:
        title = f'Number of tasks trained on'
    ax.set_title(title, **font_dict)

    
DEFAULT_Y_LABEL = 'Log(examples to criterion)'
    

def plot_processed_results_all_dimensions(result_set, data_index, title, ylim=None, log_x=False, log_y=None,
                                          sem_n=1, shade_error=False, font_dict=None, plot_y_label=DEFAULT_Y_LABEL,
                                          times_trained_colormap=DEFAULT_COLORMAP, tasks_trained_colormap=DEFAULT_COLORMAP,
                                          log_y_custom_ticks=DEFAULT_LOG_SCALE_CUSTOM_TICKS):
    NROWS = 4
    NCOLS = 2
    COL_WIDTH = 5
    ROW_HEIGHT = 5 
    WIDTH_SPACING = 1
    HEIGHT_SPACING = 0
    
    figure = plt.figure(figsize=(NCOLS * COL_WIDTH + WIDTH_SPACING, NROWS * ROW_HEIGHT + HEIGHT_SPACING))
    plt.subplots_adjust(top=0.925, hspace=0.2, wspace=0.25)
    
    if font_dict is None:
        font_dict = {}
    
    figure.suptitle(title, fontsize=font_dict['fontsize'] * 1.5)

    if log_y is None:
        log_y = 'log' in result_set[0][data_index].name
        
    if not hasattr(sem_n, '__len__'):
        sem_n = [sem_n] * len(CONDITION_ANALYSES_FIELDS)
        
    if isinstance(times_trained_colormap, str):
        times_trained_colormap = plt.get_cmap(times_trained_colormap)
        
    if isinstance(tasks_trained_colormap, str):
        tasks_trained_colormap = plt.get_cmap(tasks_trained_colormap)
        
    for dimension_index, dimension_name in enumerate(CONDITION_ANALYSES_FIELDS):
        num_times_trained_ax = plt.subplot(NROWS, NCOLS, NCOLS * dimension_index + 1)
            
        results = result_set[dimension_index][data_index]

        title = ''
        if dimension_index == 0:
            title = None  # sets the default title for this plot

        x_label = ''
        if dimension_index == NROWS - 1:
            x_label = None  # sets the default x-label for this plot

        y_label = plot_y_label
        
        examples_by_times_trained_on(num_times_trained_ax, results, times_trained_colormap, ylim=ylim, 
                                     log_x=log_x, log_y=log_y, shade_error=shade_error, sem_n=sem_n[dimension_index],
                                     font_dict=font_dict, x_label=x_label, y_label=y_label, 
                                     title=title, log_y_custom_ticks=log_y_custom_ticks)

        num_tasks_trained_ax = plt.subplot(NROWS, NCOLS, NCOLS * dimension_index + 2)
        y_label = dimension_name.capitalize()
        
        examples_by_num_tasks_trained(num_tasks_trained_ax, results, tasks_trained_colormap, ylim=ylim, 
                                      log_x=log_x, log_y=log_y, shade_error=shade_error, sem_n=sem_n[dimension_index],
                                      font_dict=font_dict, x_label=x_label, y_label=y_label, y_label_right=True, 
                                      title=title, log_y_custom_ticks=log_y_custom_ticks)
        
    plt.show()

    
PER_MODEL_NROWS = 5
PER_MODEL_NCOLS = 4

PER_MODEL_COL_WIDTH = 5
PER_MODEL_ROW_HEIGHT = 6
    
    
def plot_per_model_per_dimension(baseline, per_query_replication, plot_func, super_title,
                                 font_dict=None, colormap=DEFAULT_COLORMAP, 
                                 ylim=None, log_x=True, log_y=True, shade_error=True, 
                                 sem_n=1, baseline_sem_n=1, data_index=None, plot_y_label=DEFAULT_Y_LABEL):
    
    plt.figure(figsize=(PER_MODEL_NCOLS * (PER_MODEL_COL_WIDTH + 1), 
                        PER_MODEL_NROWS * PER_MODEL_ROW_HEIGHT))
    plt.subplots_adjust(top=0.94, hspace=0.25, wspace=0.25)
    
    if font_dict is None:
        font_dict = {}

    if not hasattr(sem_n, '__len__'):
        sem_n = [sem_n] * len(CONDITION_ANALYSES_FIELDS)
        
    if not hasattr(baseline_sem_n, '__len__'):
        baseline_sem_n = [baseline_sem_n] * len(CONDITION_ANALYSES_FIELDS)
        
    if data_index is None:
        data_index = int(log_y)
        
    colors = plt.get_cmap(colormap)
    plt.suptitle(super_title, fontsize=font_dict['fontsize'] * 1.5)
    
    # plot the baseline
    for dimension_index, dimension_name in enumerate(CONDITION_ANALYSES_FIELDS):
        ax = plt.subplot(PER_MODEL_NROWS, PER_MODEL_NCOLS, dimension_index + 1)
            
        results = baseline[dimension_index][data_index]

        title = dimension_name.capitalize()

        x_label = ''

        y_label = ''
        y_label_right = False
        if dimension_index == 0:
            y_label = plot_y_label
        elif dimension_index == 3:
            y_label = f'No query\nmodulation'
            y_label_right = True

        plot_func(ax, results, colors, ylim=ylim, log_x=log_x, log_y=log_y, shade_error=shade_error, 
                  sem_n=baseline_sem_n[dimension_index], font_dict=font_dict, 
                  x_label=x_label, y_label=y_label, y_label_right=y_label_right, title=title)
    
    # plot per query
    for replication_level, replication_analyses in per_query_replication.items():
        for dimension_index, dimension_name in enumerate(CONDITION_ANALYSES_FIELDS):
            ax = plt.subplot(PER_MODEL_NROWS, PER_MODEL_NCOLS, 
                             replication_level * PER_MODEL_NCOLS + dimension_index + 1)
            
            results = replication_analyses[dimension_index][data_index]

            title = ''
#             if replication_level == 1:
#                 title = dimension_name.capitalize()
                
            x_label = ''
            if replication_level + 1 == PER_MODEL_NROWS:
                x_label = None
                
            y_label = ''
            y_label_right = False
            if dimension_index == 0:
                y_label = plot_y_label
            elif dimension_index == 3:
                y_label = f'Query modulation\nat conv-{replication_level}'
                y_label_right = True
    
            plot_func(ax, results, colors, ylim=ylim, log_x=log_x, log_y=log_y, shade_error=shade_error, 
                      sem_n=sem_n[dimension_index], font_dict=font_dict, 
                      x_label=x_label, y_label=y_label, y_label_right=y_label_right, title=title)
        
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
    
    
DEFAULT_COMPARISON_HLINE_STYLE = dict(linestyle='--', linewidth=4, color='black', alpha=0.25)

    
def combined_comparison_plots(baseline, per_query_replication, super_title,
                              comparison_mod_level, dimension_index=COMBINED_INDEX, comparison_func=np.subtract,
                              font_dict=None, comparison_first=False, ylim=None, data_index=None, 
                              log_x=True, log_y=True, shade_error=True, sem_n=1, baseline_sem_n=1, 
                              times_trained_colormap=DEFAULT_COLORMAP, tasks_trained_colormap=DEFAULT_COLORMAP,
                              null_hline_y=None, null_hline_style=None, plot_y_label=''):
    
    if comparison_mod_level < 0 or comparison_mod_level > 4:
        raise ValueError('Comparison model level should be between 0 and 4 inclusive')
        
    COMPARISON_NROWS = (PER_MODEL_NROWS - 1)
    COMPARISON_NCOLS = 2
    COL_WIDTH = 5
    ROW_HEIGHT = 5 
    WIDTH_SPACING = 1
    HEIGHT_SPACING = 0
    
    figure = plt.figure(figsize=(COMPARISON_NCOLS * COL_WIDTH + WIDTH_SPACING, 
                                 COMPARISON_NROWS * ROW_HEIGHT + HEIGHT_SPACING))
    plt.subplots_adjust(top=0.925 - 0.02 * super_title.count('\n'), hspace=0.25, wspace=0.2)
    
    if font_dict is None:
        font_dict = {}
        
    if data_index is None:
        data_index = int(log_y)
        
    if null_hline_y is None:
        if comparison_func == np.subtract:
            null_hline_y = 0
        elif comparison_func == np.divide:
            null_hline_y = 1
        else:
            raise ValueError('If not using np.subract or np.divide, please provide null_hline_y value')
            
    hline_style = DEFAULT_COMPARISON_HLINE_STYLE.copy()
    if null_hline_style is not None:
        hline_style.update(null_hline_style)
    
    null_hline_style = hline_style
        
    plt.suptitle(super_title, fontsize=font_dict['fontsize'] * 1.5)
    
    if comparison_mod_level == 0:
        comparison_set = baseline
    else:
        comparison_set = per_query_replication[comparison_mod_level]
        
    times_trained_colors = plt.get_cmap(times_trained_colormap)
    tasks_trained_colors = plt.get_cmap(tasks_trained_colormap)
    
    # plot the baseline
    if comparison_mod_level != 0:
        num_times_trained_ax = plt.subplot(COMPARISON_NROWS, COMPARISON_NCOLS, 1)

        if comparison_first:
            results_mean = comparison_func(comparison_set[dimension_index][data_index].mean,
                                           baseline[dimension_index][data_index].mean)
        else:
            results_mean = comparison_func(comparison_set[dimension_index][data_index].mean,
                                           baseline[dimension_index][data_index].mean)
            
        # TODO: fix the stddev computation if we ever decide to use it here
        results_std = comparison_set[dimension_index][data_index].std
        results = ResultSet(name='diff', mean=results_mean, std=results_std)

        title = None  # use the default title
        x_label = ''
        y_label = plot_y_label

        examples_by_times_trained_on(num_times_trained_ax, results, times_trained_colors, ylim=ylim, 
                                     log_x=log_x, log_y=log_y, shade_error=shade_error, sem_n=baseline_sem_n,
                                     font_dict=font_dict, x_label=x_label, y_label=y_label, 
                                     title=title, hline_y=null_hline_y, hline_style=null_hline_style)

        num_tasks_trained_ax = plt.subplot(COMPARISON_NROWS, COMPARISON_NCOLS, 2)
        y_label = f'No query\nmodulation'

        examples_by_num_tasks_trained(num_tasks_trained_ax, results, tasks_trained_colors, ylim=ylim, 
                                      log_x=log_x, log_y=log_y, shade_error=shade_error, sem_n=baseline_sem_n,
                                      font_dict=font_dict, x_label=x_label, y_label=y_label, y_label_right=True,
                                      title=title, hline_y=null_hline_y, hline_style=null_hline_style)
    
    # plot per query
    for replication_level, replication_analyses in per_query_replication.items():
        if replication_level == comparison_mod_level:
            continue

        replication_level_for_axes = replication_level
        if replication_level > comparison_mod_level:
            replication_level_for_axes -= 1

        num_times_trained_ax = plt.subplot(COMPARISON_NROWS, COMPARISON_NCOLS, 
                                           replication_level_for_axes * COMPARISON_NCOLS + 1)

        if comparison_first:
            results_mean = comparison_func(comparison_set[dimension_index][data_index].mean,
                                           replication_analyses[dimension_index][data_index].mean)
        else:
            results_mean = comparison_func(replication_analyses[dimension_index][data_index].mean,
                                           comparison_set[dimension_index][data_index].mean)

        # TODO: fix the stddev computation if we ever decide to use it
        results_std = comparison_set[dimension_index][data_index].std
        results = ResultSet(name='diff', mean=results_mean, std=results_std)

        title = ''
        if replication_level == 1 and comparison_mod_level == 0:
            title = None  # use the default title

        x_label = ''
        if replication_level_for_axes + 1 == COMPARISON_NROWS:
            x_label = None  # use the default x-label

        y_label = plot_y_label

        examples_by_times_trained_on(num_times_trained_ax, results, times_trained_colors, ylim=ylim, 
                                     log_x=log_x, log_y=log_y, shade_error=shade_error, sem_n=sem_n,
                                     font_dict=font_dict, x_label=x_label, y_label=y_label, 
                                     title=title, hline_y=null_hline_y, hline_style=null_hline_style)

        num_tasks_trained_ax = plt.subplot(COMPARISON_NROWS, COMPARISON_NCOLS, 
                                           replication_level_for_axes * COMPARISON_NCOLS + 2)
        y_label = f'Query modulation\nat conv-{replication_level}'

        examples_by_num_tasks_trained(num_tasks_trained_ax, results, tasks_trained_colors, ylim=ylim, 
                                      log_x=log_x, log_y=log_y, shade_error=shade_error, sem_n=sem_n,
                                      font_dict=font_dict, x_label=x_label, y_label=y_label, y_label_right=True,
                                      title=title, hline_y=null_hline_y, hline_style=null_hline_style)
        
    plt.show()
    