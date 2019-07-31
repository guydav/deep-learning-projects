import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.special import factorial
from scipy.stats import binom_test, wilcoxon

import os
import pickle
from datetime import datetime
import tabulate
import wandb
from collections import namedtuple, defaultdict, OrderedDict
import json

API = wandb.Api()
MAX_HISTORY_SAMPLES = 4000


DATASET_CORESET_SIZE = 22500
ACCURACY_THRESHOLD = 0.95
TASK_ACC_COLS = [f'Test Accuracy, Query #{i}' for i in range(1, 11)]


QUERY_NAMES = ['blue', 'brown', 'cyan', 'gray', 'green', 
               'orange', 'pink', 'purple', 'red', 'yellow', 
               'cone', 'cube', 'cylinder', 'dodecahedron', 'ellipsoid',
               'octahedron', 'pyramid', 'rectangle', 'sphere', 'torus', 
               'chain_mail', 'marble', 'maze', 'metal', 'metal_weave',
               'polka', 'rubber', 'rug', 'tiles', 'wood_plank']

COLOR = 'color'
COLOR_INDEX = 0
SHAPE = 'shape'
SHAPE_INDEX = 1
TEXTURE = 'texture'
TEXTURE_INDEX = 2
COMBINED = 'combined'
COMBINED_INDEX = 3
DIMENSION_NAMES = [COLOR, SHAPE, TEXTURE]

RESULT_SET_FIELDS = ['name', 'mean', 'std']
ANALYSIS_SET_FIELDS = ['examples', 'log_examples', 'accuracies', 'accuracy_drops',
                       'first_task_accuracies', 'new_task_accuracies', 'accuracy_counts',
                       'examples_by_task']
CONDITION_ANALYSES_FIELDS = DIMENSION_NAMES + [COMBINED]
TOTAL_CURVE_FIELDS = ['raw', 'mean', 'std', 'sem']

ResultSet = namedtuple('ResultSet', RESULT_SET_FIELDS)
AnalysisSet = namedtuple('AnalysisSet', ANALYSIS_SET_FIELDS)
ConditionAnalysesSet = namedtuple('ConditionAnalysesSet', CONDITION_ANALYSES_FIELDS)
TotalCurveResults = namedtuple('TotalCurveResults', TOTAL_CURVE_FIELDS)

NAMED_TUPLE_CLASSES = (ResultSet, AnalysisSet, ConditionAnalysesSet, TotalCurveResults)
for NamedTupleClass in NAMED_TUPLE_CLASSES:
    NamedTupleClass.__new__.__defaults__ = (None,) * len(NamedTupleClass._fields)


RESULT_SET_NAMES = ('Examples to criterion', 'Log examples to criterion', 
                    'New task accuracy', 'New task accuracy delta', 
                    'First task accuracy by epoch', 'New task accuracy by epoch')
ANALYSIS_FIELDS_TO_NAMES = {field: name for (name, field) in 
                            zip(ANALYSIS_SET_FIELDS, RESULT_SET_NAMES)}
ANALYSIS_NAMES_TO_FIELDS = {name: field for (name, field) in 
                            zip(ANALYSIS_SET_FIELDS, RESULT_SET_NAMES)}


CACHE_PATH = './analyses_caches/meta_learning_analyses_cache.pickle'
BACKUP_CACHE_PATH = './analyses_caches/meta_learning_analyses_cache_{date}.pickle'


def refresh_cache(new_values_dict=None, cache_path=CACHE_PATH):
    if new_values_dict is None:
        new_values_dict = {}
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            cache = pickle.load(cache_file)
    
    else:
        cache = {}
    
    cache.update(new_values_dict)
    
    if os.path.exists(cache_path):
        os.rename(CACHE_PATH, BACKUP_CACHE_PATH.format(date=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))

    with open(cache_path, 'wb') as cache_file:
        pickle.dump(cache, cache_file)

    return cache
    

def examples_per_epoch(task, latest_task):
    if task == latest_task:
        return 22500
    
    return 22500 // (latest_task - 1)
            

def parse_run_results(current_run_id=None, current_run=None, samples=MAX_HISTORY_SAMPLES):
    if current_run_id is None and current_run is None:
        print('Must provide either a current run or its id')
        return
    
    if current_run is None:
        current_run = API.run(f'meta-learning-scaling/sequential-benchmark-baseline/{current_run_id}')
        
    current_df = current_run.history(pandas=True, samples=samples)
    
    examples_to_criterion = np.empty((10, 10))
    examples_to_criterion.fill(np.nan)
    absolute_accuracy = np.empty((9, 9))
    absolute_accuracy.fill(np.nan)
    accuracy_drop = np.empty((9, 9))
    accuracy_drop.fill(np.nan)
    
    first_task_accuracy_by_epoch = np.empty((10, samples))
    first_task_accuracy_by_epoch.fill(np.nan)
    new_task_accuracy_by_epoch = np.empty((10, samples))
    new_task_accuracy_by_epoch.fill(np.nan)
    
    first_row_blank = int(np.isnan(current_df['Test Accuracy'][0]))
    
    first_task_finished = current_df['Test Accuracy, Query #2'].first_valid_index() - first_row_blank
    examples_to_criterion[0, 0] = first_task_finished * examples_per_epoch(1, 1)
    absolute_accuracy[0, 0] = current_df['Test Accuracy, Query #1'][first_task_finished + 1]
    accuracy_drop[0, 0] = current_df['Test Accuracy, Query #1'][first_task_finished] - absolute_accuracy[0, 0]
    
    first_task_accuracy_by_epoch[0, 0:first_task_finished] = current_df['Test Accuracy, Query #1'][first_row_blank:first_task_finished + 1]
    new_task_accuracy_by_epoch[0, 0:first_task_finished] = current_df['Test Accuracy, Query #1'][first_row_blank:first_task_finished + 1]

    for current_task in range(2, 11):
        current_task_start = current_df[f'Test Accuracy, Query #{current_task}'].first_valid_index()

        if current_task == 10:
            current_task_end = current_df.shape[0]
        else:
            current_task_end = current_df[f'Test Accuracy, Query #{current_task + 1}'].first_valid_index()

        current_task_subset = current_df[TASK_ACC_COLS][current_task_start:current_task_end] > 0.95

        for task in range(1, current_task + 1):
            number_times_learned = current_task - task + 1
            number_total_tasks = current_task

            examples_to_criterion[number_times_learned - 1, number_total_tasks - 1] = examples_per_epoch(task, current_task) * \
                (current_task_subset[f'Test Accuracy, Query #{task}'].idxmax() - current_task_start + 1)
            
            if current_task < 10:
                absolute_accuracy[number_times_learned - 1, number_total_tasks - 1] = \
                    current_df[f'Test Accuracy, Query #{task}'][current_task_end]
                accuracy_drop[number_times_learned - 1, number_total_tasks - 1] = \
                    current_df[f'Test Accuracy, Query #{task}'][current_task_end - 1] - \
                    absolute_accuracy[number_times_learned - 1, number_total_tasks - 1]
                
        first_task_accuracy_by_epoch[current_task - 1, 0:current_task_end - current_task_start] = current_df['Test Accuracy, Query #1'][current_task_start:current_task_end]
        new_task_accuracy_by_epoch[current_task - 1, 0:current_task_end - current_task_start] = current_df[f'Test Accuracy, Query #{current_task}'][current_task_start:current_task_end]
        
            
    return examples_to_criterion, absolute_accuracy, accuracy_drop, first_task_accuracy_by_epoch, new_task_accuracy_by_epoch


def parse_run_results_with_new_task_accuracy_and_equal_size(current_run_id=None, current_run=None, samples=MAX_HISTORY_SAMPLES):
    if current_run_id is None and current_run is None:
        print('Must provide either a current run or its id')
        return
    
    if current_run is None:
        current_run = API.run(f'meta-learning-scaling/sequential-benchmark-baseline/{current_run_id}')
        
    current_df = current_run.history(pandas=True, samples=samples)
    
    examples_to_criterion = np.empty((10, 10))
    examples_to_criterion.fill(np.nan)
    absolute_accuracy = np.empty((10, 10))
    absolute_accuracy.fill(np.nan)
    absolute_accuracy_equal_size = np.empty((10, 10))
    absolute_accuracy_equal_size.fill(np.nan)
    
    first_task_accuracy_by_epoch = np.empty((10, samples))
    first_task_accuracy_by_epoch.fill(np.nan)
    new_task_accuracy_by_epoch = np.empty((10, samples))
    new_task_accuracy_by_epoch.fill(np.nan)
    
    first_row_blank = int(np.isnan(current_df['Test Accuracy'][0]))
    
    # import pdb; pdb.set_trace()
    
    first_task_finished = current_df['Test Accuracy, Query #2'].first_valid_index() - first_row_blank
    examples_to_criterion[0, 0] = first_task_finished * examples_per_epoch(1, 1)
    absolute_accuracy[0, 0] = current_df['Test Accuracy, Query #1'][first_row_blank] # index 0 is all NaNs, presumably the initalization
    absolute_accuracy_equal_size[0, 0] = current_df['Test Accuracy, Query #1'][first_row_blank]
    
    first_task_accuracy_by_epoch[0, 0:first_task_finished] = current_df['Test Accuracy, Query #1'][first_row_blank:first_task_finished]
    new_task_accuracy_by_epoch[0, 0:first_task_finished] = current_df['Test Accuracy, Query #1'][first_row_blank:first_task_finished]

    for current_task in range(2, 11):
        current_task_start = current_df[f'Test Accuracy, Query #{current_task}'].first_valid_index()

        if current_task == 10:
            current_task_end = current_df.shape[0]
        else:
            current_task_end = current_df[f'Test Accuracy, Query #{current_task + 1}'].first_valid_index()

        current_task_subset = current_df[TASK_ACC_COLS][current_task_start:current_task_end] > 0.95

        for task in range(1, current_task + 1):
            number_times_learned = current_task - task + 1
            number_total_tasks = current_task

            examples_to_criterion[number_times_learned - 1, number_total_tasks - 1] = examples_per_epoch(task, current_task) * \
                (current_task_subset[f'Test Accuracy, Query #{task}'].idxmax() - current_task_start + 1)
            
            absolute_accuracy[number_times_learned - 1, number_total_tasks - 1] = \
                current_df[f'Test Accuracy, Query #{task}'][current_task_start]
            
            if task == current_task:
                equal_size_epochs = current_task_start
            else:
                equal_size_epochs = min(current_task_start + current_task - 1, current_task_end - 1)
            
            absolute_accuracy_equal_size[number_times_learned - 1, number_total_tasks - 1] = \
                current_df[f'Test Accuracy, Query #{task}'][equal_size_epochs]

                
        first_task_accuracy_by_epoch[current_task - 1, 0:current_task_end - current_task_start] = current_df['Test Accuracy, Query #1'][current_task_start:current_task_end]
        new_task_accuracy_by_epoch[current_task - 1, 0:current_task_end - current_task_start] = current_df[f'Test Accuracy, Query #{current_task}'][current_task_start:current_task_end]
        
            
    return examples_to_criterion, absolute_accuracy, absolute_accuracy_equal_size, first_task_accuracy_by_epoch, new_task_accuracy_by_epoch


def parse_forgetting_results(current_run_id=None, current_run=None, samples=MAX_HISTORY_SAMPLES, 
                             max_samples_per_curve=31):
    if current_run_id is None and current_run is None:
        print('Must provide either a current run or its id')
        return
    
    if current_run is None:
        current_run = API.run(f'meta-learning-scaling/sequential-benchmark-forgetting-experiment-revisited/{current_run_id}')
        
    print(f'Starting to parse run {current_run.name}')
    raw_df = current_run.history(pandas=True, samples=samples)
    
    forgetting_trjectories = np.empty((10, 10, max_samples_per_curve))
    forgetting_trjectories.fill(np.nan)
    
    if 'step_resumed_from' in current_run.config:
        step_resumed_from = current_run.config['step_resumed_from']
        post_resume_step = raw_df[900:1100]['_timestamp'].diff().idxmax()
        current_df = raw_df.drop(range(step_resumed_from + 1, post_resume_step + 1), axis=0, inplace=False)
        current_df.reset_index(inplace=True)
        
    else:
        current_df = raw_df 
    
    for new_task in range(2, 11):
        forgetting_start = current_df[f'Test Accuracy, Query #{new_task}'].first_valid_index() - 1
        
        if new_task == 10:
            forgetting_end = len(current_df) - 1
        else:
            forgetting_end = current_df[f'Test Accuracy, Query #{new_task + 1}'].first_valid_index() - 1 
            
        forgetting_len = min(forgetting_end - forgetting_start, max_samples_per_curve)
    
        for forgetting_task in range(1, new_task):
            number_times_learned = new_task - forgetting_task
            number_total_tasks = new_task
            trajectory = np.array(current_df[f'Test Accuracy, Query #{forgetting_task}'][forgetting_start:forgetting_start + forgetting_len], dtype=np.float64)

            if np.any(np.isnan(trajectory)):
                print(f'Found nans for new task {new_task} and forgetting task {forgetting_task}')
                print(forgetting_start, forgetting_end)
                print(trajectory)
            
            forgetting_trjectories[number_times_learned - 1, number_total_tasks - 1, :forgetting_len] = trajectory
                
    
    return forgetting_trjectories


def parse_total_task_training_curve(current_run_id=None, current_run=None, samples=MAX_HISTORY_SAMPLES):
    if current_run_id is None and current_run is None:
        print('Must provide either a current run or its id')
        return
    
    if current_run is None:
        current_run = API.run(f'meta-learning-scaling/sequential-benchmark-baseline/{current_run_id}')
        
    current_df = current_run.history(pandas=True, samples=samples)
    total_task_curve = defaultdict(dict)
    task_total_trials = defaultdict(lambda: 0)
    
    first_row_blank = int(np.isnan(current_df['Test Accuracy'][0]))
    first_task_finished = current_df['Test Accuracy, Query #2'].first_valid_index() - first_row_blank
    episode_trials = examples_per_epoch(1, 1)
    
    for i, acc in enumerate(current_df['Test Accuracy, Query #1'][first_row_blank:first_task_finished]):
        total_task_curve[1][(i + 1) * episode_trials] = acc
        
    task_total_trials[1] = (first_task_finished - first_row_blank) * episode_trials
        
    for current_task in range(2, 11):
        current_task_start = current_df[f'Test Accuracy, Query #{current_task}'].first_valid_index()

        if current_task == 10:
            current_task_end = current_df.shape[0]
        else:
            current_task_end = current_df[f'Test Accuracy, Query #{current_task + 1}'].first_valid_index()
            
        current_task_length = current_task_end - current_task_start

        for task in range(1, current_task + 1):
            episode_trials = examples_per_epoch(task, current_task)
            
            for i, acc in enumerate(current_df[f'Test Accuracy, Query #{task}'][current_task_start:current_task_end]):
                total_task_curve[task][task_total_trials[task] + (i + 1) * episode_trials] = acc
                
            task_total_trials[task] = task_total_trials[task] + (current_task_length * episode_trials)
    
    # print([(t, len(total_task_curve[t].keys())) for t in range(1, 11)])
    
    return total_task_curve


def parse_simultaneous_training(current_run_id=None, current_run=None, samples=MAX_HISTORY_SAMPLES):
    if current_run_id is None and current_run is None:
        print('Must provide either a current run or its id')
        return
    
    if current_run is None:
        current_run = API.run(f'meta-learning-scaling/sequential-benchmark-baseline/{current_run_id}')
        
    current_df = current_run.history(pandas=True, samples=samples)
    simultaneous_task_accuracies = current_df['Test Per-Query Accuracy (list)']
    first_row_blank = int(np.isnan(simultaneous_task_accuracies[0]))
    return np.array(list(simultaneous_task_accuracies[first_row_blank:]))


def process_multiple_runs_simultaneous_training(runs, debug=False, ignore_runs=None, samples=MAX_HISTORY_SAMPLES):
    simultaneous_training_results = []
    
    for i, run in enumerate(runs):
        if i > 0 and i % 10 == 0:
            print(run.name, i)
        else:
            print(run.name)
        
        if ignore_runs is not None and run.name in ignore_runs:
            continue
        
        simultaneous_training_results.append(parse_simultaneous_training(current_run=run, samples=samples))
        
    max_length = max([res.shape[0] for res in simultaneous_training_results])
    for i in range(len(simultaneous_training_results)):
        curr_length = simultaneous_training_results[i].shape[0]
        simultaneous_training_results[i] = np.pad(simultaneous_training_results[i], ((0, max_length - curr_length), (0, 0)), 
                                                  'edge') #'constant', constant_values=np.nan)        
    
    stacked_results = np.stack(simultaneous_training_results, axis=2)      
    
    all_results_mean = np.nanmean(stacked_results, axis=(1, 2))
    all_results_std = np.nanstd(stacked_results, axis=(1, 2))
    all_results_sem = all_results_std / np.sqrt(np.prod(stacked_results.shape[1:]))
    
    per_task_results_mean = np.nanmean(stacked_results, axis=(2))
    per_task_results_std = np.nanstd(stacked_results, axis=(2))
    per_task_results_sem = per_task_results_std / np.sqrt(stacked_results.shape[2])
    
    return stacked_results, all_results_mean, all_results_std, all_results_sem, per_task_results_mean, per_task_results_std, per_task_results_sem
    

PRINT_HEADERS = ['###'] + [str(x) for x in range(1, 11)]


def pretty_print_results(results, **kwargs):
    result_rows = [[str(i + 1)] + list(results[i]) for i in range(len(results))]
    tab_args = dict(tablefmt='fancy_grid')
    tab_args.update(kwargs)
    print(tabulate.tabulate(result_rows, PRINT_HEADERS, **tab_args))


DEFAULT_PROJECT_PATH = 'meta-learning-scaling/sequential-benchmark-baseline'
    
    
def load_runs(max_rep_id, project_path=DEFAULT_PROJECT_PATH, split_runs_by_dimension=True,
              valid_run_ids=None, debug=False):
    runs = API.runs(project_path)
    
    results = ConditionAnalysesSet([], [], [], [])
    
    for run in runs:
        config = json.loads(run.json_config)
        run_id = config['dataset_random_seed']['value']
        
        if isinstance(run_id, dict):
            run_id = run_id['value']
        
        run_id = int(run_id)
        
        if debug: print(run.name, run_id)
        
        # run_name = run.description.split('\n')[0]
        # run_id = int(run_name[run_name.rfind('-') + 1:])
        
        if valid_run_ids is None or run_id in valid_run_ids:
            dimension = (run_id // 1000) - 1
            rep = run_id % 1000
            if rep < max_rep_id:
                # combined / all runs
                results[COMBINED_INDEX].append(run)
                # by-dimension
                if split_runs_by_dimension:
                        results[dimension].append(run)
            
    return results


def query_modulated_runs_by_dimension(max_rep_id):
    runs = API.runs('meta-learning-scaling/sequential-benchmark-task-modulated')
    
    results = {level: ConditionAnalysesSet([], [], [], []) for level in range(1, 5)}
    
    for run in runs:
        level, run_id = [int(x) for x in run.description.split('\n')[0][-6:].split('-')]
        dimension = (run_id // 1000) - 1
        rep = run_id % 1000
        if rep < max_rep_id:
            results[level][dimension].append(run)
            # combined / all runs
            results[level][3].append(run)
            
    return results


def process_multiple_runs(runs, debug=False, ignore_runs=None, samples=MAX_HISTORY_SAMPLES, parse_func=parse_run_results):
    examples = []
    log_examples = []
    abs_accuracies = []
    accuracy_drops = []
    first_task_accuracies = []
    new_task_accuracies = []
    
    examples_by_task = np.zeros((30, 10))
    counts_by_task = np.zeros((30, 10))
    
    for i, run in enumerate(runs):
        if i > 0 and i % 10 == 0:
            print(run.name, i)
        else:
            print(run.name)
        
        if ignore_runs is not None and run.name in ignore_runs:
            continue
        
        examples_to_criterion, absolute_accuracy, accuracy_drop, first_task_acc, new_task_acc = parse_func(current_run=run, samples=samples)
        # print(examples_to_criterion)
        # print(np.log(examples_to_criterion))
        examples.append(examples_to_criterion)
        log_examples.append(np.log(examples_to_criterion))
        abs_accuracies.append(absolute_accuracy)
        accuracy_drops.append(accuracy_drop)
        first_task_accuracies.append(first_task_acc)
        new_task_accuracies.append(new_task_acc)
        
        for index, task in enumerate(run.config['query_order']):
            task_examples = np.diag(examples_to_criterion, index)
            examples_by_task[task,:10 - index] += task_examples
            counts_by_task[task,:10 - index] += 1
            
    # Removing all extraneous nans
    print('Removing extraneous nans')
    first_task_accuracies = np.array(first_task_accuracies)
    new_task_accuracies = np.array(new_task_accuracies)
    
    max_first_nan_idx = np.max(np.argmax(np.isnan(first_task_accuracies), axis=2))
    print('Max first nan index:', max_first_nan_idx)
    
    first_task_accuracies = first_task_accuracies[:, :, :max_first_nan_idx]
    new_task_accuracies = new_task_accuracies[:, :, :max_first_nan_idx]
    accuracy_counts = np.count_nonzero(~np.isnan(new_task_accuracies), axis=0)

    output = {}
    for result_set, name, field in zip((examples, log_examples, abs_accuracies, accuracy_drops, 
                                        first_task_accuracies, new_task_accuracies), 
                                       RESULT_SET_NAMES, 
                                       ANALYSIS_SET_FIELDS):
        print(name, field)
        output[field] = ResultSet(name=name, 
                                  mean=np.nanmean(result_set, axis=0), 
                                  std=np.nanstd(result_set, axis=0))

        
    # eliminate zeros to allow dividing later
    accuracy_counts[accuracy_counts == 0] = 1
    output['accuracy_counts'] = accuracy_counts
        
    # to avoid division by zero
    counts_by_task[counts_by_task == 0] = 1
    average_examples_by_task = np.divide(examples_by_task, counts_by_task)
    output['examples_by_task'] = average_examples_by_task
    
    analysis = AnalysisSet(**output)
        
    if debug:
        return analysis, examples
    
    return analysis

def process_multiple_runs_total_task_training_curves(runs, debug=False, ignore_runs=None, samples=MAX_HISTORY_SAMPLES):
    aggregate_total_task_curve = defaultdict(lambda: defaultdict(list))
    
    for i, run in enumerate(runs):
        if i > 0 and i % 10 == 0:
            print(run.name, i)
        else:
            print(run.name)
        
        if ignore_runs is not None and run.name in ignore_runs:
            continue
        
        total_task_curve = parse_total_task_training_curve(current_run=run, samples=samples)
        for task in total_task_curve:
            for num_trials, acc in total_task_curve[task].items():
                aggregate_total_task_curve[task][num_trials].append(acc)
            
            
    aggregate_total_task_curve_raw = defaultdict(OrderedDict)
    aggregate_total_task_curve_mean = defaultdict(OrderedDict)
    aggregate_total_task_curve_std = defaultdict(OrderedDict)
    aggregate_total_task_curve_sem = defaultdict(OrderedDict)
    
    for task in range(1, 11):
        for num_trials in sorted(aggregate_total_task_curve[task]):
            accs = aggregate_total_task_curve[task][num_trials]
            
            aggregate_total_task_curve_raw[task][num_trials] = accs
            aggregate_total_task_curve_mean[task][num_trials] = np.nanmean(accs)
            
            std = np.nanstd(accs)
            if np.isnan(std):
                print(f'For accs {accs}, std is nan')
            
            aggregate_total_task_curve_std[task][num_trials] = std
            aggregate_total_task_curve_sem[task][num_trials] = std / (len(accs) ** 0.5)
            
    return aggregate_total_task_curve_raw, aggregate_total_task_curve_mean, aggregate_total_task_curve_std, aggregate_total_task_curve_sem


def sign_test(values):
    n = len(values)
    row_faster_results = np.empty((n, n))
    row_faster_results.fill(np.nan)
    
    col_faster_results = np.empty((n, n))
    col_faster_results.fill(np.nan)
    
    wilcoxon_statistics = np.empty((n, n))
    wilcoxon_statistics.fill(np.nan)
    
    wilcoxon_p_values = np.empty((n, n))
    wilcoxon_p_values.fill(np.nan)
    
    for row in range(n):
        for col in range(row + 1, n):
            row_means = values[row].mean
            row_means = row_means[~np.isnan(row_means)]
            col_means = values[col].mean
            col_means = col_means[~np.isnan(col_means)]
    
            # Run both explicit comparisons, to account for the possibility of ties
            row_faster = np.sum(row_means <col_means )
            row_faster_results[row, col] = row_faster
            
            col_faster = np.sum(row_means > col_means)
            col_faster_results[row, col] = col_faster
            
            s, p = wilcoxon(row_means, col_means)
            wilcoxon_statistics[row, col] = s
            wilcoxon_p_values[row, col] = p
            
    return row_faster_results, col_faster_results, wilcoxon_statistics, wilcoxon_p_values


MODULATION_LEVELS = ['\\thead[cl]{None}'] + [f'\\thead[cl]{{ {i} }}' for i in range(1, 5)] 
SIGN_TEST_PRINT_HEADERS = ['\\thead[cl]{Modulation level}'] + [f'\\thead[cl]{{ {i} }}' for i in range(1, 5)] 


def pretty_print_sign_test_results(row_faster_results, col_faster_results, 
                                   wilcoxon_statistics=None, wilcoxon_p_values=None,
                                   higher_better=False, **kwargs):
    n = row_faster_results.shape[0]
    print_results = [[''] * n for _ in range(n)]
    if wilcoxon_statistics is not None:
        wilcoxon_results = [[''] * n for _ in range(n)]
    
    for row in range(n):
        for col in range(row + 1, n):
            row_val = int(row_faster_results[row, col])
            col_val = int(col_faster_results[row, col])
            if higher_better:
                row_val, col_val = col_val, row_val
                
            result = col_val # max(col_val, row_val)
            n_binom = col_val + row_val
            p = binom_test(max(col_val, row_val), n_binom)
            
            print_results[row][col] = f'\\makecell[cl]{{ {result} $(n={n_binom})$ \\\\ $p={p:.4f}{(p < 0.05 and p > 0.01) and "^{*}" or ""}{p < 0.01 and "^{**}" or ""}$}}'
            
            if wilcoxon_statistics is not None:
                wilcoxon_results[row][col] = f'\n{wilcoxon_statistics[row, col]:.4f}, {wilcoxon_p_values[row, col]:.4f}'
            
    tab_args = dict(tablefmt='fancy_grid')
    tab_args.update(kwargs)
            
    print_results = [[MODULATION_LEVELS[i]] + row[1:] for i, row in enumerate(print_results)]
    print(tabulate.tabulate(print_results[:-1], SIGN_TEST_PRINT_HEADERS, **tab_args))
    
    if wilcoxon_statistics is not None:
        wilcoxon_results = [[MODULATION_LEVELS[i]] + row[1:] for i, row in enumerate(wilcoxon_results)]
        print(tabulate.tabulate(wilcoxon_results[:-1], SIGN_TEST_PRINT_HEADERS, **tab_args))

        
def sign_test_with_sem(values, sample_sizes):
    n = len(values)
    row_faster_results = np.empty((n, n))
    row_faster_results.fill(np.nan)
    
    col_faster_results = np.empty((n, n))
    col_faster_results.fill(np.nan)
    
    wilcoxon_statistics = np.empty((n, n))
    wilcoxon_statistics.fill(np.nan)
    
    wilcoxon_p_values = np.empty((n, n))
    wilcoxon_p_values.fill(np.nan)
    
    for row in range(n):
        for col in range(row, n):
            row_means = values[row].mean
            row_means = row_means[~np.isnan(row_means)]
            row_sems = values[row].std / (sample_sizes[row] ** 0.5)
            row_sems = row_sems[~np.isnan(row_sems)]
            
            col_means = values[col].mean
            col_means = col_means[~np.isnan(col_means)]
            col_sems = values[col].std / (sample_sizes[col] ** 0.5)
            col_sems = col_sems[~np.isnan(col_sems)]
    
            # Run both explicit comparisons, to account for the possibility of ties
            row_faster_results[row, col] = np.sum(row_means + row_sems < col_means - col_sems)
            col_faster_results[row, col] = np.sum(row_means - row_sems > col_means + col_sems)
            
            signs = row_means < col_means
            signs[signs == 0] = -1
            
            shifted_row_means = row_means + np.multiply(row_sems, signs)
            shifted_col_means = col_means + np.multiply(col_sems, -1 * signs)
            s, p = wilcoxon(shifted_row_means, shifted_col_means)
            wilcoxon_statistics[row, col] = s
            wilcoxon_p_values[row, col] = p
            
    return row_faster_results, col_faster_results, wilcoxon_statistics, wilcoxon_p_values
