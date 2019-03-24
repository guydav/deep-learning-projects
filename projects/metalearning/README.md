# Meta-learning project codebase

The main codebase for Guy Davidson and Michael C. Mozer's research project on meta-learning scaling, which is also Guy Davidson's Capstone project for his undergraduate degree at the Minerva schools at KGI. 

This codebase relies on the dataset generation code at https://github.com/guydav/clevr-dataset-gen 

This code implements the models investigated in the Capstone paper, the baseline (simultaneous, heterogeneous dimensions) training condition, and the sequential benchmark in both the homogeneous dimension condition and the heterogeneous dimensions control. 

## Core contents

* `base_model.py`: The baseline model all models are implemented above, which implements universall useful test and training loop functionality, and supports a number of different loss functions and accuracy metrics. Also houses the main training loop called on a model (`train()`). 
* `benchmarks.py`: Implements the equivalent of the main training loop, but for the squential benchmark.
* `cnnmlp.py`: Implements the convolutional neural network models used in this project:
	* `CNNMLP`: A highly simple base model, including a convolutional module followed by a fully-connected module.
	* `PoolingDropoutCNNMLP`: a more complicated version of the above model, supporting different regularization techniques (dropout, spatial dropout, weight decay, learning rate scheduler).
	* `QueryModulatingCNNMLP`: the version of the model performing query-based modulation at one of the convolutional layers.
* `dataset.py`: Implements the various different PyTorch datasets used. Two dataset classes used all queries, and are used for the simultaneous condition, differing only by whether they read the ground-truth answers from the dataset file (`MetaLearningH5Dataset`) or compute it from the saved scene descriptions (`MetaLearningH5DatasetFromDescription`). The third dataset class,`SequentialBenchmarkMetaLearningDataset`, is used for the sequential benchmarks, and introduces the tasks according to the logic described in the paper.

## Additional scripts

In addition to the core contents dsecribed above, a few scripts exist to help run the different benchmarks. All utilize argpase to read arguments from the command line, and will therefore provide a help message if run with no arguments.

* `run_sequential_benchmark.py`: Runs the baseline condition sequential (homogeneous dimension) benchmark, utilizing the Latin square design as described in the paper.
* `run_control_sequential_benchmark.py`: Runs the control condition sequential benchmark (heterogeneous dimensions), iterating through different permutations of the dimensions as described in the paper.
* `run_query_modulated_sequential_benchmark.py`: Runs the sequential benchmark on the query-modulated models as described in the paper.

To run these scripts, we use a few shell scripts, since PyTorch does not release all GPU memory (or we are leaking some of it) until the Python process closes down. See below for examples of how to run them. Broadly, the all expect to receive `<GPU id> <start index> <end index, exclusive> <name> <seed>`, and pass additional arguments to the underlying Python script.

```
./batch_sequential_benchmark_latin_square.sh 0 0 10 'Baseline' 2030 --wandb_project "sequential-benchmark-baseline" --benchmark_dimension 1

./batch_query_modulated_sequential_benchmark_latin_square.sh 0 0 10 'Query-modulated' 2000 --wandb_project "sequential-benchmark-task-modulated" --benchmark_dimension 1 --modulation_level 4

./batch_control_sequential_benchmark.sh 0 0 6 'Baseline-control' 110
```
