import os
import math
import time
import cycler
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import defaultdict
from joblib import Parallel, delayed

import sys
sys.path.insert(0, os.getcwd() + '/../../')

from graph_tiger.graphs import graph_loader
from graph_tiger.measures import run_measure, get_measures


measure_style = {
    # graph measures
    'binary_connectivity': ':',
    'node_connectivity': ':',
    'edge_connectivity': ':',
    'diameter': ':',
    'average_distance': ':',
    'average_inverse_distance': ':',
    'average_vertex_betweenness': ':',
    'average_edge_betweenness': ':',
    'average_clustering_coefficient': ':',
    'largest_connected_component': ':',

    'spectral_radius': '--',
    'spectral_gap': '--',
    'natural_connectivity': '--',
    'spectral_scaling': '--',
    'generalized_robustness_index': '--',

    'algebraic_connectivity': '-',
    'number_spanning_trees': '-',
    'effective_resistance': '-',

    'average_vertex_betweenness_approx': '-.',
    'average_edge_betweenness_approx': '-.',
    'natural_connectivity_approx': '-.',
    'number_spanning_trees_approx': '-.',
    'effective_resistance_approx': '-.',
}

graph_approx = ['average_vertex_betweenness_approx', 'average_edge_betweenness_approx']
spectral_approx = ['natural_connectivity_approx', 'number_spanning_trees_approx', 'effective_resistance_approx']

not_parallelized = ['spectral_radius', 'spectral_gap', 'natural_connectivity', 'spectral scaling',
                    'generalized_robustness_index', 'algebraic_connectivity', 'num_spanning_trees',
                    'effective_resistance']


def measure_time(graph, measure, timeout):
    start = time.time()

    if measure in spectral_approx:
        k = 30
        measure = measure.replace('_approx', '')
    elif measure in graph_approx:
        k = int(0.1 * len(graph))
        measure = measure.replace('_approx', '')
    else:
        k = np.inf

    result = run_measure(graph, measure, k, timeout=timeout)

    end = time.time()
    run_time = math.ceil(end - start)

    if result is None:
        run_time = None

    return len(graph), run_time


def parallelize_evaluation(graphs, measure, timeout):
    if measure in not_parallelized:
        n_jobs = 1
    else:
        n_jobs = len(graphs)

    data = Parallel(n_jobs=n_jobs)(
        delayed(measure_time)(graph, measure, timeout)
        for graph in graphs)

    sorted(data, key=itemgetter(0))
    run_times = [d[1] for d in data]

    return run_times


def test_scalability(timeout=10):
    results = defaultdict(list)

    nodes = [100, 1000, 10000, 100000, 1000000]
    graphs = [graph_loader(graph_type='CSF', n=n, seed=1) for n in nodes]

    measures = get_measures()

    measures = measures + graph_approx + spectral_approx

    for measure in tqdm(measures):
        run_times = parallelize_evaluation(graphs, measure, timeout)
        results[measure] = run_times

    color = plt.cm.gist_ncar(np.linspace(0, 0.9, len(measures)))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    mpl.rcParams['legend.fontsize'] = 5

    fig, axes = plt.subplots(ncols=5, figsize=(5*6 - 1, 5))
    for (measure, run_time) in results.items():

        axes[0].plot(nodes, run_time, label=measure, linewidth=1, linestyle=measure_style[measure])
        axes[0].set_xlabel('Number of nodes')
        axes[0].set_ylabel('Time in seconds')

        axes[1].plot(nodes, run_time, label=measure, linewidth=1, linestyle=measure_style[measure])
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Number of nodes (log-scale)')
        axes[1].set_ylabel('Time in seconds')

        axes[2].plot(nodes, run_time, label=measure, linewidth=1, linestyle=measure_style[measure])
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].set_xlabel('Number of nodes (log-scale)')
        axes[2].set_ylabel('Time in seconds (log-scale)')

    axes[0].legend()
    plt.title('Clustered Scale Free Graph')

    save_dir = os.getcwd() + '/plots/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + 'scalability.pdf')


if __name__ == '__main__':
    test_scalability(timeout=60*30)
