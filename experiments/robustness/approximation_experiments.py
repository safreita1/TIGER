import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.getcwd() + '/../../')

from graph_tiger.graphs import graph_loader
from graph_tiger.measures import run_measure


def plot_results(x_data, results, result_type, measures, n, start, step):
    num_measures = int(len(measures) / 2)
    fig, axes = plt.subplots(ncols=num_measures, figsize=(num_measures*6 - 1, 5))

    for index, metric_name in enumerate(measures):
        if index == num_measures:
            break

        error = np.round(np.abs(results[:, num_measures + index] - results[:, index]), 2)
        axes[index].plot(x_data, error, label=metric_name)

        axes[index].set_title(metric_name)
        axes[index].set_xlabel('k')
        axes[index].set_ylabel('Error')

        if metric_name == 'number_spanning_trees':
            axes[index].set_yscale('log')

    plt.legend(loc="upper right")

    save_dir = os.getcwd() + '/plots/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + 'approximation_{}_n={},start={},step={}.pdf'.format(result_type, n, start, step))
    plt.show()


def run(graph, measures, k):
    result = []

    for measure in measures:
        if '_approx' in measure:
            measure = measure.replace('_approx', '')
            r = run_measure(graph=graph, measure=measure, k=k)
        else:
            r = run_measure(graph=graph, measure=measure)

        result.append(r)

    return result


def run_analysis(n, runs, k_start, k_step, measures):
    graphs = [graph_loader(graph_type='CSF', n=n, seed=s) for s in range(runs)]

    approx_results = []
    k_values = list(range(k_start, n, k_step)) + [np.inf]
    for k in tqdm(k_values):
        results = []
        for i in range(runs):
            r = run(graphs[i], measures, k=k)
            results.append(r)

        k_avg = np.mean(results, axis=0)
        approx_results.append(k_avg)

    return np.stack(approx_results)


def main():
    measures_graph = [
        'average_vertex_betweenness',
        'average_edge_betweenness',
        'average_vertex_betweenness_approx',
        'average_edge_betweenness_approx'
    ]

    measures_spectral = [
        'natural_connectivity',
        'number_spanning_trees',
        'effective_resistance',
        'natural_connectivity_approx',
        'number_spanning_trees_approx',
        'effective_resistance_approx',
    ]

    n = 300
    start = 5
    step = 10

    n_s = 300
    start_s = 5
    step_s = 10

    graph_results = run_analysis(n=n, runs=30, k_start=start, k_step=step, measures=measures_graph)

    x_data = list(range(start, n, step)) + [300]
    plot_results(x_data, graph_results, "graph", measures_graph, n, start, step)

    # spectral_results = run_analysis(n=n_s, runs=30, k_start=start_s, k_step=step_s, measures=measures_spectral)
    #
    # x_data_s = list(range(start_s, n_s, step_s)) + [300]
    # plot_results(x_data_s, spectral_results, "spectral", measures_spectral, n, start_s, step_s)


if __name__ == '__main__':
    main()