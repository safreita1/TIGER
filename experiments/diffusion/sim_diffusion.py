import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
sys.path.insert(0, os.getcwd() + '/../../')

from graph_tiger.graphs import as_733
from graph_tiger.diffusion import Diffusion


def plot_results(graph, params, results):
    plt.figure(figsize=(6.4, 4.8))

    title = '{}_epidemic:diffusion={},method={},k={}'.format(params['model'], params['diffusion'],
                                                                          params['method'], params['k'])

    for strength, result in results.items():
        result_norm = [r / len(graph) for r in result]
        plt.plot(result_norm, label="Effective strength: {}".format(strength))

    plt.xlabel('Steps')
    plt.ylabel('Infected Nodes')
    plt.legend()
    plt.yscale('log')
    plt.ylim(0.001, 1)
    plt.title(title)

    save_dir = os.getcwd() + '/plots/' + title + '/'
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_dir + title + '.pdf')
    plt.show()
    plt.clf()


def run_epidemic_experiment(params):
    graph = as_733().copy()
    results = defaultdict(list)

    b_list = np.arange(0, 0.005, 0.001)  # transmission probability
    for idx, b in enumerate(b_list):

        params['b'] = b

        if idx == 1:
            params['plot_transition'] = True
            params['gif_animation'] = True
            params['gif_snaps'] = True
        else:
            params['plot_transition'] = False
            params['gif_animation'] = False
            params['gif_snaps'] = False

        ds = Diffusion(graph, **params)

        result = ds.run_simulation()
        results[ds.get_effective_strength()] = result

    plot_results(graph, params, results)


def main():
    # baseline
    sis_params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 1,
        'steps': 5000,

        'robust_measure': 'largest_connected_component',

        'k': 15,
        'diffusion': None,
        'method': None,

        'plot_transition': False,
        'gif_animation': False,
        'seed': 1,

        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 20
    }
    run_epidemic_experiment(sis_params)

    # increase diffusion
    sis_params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,

        'robust_measure': 'largest_connected_component',

        'k': 50,
        'diffusion': 'max',
        'method': 'add_edge_random',

        'plot_transition': False,
        'gif_animation': False,
        'seed': 1,

        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 20
    }
    # run_epidemic_experiment(sis_params)

    # decrease diffusion
    sis_params = {
        'model': 'SIS',
        'b': 0.00208,
        'd': 0.01,
        'c': 1,
        'runs': 10,
        'steps': 5000,

        'robust_measure': 'largest_connected_component',

        'k': 5,
        'diffusion': 'min',
        'method': 'ns_node',

        'plot_transition': False,
        'gif_animation': False,
        'seed': 1,

        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 20
    }
    run_epidemic_experiment(sis_params)


if __name__ == '__main__':
    main()
