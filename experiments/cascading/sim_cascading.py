import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
sys.path.insert(0, os.getcwd() + '/../../')

from graph_tiger.graphs import electrical
from graph_tiger.cascading import Cascading


def plot_results(graph, params, results, xlabel='Steps', line_label='', experiment=''):
    plt.figure(figsize=(6.4, 4.8))

    title = '{}:step={},l={},r={},k_a={},attack={},k_d={},defense={}'.format(experiment, params['steps'], params['l'],
                                                                             params['r'], params['k_a'],  params['attack'],
                                                                             params['k_d'], params['defense'])
    for strength, result in results.items():

        result_norm = [r / len(graph) for r in result]
        plt.plot(result_norm, label="{}: {}".format(line_label, strength))

    plt.xlabel(xlabel)
    plt.ylabel(params['robust_measure'])
    plt.ylim(0, 1)

    save_dir = os.getcwd() + '/plots/' + experiment + '/'
    os.makedirs(save_dir, exist_ok=True)

    plt.legend()
    plt.title(title)
    plt.savefig(save_dir + title + '.pdf')
    plt.show()

    plt.clf()


def experiment_redundancy(graph):
    params = {
        'runs': 10,
        'steps': 100,
        'seed': 1,

        'l': 0.8,
        'r': 0.2,
        'c': int(0.1 * len(graph)),

        'k_a': 5,
        'attack': 'id_node',
        'attack_approx': None,  # int(0.1 * len(graph)),

        'k_d': 0,
        'defense': None,

        'robust_measure': 'largest_connected_component',

        'plot_transition': False,
        'gif_animation': False,

        'edge_style': None,
        'node_style': 'spectral',
        'fa_iter': 2000,

    }

    results = defaultdict(list)
    redundancy = np.arange(0, 0.5, .1)

    for idx, r in enumerate(redundancy):
        params['r'] = r

        if idx == 2:
            params['plot_transition'] = True
            params['gif_animation'] = True
            params['gif_snaps'] = True
        else:
            params['plot_transition'] = False
            params['gif_animation'] = False
            params['gif_snaps'] = False

        cf = Cascading(graph, **params)
        results[r] = cf.run_simulation()

    plot_results(graph, params, results, xlabel='Steps', line_label='Redundancy', experiment='redundancy')


def experiment_attack(graph):
    params = {
        'runs': 10,
        'steps': 100,
        'seed': 1,

        'l': 0.8,
        'r': 0.4,
        'c': int(0.1 * len(graph)),

        'k_a': 5,
        'attack': 'rnd_node',
        'attack_approx': None,  # int(0.1 * len(graph)),

        'k_d': 0,
        'defense': None,

        'robust_measure': 'largest_connected_component',

        'plot_transition': False,
        'gif_animation': False,

        'edge_style': None,
        'node_style': 'spectral',
        'fa_iter': 2000,

    }

    # rnd_node attack
    results = defaultdict(list)

    attack_strength = np.arange(2, 11, 2)

    for idx, k_a in enumerate(attack_strength):
        params['k_a'] = k_a

        if idx == 2:
            params['plot_transition'] = False
            params['gif_animation'] = False
        else:
            params['plot_transition'] = False
            params['gif_animation'] = False

        cf = Cascading(graph, **params)
        results[k_a] = cf.run_simulation()

    plot_results(graph, params, results, xlabel='Steps', line_label='k_a', experiment='rnd_node_attack')

    # targeted attack
    params['attack'] = 'id_node'

    results = defaultdict(list)
    for idx, k_a in enumerate(attack_strength):
        params['k_a'] = k_a

        if idx == 2:
            params['plot_transition'] = False
            params['gif_animation'] = False
        else:
            params['plot_transition'] = False
            params['gif_animation'] = False

        cf = Cascading(graph, **params)
        results[k_a] = cf.run_simulation()

    plot_results(graph, params, results, xlabel='Steps', line_label='k_a', experiment='id_node_attack')


def experiment_defense(graph):
    params = {
        'runs': 10,
        'steps': 100,
        'seed': 1,

        'l': 0.8,
        'r': 0.2,
        'c': int(0.1 * len(graph)),

        'k_a': 5,
        'attack': 'id_node',
        'attack_approx': None,  # int(0.1 * len(graph)),

        'k_d': 0,
        'defense': 'add_edge_preferential',

        'robust_measure': 'largest_connected_component',

        'plot_transition': False,
        'gif_animation': False,

        'edge_style': None,
        'node_style': 'spectral',
        'fa_iter': 2000,

    }

    # edge defense
    results = defaultdict(list)
    defense_strength = np.arange(10, 51, 10)

    for idx, k_d in enumerate(defense_strength):
        params['k_d'] = k_d

        if idx == 2:
            params['plot_transition'] = False
            params['gif_animation'] = False
        else:
            params['plot_transition'] = False
            params['gif_animation'] = False

        cf = Cascading(graph, **params)
        results[k_d] = cf.run_simulation()

    plot_results(graph, params, results, xlabel='Steps', line_label='k_d', experiment='add_edge_pref')

    # node defense
    params['defense'] = 'pr_node'
    defense_strength = np.arange(1, 10, 2)

    results = defaultdict(list)

    for idx, k_d in enumerate(defense_strength):
        params['k_d'] = k_d

        if idx == 2:
            params['plot_transition'] = False
            params['gif_animation'] = False
        else:
            params['plot_transition'] = False
            params['gif_animation'] = False

        cf = Cascading(graph, **params)
        results[k_d] = cf.run_simulation()

    plot_results(graph, params, results, xlabel='Steps', line_label='k_d', experiment='add_node_pr')


def main():
    graph = electrical().copy()

    experiment_redundancy(graph)
    # experiment_attack(graph)
    # experiment_defense(graph)


if __name__ == '__main__':
    main()
