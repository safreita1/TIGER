import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.getcwd() + '/../../')

from graph_tiger.graphs import graph_loader
from graph_tiger.defenses import Defense


def plot_results(graph, steps, results, title):
    plt.figure(figsize=(6.4, 4.8))

    for method, result in results.items():
        result = [r / len(graph) for r in result]
        plt.plot(list(range(steps)), result, label=method)

    plt.ylim(0, 1)
    plt.ylabel('LCC')
    plt.xlabel('N_rm / N')
    plt.title(title)
    plt.legend()

    save_dir = os.getcwd() + '/plots/'
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_dir + title + '.pdf')
    plt.show()
    plt.clf()


def main():
    graph = graph_loader(graph_type='ky2', seed=1)

    params = {
        'runs': 10,
        'steps': 30,
        'seed': 1,

        'k_a': 30,
        'attack': 'rb_node',
        'attack_approx': int(0.1*len(graph)),

        'defense': 'rewire_edge_preferential',

        'plot_transition': False,
        'gif_animation': False,

        'edge_style': None,
        'node_style': None,
        'fa_iter': 20
    }

    edge_defenses = ['rewire_edge_random', 'rewire_edge_random_neighbor', 'rewire_edge_preferential_random', 'add_edge_random', 'add_edge_preferential']

    print("Running edge defenses")
    results = defaultdict(str)
    for defense in edge_defenses:
        params['defense'] = defense

        a = Defense(graph, **params)
        results[defense] = a.run_simulation()
    plot_results(graph, params['steps'], results, title='water:edge_defense_runs={},attack={},'.format(params['runs'], params['attack']))


if __name__ == '__main__':
    main()
