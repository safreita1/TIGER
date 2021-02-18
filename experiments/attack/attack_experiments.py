import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.getcwd() + '/../../')

from graph_tiger.graphs import graph_loader
from graph_tiger.attacks import Attack


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
        'runs': 1,
        'steps': 30,
        'seed': 1,

        'attack': 'rb_node',
        'attack_approx': int(0.1*len(graph)),

        'plot_transition': True,
        'gif_animation': True,
        'gif_snaps': True,

        'edge_style': None,
        'node_style': None,
        'fa_iter': 20
    }

    print("Creating example visualization")
    a = Attack(graph, **params)
    a.run_simulation()

    node_attacks = ['rnd_node', 'id_node', 'rd_node', 'ib_node', 'rb_node']
    edge_attacks = ['rnd_edge', 'id_edge', 'rd_edge', 'ib_edge', 'rb_edge']

    params['runs'] = 10
    params['steps'] = len(graph) - 1
    params['plot_transition'] = False
    params['gif_animation'] = False
    params['gif_snaps'] = False

    print("Running node attacks")
    results = defaultdict(str)
    for attack in node_attacks:
        params['attack'] = attack

        if 'rb' in attack or 'ib' in attack:
            params['attack_approx'] = int(0.1*len(graph))
        else:
            params['attack_approx'] = None

        a = Attack(graph, **params)
        results[attack] = a.run_simulation()
    plot_results(graph, params['steps'], results, title='water:node-attacks_runs={}'.format(params['runs']))

    print("Running edge attacks")
    results = defaultdict(str)
    for attack in edge_attacks:
        params['attack'] = attack

        if 'rb' in attack or 'ib' in attack:
            params['attack_approx'] = int(0.1*len(graph))
        else:
            params['attack_approx'] = None

        a = Attack(graph, **params)
        results[attack] = a.run_simulation()
    plot_results(graph, params['steps'], results, title='water:edge-attacks_runs={}'.format(params['runs']))


if __name__ == '__main__':
    main()
