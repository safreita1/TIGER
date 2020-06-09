import numpy as np

from graphs import karate
from defenses import run_defense_method, get_defense_methods, get_defense_category


def test_defense_strength():
    """
    check that valid nodes are returned
    :return:
    """
    graph = karate()

    methods = get_defense_methods()
    strength = list(range(1, 20))

    for method in methods:
        if get_defense_category(method) == 'node':
            for k in strength:
                nodes = run_defense_method(graph, method=method, k=k)

                # print(method, k, nodes)
                assert len(nodes) == k


def test_method_selection():
    """
    check that valid nodes are returned
    :return:
    """

    ground_truth = {  # karate graph top 4 nodes to be monitored or top 4 edges to added/rewired
        'ns_node': [33, 0, 2, 32],
        'pr_node': [33, 0, 32, 2],
        'eig_node': [33, 0, 2, 32],
        'id_node': [33, 0, 32, 2],
        'ib_node': [0, 33, 32, 2],
        'rnd_node': [14, 19, 3, 27],

        'add_edge_pr': {
            'added': [(33, 0), (0, 32), (33, 2), (33, 1)]
        },

        'add_edge_eig': {
            'added': [(33, 0), (33, 2), (0, 32), (33, 1)]
        },

        'add_edge_deg': {
            'added': [(33, 0), (0, 32), (33, 2), (33, 1)]
        },

        'add_edge_random': {
            'added': [(14, 19), (16, 22), (29, 20), (31, 15)]
        },

        'add_edge_preferential': {
            'added': [(11, 9), (12, 14), (15, 16), (17, 18)]
        },

        'rewire_edge_random': {
            'added': [(21, 26), (30, 31), (18, 26), (17, 29)],
            'removed': [(27, 33), (32, 33), (9, 33), (2, 9)]
        },

        'rewire_edge_random_neighbor': {
            'added': [(16, 22), (9, 12), (3, 23), (16, 30)],
            'removed':  [(14, 33), (19, 1), (3, 12), (27, 2)]

        },

        'rewire_edge_preferential': {
            'added': [(18, 12), (10, 9), (28, 5), (1, 16)],
            'removed': [(33, 18), (0, 10), (33, 28), (0, 1)]

        },

        'rewire_edge_preferential_random': {
            'added': [(27, 19), (32, 8), (9, 32), (9, 10)],
            'removed': [(27, 33), (32, 33), (9, 33), (2, 9)]
        }
    }

    graph = karate()

    k = 4
    methods = get_defense_methods()

    for method in methods:
        values = run_defense_method(graph, method=method, k=k, seed=1)

        # print(method, values)
        if get_defense_category(method) == 'node':
            assert values == ground_truth[method]
        else:
            assert np.array_equal(values['added'], ground_truth[method]['added'])
            if 'removed' in values:
                assert np.array_equal(values['removed'], ground_truth[method]['removed'])


def main():
    test_defense_strength()
    test_method_selection()


if __name__ == '__main__':
    main()
