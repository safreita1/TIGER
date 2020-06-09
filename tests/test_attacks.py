from graphs import karate
from attacks import run_attack_method, get_attack_methods


def test_attack_strength():
    """
    check that valid nodes are returned
    :return:
    """
    graph = karate()

    methods = get_attack_methods()
    strength = list(range(1, 20))

    for method in methods:
        for k in strength:
            nodes = run_attack_method(graph, method=method, k=k)

            assert len(nodes) == k


def test_method_selection():
    """
    check that valid nodes are returned
    :return:
    """

    ground_truth = {  # karate graph top 4 nodes to be removed
        'ns_node': [33, 0, 2, 32],
        'pr_node': [33, 0, 32, 2],
        'eig_node': [33, 0, 2, 32],
        'id_node': [33, 0, 32, 2],
        'rd_node': [33, 0, 32, 1],
        'ib_node': [0, 33, 32, 2],
        'rb_node': [0, 33, 32, 2],

        'ns_line_edge': [(32, 33), (8, 33), (31, 33), (13, 33)],
        'pr_line_edge': [(32, 33), (0, 2), (0, 1), (0, 31)],
        'eig_line_edge': [(32, 33), (8, 33), (31, 33), (13, 33)],
        'deg_line_edge': [(32, 33), (0, 2), (0, 1), (31, 33)],
        'id_edge': [(32, 33), (0, 2), (0, 1), (2, 32)],
        'rd_edge': [(32, 33), (0, 2), (0, 1), (2, 32)],
        'ib_edge': [(0, 31), (0, 6), (0, 5), (0, 2)],
        'rb_edge': [(0, 31), (0, 2), (0, 8), (13, 33)],
    }

    graph = karate()

    k = 4
    methods = get_attack_methods()

    for method in methods:
        values = run_attack_method(graph, method=method, k=k, seed=1)

        # print(method, values)
        if 'rnd' not in method:
            assert values == ground_truth[method]


def main():
    test_method_selection()
    test_attack_strength()


if __name__ == '__main__':
    main()
