import numpy as np

from graphs import o4_graph, p4_graph, c4_graph, k4_1_graph, k4_2_graph
from graphs import two_c4_0_bridge, two_c4_1_bridge, two_c4_2_bridge, two_c4_3_bridge

from measures import run_measure


def test_measures():

    measure_ground_truth = {  # graph order: o4, p4, c4, k4_1, c4_0, c4_1, c4_2, c4_3
        'node_connectivity': [0, 1, 2, 2, 3, 0, 1, 1, 1],
        'edge_connectivity': [0, 1, 2, 2, 3, 0, 1, 1, 1],
        'diameter': [None, 3, 2, 2, 1, None, 5, 5, 5],
        'average_distance': [None, 1.67, 1.33, 1.17, 1, None, 2.29, 2.29, 2.29],
        'average_inverse_distance': [0, 0.72, 0.83, 0.92, 1.0, 0.36, 0.58, 0.58, 0.58],
        'average_vertex_betweenness': [0, 4, 3.5, 3.25, 3, 3.5, 11.5, None, None],
        'average_edge_betweenness': [0, 3.33, 2.0, 1.4, 1, 2, 7.11, 7.11, 7.11],
        'average_clustering_coefficient': [0, 0, 0, 0.83, 1, 0, 0, None, None],
        'largest_connected_component': [1, 4, 4, 4, 4, 4, 8, 8, 8],

        'spectral_radius': [0, 1.62, 2, 2.56, 3, 2, 2.34, 2.9, 3.65],
        'spectral_gap': [0, 1, 2, 2.56, 4, 0, 0.53, 1.19, 2],
        'natural_connectivity': [0, 0.65, 0.87, 1.29, 1.67, 0.87, 0.97, 1.28, 1.81],
        'spectral_scaling': [None, 7.18, 7.28, 0.17, 0.09, None, None, 7.04, 6.93],
        'generalized_robustness_index': [None, 7.18, 7.28, 0.17, 0.09, None, None, 7.04, 6.93],

        'algebraic_connectivity': [0, 0.59, 2, 2, 4, 0, 0.29, 0.4, 0.45],
        'number_spanning_trees': [0, 1, 4, 8, 16, 0, 16, 32, 48],
        'effective_resistance': [np.inf, 10, 5, 4, 3, np.inf, 46, 38, 35.33]
    }

    graphs = [o4_graph(), p4_graph(), c4_graph(), k4_1_graph(), k4_2_graph(),
              two_c4_0_bridge(), two_c4_1_bridge(), two_c4_2_bridge(), two_c4_3_bridge()]

    for measure_name, graph_values in measure_ground_truth.items():
        for idx, graph in enumerate(graphs):

            value = run_measure(graph, measure_name)
            if value is not None: value = round(value, 2)

            # print(idx, measure_name, value)
            assert value, graph_values[idx]

