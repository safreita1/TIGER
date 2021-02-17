import math
import stopit
import numpy as np
import networkx as nx

from utils import get_adjacency_spectrum, get_laplacian_spectrum


@stopit.threading_timeoutable()
def run_measure(graph, measure, k=np.inf):
    """
    Evaluates graph robustness according to a specified measure

    :param graph: undirected NetworkX graph to measure
    :param measure: string containing the robustness measure to evaluate
    :param k: an integer for fast approximation of certain robustness measures. small k = fast, large k = precise
    :param timeout: allows the user to stop running the measure after 'x' seconds.
    :return: a float representing the robustness of the graph, or None if it times out or an error occurs
    """
    try:
        if k is not np.inf:
            result = measures[measure](graph, k)
        else:
            result = measures[measure](graph)

        return result

    except stopit.TimeoutException:
        print('timed out', measure)
        return None

    except Exception as e:
        print('error', e, measure)
        return None


def get_measures():
    """
    Returns a list of strings representing all of the available graph robustness measures

    :return: list of strings
    """
    return list(measures.keys())


'''
Graph Connectivity Measures
'''


# def binary_connectivity(graph):
#     connected = 1
#
#     pairs = nx.algorithms.all_pairs_node_connectivity(graph)
#     for vertex, connections in pairs.items():
#         if list(connections.values()).count(0) > 0:
#             connected = 0
#             break
#
#     return connected


def node_connectivity(graph):
    """
    Larger vertex (node) connectivity -> harder to disconnect graph -> more robust graph.

    :param graph: undirected NetworkX graph
    :return: an integer
    """
    return nx.algorithms.node_connectivity(graph)


def edge_connectivity(graph):
    """
    Larger edge connectivity -> harder to disconnect graph -> more robust graph.

    :param graph: undirected NetworkX graph
    :return: an integer
    """
    return nx.algorithms.edge_connectivity(graph)


def avg_distance(graph):
    """
    The smaller the average shortest path distance, the more robust the graph.
    This can be viewed through the lens of network connectivity i.e.,  smaller avg. distance -> better connected graph
    Undefined for disconnected graphs.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    return round(nx.average_shortest_path_length(graph), 2)


def avg_inverse_distance(graph):
    """
    The larger the average inverse shortest path distance, the more robust the graph.
    This can be viewed through the lens of network connectivity i.e., larger avg. inverse distance -> better connected graph
    -> more robust graph.
    Resolves the issue of not working for disconnected graphs in the avg_distance() function.
    Undefined for disconnected graphs.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    return round(nx.global_efficiency(graph), 2)


def diameter(graph):
    """
    Measures the graph's diameter i.e., longest shortest path in the network.
    The smaller the diameter the more robust the graph i.e., smaller diameter -> better connected graph -> more robust graph

    :param graph: undirected NetworkX graph
    :return: an integer
    """
    return nx.diameter(graph)


def avg_vertex_betweenness(graph, k=np.inf):
    """
    The smaller the average vertex betweenness, the more robust the graph. We can view this as the
    load of the network being better distributed and less dependent on a few nodes.

    :param graph: undirected NetworkX graph
    :param k: the number of nodes used to approximate betweenness centrality (k=10% of nodes is usually good)
    :return: a float
    """

    node_centralities = nx.betweenness_centrality(graph, k=min(len(graph), k), normalized=False, endpoints=True)
    avg_betw = sum(list(node_centralities.values())) / len(node_centralities)

    return round(avg_betw, 2)


def avg_edge_betweenness(graph, k=np.inf):
    """
    The smaller the average edge betweenness, the more robust the graph. We can view this as the
    load of the network being better distributed and less dependent on a few edges.

    :param graph: undirected NetworkX graph
    :param k: the number of nodes used to approximate betweenness centrality (k=10% of nodes is usually good)
    :return: a float
    """

    edge_centralities = nx.edge_betweenness_centrality(graph, k=min(len(graph), k), normalized=False)

    if len(edge_centralities) > 0:
        avg_betweenness = sum(list(edge_centralities.values())) / len(edge_centralities)

        return round(avg_betweenness, 2)
    else:
        return 0


def average_clustering_coefficient(graph):
    """
    The larger the average global clustering coefficient, the more robust the graph i.e., more triangles ->
    better connected -> more robust graph

    :param graph: undirected NetworkX graph
    :return: a float
    """
    return round(nx.average_clustering(graph), 2)


def largest_connected_component(graph):
    """
    The larger the value, the more robust the graph.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lcc = sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    return len(graph.subgraph(lcc))


"""
Adjacency Matrix Spectral Measures
"""


def spectral_radius(graph):
    """
    The larger the spectral radius (largest eigenvalue of adjacency matrix), the more robust the graph. T
    his can be viewed from its close relationship to the `path' or 'loop' capacity in a network.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lam = get_adjacency_spectrum(graph, k=1, which='LA', eigvals_only=True)

    idx = lam.argsort()[::-1]  # sort descending algebraic
    lam = lam[idx]

    return round(lam[0], 2)


def spectral_gap(graph):
    """
     The larger the spectral gap (difference largest and second largest eigenvalue of adjacency matrix.),
     the more robust the graph. Has an advantage over spectral radius since it can take into account
     undesirable bridges in the network, lowering the graph's robustness.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lam = get_adjacency_spectrum(graph, k=2, which='LA', eigvals_only=True)

    idx = lam.argsort()[::-1]  # sort descending algebraic
    lam = lam[idx]

    return round(lam[0] - lam[1], 2)


def natural_connectivity(graph, k=np.inf):
    """
    The larger the natural connectivity (average eigenvalue of adjacency matrix), the more robust the graph.
    This can be seen through the relationship to the number of closed walks,
    which naturally captures the notion of network connectivity and alternative  paths in a network.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lam = get_adjacency_spectrum(graph, k=k, which='LA', eigvals_only=True)

    idx = lam.argsort()[::-1]  # sort descending algebraic
    lam = lam[idx]

    return round(math.log(sum(np.exp(lam.real)) / len(lam)), 2)


def odd_subgraph_centrality(i, lam, u):
    """
    Used in the calculation of spectral scaling.

    :param i: node index
    :param lam: largest eigenvalue
    :param u: largest eigenvector
    :return:
    """

    sc = 0
    for j in range(len(lam)):
        sc += np.power(u[i, j], 2) * np.sinh(lam[j])

    return sc


def spectral_scaling(graph, k=np.inf):
    """
    The smaller the value, the more robust the graph. Helps determine if a graph has many bridges (bad for robustness).

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lam, u = get_adjacency_spectrum(graph, k=k, which='LM', eigvals_only=False)

    idx = np.abs(lam).argsort()[::-1]  # sort descending magnitude
    lam = lam[idx]
    u = u[:, idx]

    u[:, 0] = np.abs(u[:, 0])  # first eigenvector should be positive

    sc = 0
    for i in range(len(graph)):
        sc += np.power(np.log10(u[:, 0][i]) - (np.log10(np.power(np.sinh(lam[0]), -0.5)) + 0.5 * np.log10(odd_subgraph_centrality(i, lam, u))), 2)

    sc = np.sqrt(sc / len(graph))
    if np.isnan(sc) or np.isinf(sc): sc = None

    return sc


def generalized_robustness_index(graph, k=30):
    """
    This can be considered a fast approximation of spectral scaling. The smaller the value, the more robust the graph.
    Also helps determine if a graph has many bridges (bad for robustness).

    :param graph: undirected NetworkX graph
    :return: a float
    """
    return spectral_scaling(graph, k=k)


'''
Laplacian Spectral Measures
'''


def algebraic_connectivity(graph):
    """
    The larger the algebraic connectivity, the more robust the graph.
    This is due to it's close connection to edge connectivity, where it serves as a lower bound:
    0 < u_2 < node connectivity < edge connectivity. This means that a network with larger algebraic connectivity
    is harder to disconnect.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lam = get_laplacian_spectrum(graph, k=2)

    alg_connect = round(lam[1], 2)

    return alg_connect


def num_spanning_trees(graph, k=np.inf):
    """
    The larger the number of spanning trees, the more robust the graph.
    This can be viewed from the perspective of network connectivity, where
    a larger set of spanning trees means more alternative pathways in the network.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lam = get_laplacian_spectrum(graph, k=k)

    num_trees = round(np.prod(lam[1:]) / len(graph), 2)

    return num_trees


def effective_resistance(graph, k=np.inf):
    """
    The smaller the effective resistance, the more robust the graph.

    :param graph: undirected NetworkX graph
    :return: a float
    """
    lam = get_laplacian_spectrum(graph, k=k)

    resistance = round(len(graph) * np.sum(1.0 / lam[1:]), 2)

    if resistance < 0: resistance = np.inf

    return resistance


measures = {
    # graph connectivity
    'node_connectivity': node_connectivity,
    'edge_connectivity': edge_connectivity,
    'diameter': diameter,
    'average_distance': avg_distance,
    'average_inverse_distance': avg_inverse_distance,
    'average_vertex_betweenness': avg_vertex_betweenness,
    'average_edge_betweenness': avg_edge_betweenness,
    'average_clustering_coefficient': average_clustering_coefficient,
    'largest_connected_component': largest_connected_component,

    # adjacency matrix spectrum
    'spectral_radius': spectral_radius,
    'spectral_gap': spectral_gap,
    'natural_connectivity': natural_connectivity,
    'spectral_scaling': spectral_scaling,
    'generalized_robustness_index': generalized_robustness_index,

    # laplacian matrix spectrum
    'algebraic_connectivity': algebraic_connectivity,
    'number_spanning_trees': num_spanning_trees,
    'effective_resistance': effective_resistance
}
