import os
import json
import urllib.request
import networkx as nx


graph_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/datasets/'
os.makedirs(graph_dir, exist_ok=True)


def graph_loader(graph_type, **kwargs):
    """
    Loads any of the available graph models, supported user-downloaded datasets and toy graphs.
    In order to get a list of available graph options run 'get_graph_options()'.

    :param graph_type: a string representing the graph you want to load. For example, 'ER', 'WS', 'BA',
           'oregon_1' (must first download), 'electrical' (must first download)
    :param kwargs: allows user to specify specific graph model properties
    :return: an undirected NetworkX graph
    """
    if graph_type in models.keys():
        graph = models[graph_type](**kwargs)

    elif graph_type in datasets:
        download_dataset(graph_type)
        graph = datasets[graph_type]()

    elif graph_type in custom.keys():
        graph = custom[graph_type]()

    else:
        print("Graph not supported. Select from one of the following graphs: {}".format(get_graph_options()))
        graph = None

    return graph


def download_dataset(dataset):
    """
    Reading the dataset from the web.

    :param dataset: a string representing the dataset to download
    """
    url_path = graph_urls[dataset][0]
    local_path = graph_dir + '{}.txt'.format(dataset)

    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url_path, local_path)


def get_graph_urls():
    """
    Returns a dictionary of the datasets used in TIGER and the original link to download them

    :return: dictionary containing links to each dataset
    """
    return graph_urls


def get_graph_options():
    """
    Returns a formatted string containing all of the generators, datasets and custom graphs implemented in TIGER

    :return: formatted string
    """

    graph_options = {
        'models': list(models.keys()),
        'datasets': datasets,
        'custom': list(custom.keys())
    }

    return json.dumps(graph_options, indent=1)


"""
Graph Models
"""


def erdos_reyni(n, p=None, seed=None):
    """
    Returns a Erdos Reyni NetworkX graph

    :param n: number of nodes
    :param p: probability for edge creation
    :param seed: fixes the graph generation process
    :return: a NetworkX graph
    """

    if p is None: p = (1.0 / n + 0.1)
    return nx.generators.erdos_renyi_graph(n=n, p=p, seed=seed)


def watts_strogatz(n, m=4, p=0.05, seed=None):
    """
    Returns a Watts Strogatz NetworkX graph

    :param n: number of nodes
    :param m: each node is joined with its k nearest neighbors in a ring topology
    :param p: probability of rewiring each edge
    :param seed: fixes the graph generation process
    :return: a NetworkX graph
    """

    return nx.generators.connected_watts_strogatz_graph(n=n, k=m, p=p, seed=seed)


def barbasi_albert(n, m=3, seed=None):
    """
    Returns a Barabasi Albert NetworkX graph

    :param n: number of nodes
    :param m: number of edges to attach from a new node to existing nodes
    :param seed: fixes the graph generation process
    :return: a NetworkX graph
    """

    return nx.generators.barabasi_albert_graph(n=n, m=m, seed=seed)


def clustered_scale_free(n, m=3, p=0.3, seed=None):
    """
    Returns a Clustered Scale-Free NetworkX graph

    :param n: number of nodes
    :param m: the number of random edges to add for each new node
    :param p:  probability of adding a triangle after adding a random edge
    :param seed: fixes the graph generation process
    :return: a NetworkX graph
    """

    return nx.powerlaw_cluster_graph(n=n, m=m, p=p, seed=seed)


"""
Specific Graphs
"""


def wdn_ky2():
    """
    Returns the graph from: https://uknowledge.uky.edu/wdst/4/,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.Graph()

    with open(graph_dir + 'ky2.txt') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split('\t')) == 9:
                u, v = line.strip().split('\t')[1:3]
                u = u.strip()
                v = v.strip()
                if 'J' in u and 'J' in v:
                    graph.add_edge(u, v)
            else:
                name, x_pos, y_pos = line.strip().split('\t')
                name = name.strip()
                x_pos = float(x_pos.strip())
                y_pos = float(y_pos.strip())
                graph.nodes[name]['pos'] = [x_pos, y_pos]

    graph = nx.convert_node_labels_to_integers(graph)
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def as_733():
    """
    Returns the 'as19971108' graph from: http://snap.stanford.edu/data/as-733.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "as19971108.txt")
    graph = nx.convert_node_labels_to_integers(graph)
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def p2p_gnuetella08():
    """
    Returns the graph from: https://snap.stanford.edu/data/p2p-Gnutella08.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "p2p-Gnutella08.txt")
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def ca_grqc():
    """
    Returns the graph from: https://snap.stanford.edu/data/ca-GrQc.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "ca-GrQc.txt")
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def cit_hep_th():
    """
    Returns the graph from: https://snap.stanford.edu/data/cit-HepTh.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "cit-HepTh.txt")  # , create_using=nx.DiGraph()
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def wiki_vote():
    """
    Returns the graph from: https://snap.stanford.edu/data/wiki-Vote.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "wiki-Vote.txt")  # , create_using=nx.DiGraph()
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def email_eu_all():
    """
    Returns the graph from: https://snap.stanford.edu/data/email-EuAll.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "email-EuAll.txt")  # , create_using=nx.DiGraph()
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def dblp():
    """
    Returns the graph from: https://snap.stanford.edu/data/com-DBLP.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "dblp.txt")
    return graph.subgraph(max(nx.connected_components(graph), key=len))


# def gitub():
#     """
#     Returns the graph from: https://snap.stanford.edu/data/ca-GrQc.html,
#     where we preprocess it to only keep the largest connected component
#
#     :return: undirected NetworkX graph
#     """
#
#     graph = nx.read_edgelist(graph_dir + "github.csv", delimiter=',')
#     return graph.subgraph(max(nx.connected_components(graph), key=len))


def ca_astro_ph():
    """
    Returns the graph from: https://snap.stanford.edu/data/ca-AstroPh.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "ca-AstroPh.txt")
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def ca_hep_th():
    """
    Returns the graph from: https://snap.stanford.edu/data/cit-HepTh.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "ca-HepTh.txt")
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def enron_email():
    """
    Returns the graph from: https://snap.stanford.edu/data/email-Enron.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "email-enron.txt")
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def karate():
    """
    Returns the graph from: https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html,

    :return: undirected NetworkX graph
    """

    return nx.karate_club_graph()


def oregeon_1():
    """
    Returns the graph from: https://snap.stanford.edu/data/oregon1_010331.html,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_edgelist(graph_dir + "as-oregon1.txt")
    return graph.subgraph(max(nx.connected_components(graph), key=len))


def electrical():
    """
    Returns the graph from: http://konect.cc/networks/opsahl-powergrid/,
    where we preprocess it to only keep the largest connected component

    :return: undirected NetworkX graph
    """

    graph = nx.read_gml(graph_dir + "power.gml", label='id')
    return graph.subgraph(max(nx.connected_components(graph), key=len))


# def roadnet_ca():
#     """
#     Returns the graph from: https://snap.stanford.edu/data/roadNet-CA.html,
#     where we preprocess it to only keep the largest connected component
#
#     :return: undirected NetworkX graph
#     """
#
#     graph = nx.read_edgelist(graph_dir + "road-california.txt")
#     return graph.subgraph(max(nx.connected_components(graph), key=len))


"""
Custom Graphs
"""


def o4_graph():
    """
    Returns a 4 node disconnected graph

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    return G


def p4_graph():
    """
    Returns a 4 node path graph

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return G


def s4_graph():
    """
    Returns a 4 node star graph

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])
    return G


def c4_graph():
    """
    Returns a 4 node cycle graph

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    return G


def k4_1_graph():
    """
    Returns a 4 node diamond graph (1 diagonal edge)

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)])
    return G


def k4_2_graph():
    """
    Returns a 4 node diamond graph (2 diagonal edges), a.k.a. complete graph

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (2, 3)])
    return G


def two_c4_0_bridge():
    """
    Returns two disconnected 4 node cycle graphs

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                      (4, 5), (5, 6), (6, 7), (7, 4)])
    return G


def two_c4_1_bridge():
    """
    Returns two 4 node cycle graphs connected by 1 edge

    :return: undirected NetworkX graph
    """

    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                      (4, 5), (5, 6), (6, 7), (7, 4),
                      (2, 4)])
    return G


def two_c4_2_bridge():
    """
    Returns two 4 node cycle graphs connected by 2 edges

    :return: undirected NetworkX graph
    """

    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                      (4, 5), (5, 6), (6, 7), (7, 4),
                      (2, 4), (2, 4)])
    return G


def two_c4_3_bridge():
    """
    Returns two 4 node cycle graphs connected by 3 edges

    :return: undirected NetworkX graph
    """

    G = nx.MultiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                      (4, 5), (5, 6), (6, 7), (7, 4),
                      (2, 4), (2, 4), (2, 4)])
    return G


graph_urls = {
    'wiki_vote': ('https://github.com/safreita1/TIGER/blob/master/datasets/wiki-Vote.txt', 'https://snap.stanford.edu/data/wiki-Vote.txt.gz'),
    'p2p_gnuetella08': ('https://github.com/safreita1/TIGER/blob/master/datasets/p2p-Gnutella08.txt', 'https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz'),

    'dblp': ('https://github.com/safreita1/TIGER/blob/master/datasets/dblp.txt', 'https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz'),
    'ca_hep_th': ('https://github.com/safreita1/TIGER/blob/master/datasets/cit-HepTh.txt', 'https://snap.stanford.edu/data/cit-HepTh.txt.gz'),
    'cit_hep_th': ('https://github.com/safreita1/TIGER/blob/master/datasets/ca-HepTh.txt', 'https://snap.stanford.edu/data/M.txt.gz'),
    'ca_grqc': ('https://github.com/safreita1/TIGER/blob/master/datasets/ca-GrQc.txt', 'https://snap.stanford.edu/data/ca-GrQc.txt.gz'),
    'ca_astro_ph': ('https://github.com/safreita1/TIGER/blob/master/datasets/ca-AstroPh.txt', 'https://snap.stanford.edu/data/ca-AstroPh.txt.gz'),

    'email_eu_all': ('https://github.com/safreita1/TIGER/blob/master/datasets/email-EuAll.txt', 'https://snap.stanford.edu/data/email-EuAll.txt.gz'),
    'enron_email': ('https://github.com/safreita1/TIGER/blob/master/datasets/email-enron.txt', 'https://snap.stanford.edu/data/email-Enron.txt.gz'),

    'ky2': ('https://github.com/safreita1/TIGER/blob/master/datasets/ky2.txt', 'http://www.uky.edu/WDST/KYEPAzip/ky2%20EPANET.zip'),
    'as_733': ('https://github.com/safreita1/TIGER/blob/master/datasets/as19971108.txt', 'http://snap.stanford.edu/data/as-733.tar.gz'),
    'oregon_1': ('https://github.com/safreita1/TIGER/blob/master/datasets/as-oregon1.txt', 'https://snap.stanford.edu/data/oregon1_010331.txt.gz'),
    'electrical': ('https://github.com/safreita1/TIGER/blob/master/datasets/power.gml', 'http://konect.uni-koblenz.de/downloads/tsv/opsahl-powergrid.tar.bz2'),
    # 'roadnet_ca': 'https://snap.stanford.edu/data/roadNet-CA.txt.gz'
}

models = {
    'ER': erdos_reyni,
    'WS': watts_strogatz,
    'BA': barbasi_albert,
    'CSF': clustered_scale_free
}

datasets = {
    'ky2': wdn_ky2,
    'as_733': as_733,
    'p2p_gnuetella08': p2p_gnuetella08,
    'ca_grqc': ca_grqc,
    'cit_hep_th': cit_hep_th,
    'wiki_vote': wiki_vote,
    'email_eu_all': email_eu_all,
    'dblp': dblp,
    'ca_astro_ph': ca_astro_ph,
    'ca_hep_th': ca_hep_th,
    'enron_email': enron_email,
    'karate': karate,
    'oregon_1': oregeon_1,
    'electrical': electrical,
}

custom = {
    'o4': o4_graph,
    'p4': p4_graph,
    's4': s4_graph,
    'c4': c4_graph,
    'k4-1': k4_1_graph,
    'k4-2': k4_2_graph,
    'c4_no_bridge': two_c4_0_bridge,
    'c4_1_bridge': two_c4_1_bridge,
    'c4_2_bridge': two_c4_2_bridge,
    'c4_3_bridge': two_c4_3_bridge
}