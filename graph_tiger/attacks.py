import scipy
import heapq
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d

from graph_tiger.simulations import Simulation
from graph_tiger.measures import run_measure
from graph_tiger.graphs import *
from graph_tiger.utils import get_sparse_graph


def run_attack_method(graph, method, k=3, approx=None, seed=None):
    """
    Runs a specified attack on an undirected graph, returning a list of nodes to attack

    :param graph: an undirected NetworkX graph
    :param method: a string representing one of the attack methods
    :param k: number of nodes or edges to attack
    :param approx: attack approximation parameter (not available for every measure)
    :param seed: sets the seed in order to obtain reproducible attacks

    :return: a list of nodes selected for attack
    """

    attacked = []
    if method in methods and k > 0:
        if seed is not None:
            np.random.seed(seed)
        if approx is None:
            attacked = methods[method](graph, k)
        else:
            attacked = methods[method](graph, k, approx=approx)
    else:
        print("{} not implemented or k<= 0".format(method))

    return attacked


def get_attack_methods():
    """
    Gets a list of available attach methods as a list of functions

    :return: a list of all attack functions
    """

    return methods.keys()


def get_attack_category(method):
    """
    Gets the attack category e.g., 'node', 'edge' attack

    :param method: a string representing the attack method

    :return: a string representing the attack type ('node' or 'edge')
    """

    category = None

    if method in categories:
        category = categories[method]

    return category


def get_node_ns(graph, k=3):
    """
    Get k nodes to attack based on the Netshield algorithm: http://tonghanghang.org/pdfs/icdm10_netshield.pdf

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack

    :return: a list of nodes to attack
    """

    if not scipy.sparse.issparse(graph):
        sparse_graph = get_sparse_graph(graph)
    else:
        sparse_graph = graph

    lam, u = eigsh(sparse_graph, k=1, which='LA')
    lam = lam[0]

    u = np.abs(np.real(u).flatten())
    v = (2 * lam * np.ones(len(u))) * np.power(u, 2)

    nodes = []
    for i in range(k):
        B = sparse_graph[:, nodes]
        b = B * u[nodes]

        score = v - 2 * b * u
        score[nodes] = -1

        nodes.append(np.argmax(score))

    return nodes


def get_node_pr(graph, k=3):
    """
    Get k nodes to attack based on top PageRank entries

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack

    :return: a list of nodes to attack
    """

    centrality = nx.pagerank(graph, alpha=0.85)
    nodes = heapq.nlargest(k, centrality, key=centrality.get)

    return nodes


def get_node_eig(graph, k=3):
    """
    Get k nodes to attack based on top eigenvector centrality entries

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :return: a list of nodes to attack
    """

    centrality = nx.eigenvector_centrality(graph, tol=1E-3, max_iter=500)
    nodes = heapq.nlargest(k, centrality, key=centrality.get)

    return nodes


def get_node_id(graph,  k=3):
    """
    Get k nodes to attack based on Initial Degree (ID) Removal :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack

    :return: a list of nodes to attack
    """

    centrality = dict(graph.degree())
    nodes = heapq.nlargest(k, centrality, key=centrality.get)

    return nodes


def get_node_rd(graph, k=3):
    """
    Get k nodes to attack based on Recalculated Degree (RD) Removal :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack

    :return: a list of nodes to attack
    """
    graph_ = graph.copy()

    nodes = []
    for _ in range(k):
        u = get_node_id(graph_, k=1)[0]

        nodes.append(u)
        graph_.remove_node(u)

    return nodes


def get_node_ib(graph, k=3, approx=np.inf):
    """
    Get k nodes to attack based on Initial Betweenness (IB) Removal :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
    is the number of nodes in the graph

    :return: a list of nodes to attack
    """

    centrality = nx.betweenness_centrality(graph, k=min(len(graph), approx))
    nodes = heapq.nlargest(k, centrality, key=centrality.get)

    return nodes


def get_node_rb(graph, k=3, approx=np.inf):
    """
    Get k nodes to attack based on Recalculated Betweenness (RB) Removal :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack
    :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
    is the number of nodes in the graph

    :return: a list of nodes to attack
    """
    graph_ = graph.copy()

    nodes = []
    for _ in range(k):
        u = get_node_ib(graph_, k=1, approx=approx)[0]

        nodes.append(u)
        graph_.remove_node(u)

    return nodes


def get_node_rnd(graph, k=3):
    """
    Randomly select k distinct nodes to attack

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to attack

    :return: a list of nodes to attack
    """
    return np.random.choice(graph.nodes, k, replace=False).tolist()


def get_edge_line_ns(graph, k=3):
    """
    Get k edges to attack using Netshield by transforming the graph into a line graph :cite:`tong2010vulnerability,tong2012gelling`

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack

    :return: a list of edge tuples to attack
    """
    line_graph = nx.line_graph(graph)
    line_nodes = list(line_graph.nodes)

    idx = get_node_ns(line_graph, k=k)
    edges = [line_nodes[i] for i in idx]

    return edges


def get_edge_line_pr(graph, k=3):
    """
    Get k edges to attack using PageRank by transforming the graph into a line graph :cite:`tong2012gelling`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack

    :return: a list of edge tuples to attack
    """
    line_graph = nx.line_graph(graph)

    return get_node_pr(line_graph, k=k)


def get_edge_line_eig(graph, k=3):
    """
    Get k edges to attack using eigenvector centrality by transforming the graph into a line graph :cite:`tong2012gelling`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack

    :return: a list of edge tuples to attack
    """
    line_graph = nx.line_graph(graph)

    return get_node_eig(line_graph, k=k)


def get_edge_line_deg(graph, k=3):
    """
    Get k edges to attack using degree centrality by transforming the graph into a line graph :cite:`tong2012gelling`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack

    :return: a list of edge tuples to attack
    """
    line_graph = nx.line_graph(graph)

    return get_node_id(line_graph, k=k)


def get_edge_id(graph, k=3):
    """
    Get k edges to attack based on Initial Degree (ID) Removal :cite:`holme2002attack`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack

    :return: a list of edge tuples to attack
    """

    centrality = {(u, v): graph.degree(u) * graph.degree(v) for u, v in graph.edges}
    edges = heapq.nlargest(k, centrality, key=centrality.get)

    return edges


def get_edge_rd(graph, k=3):
    """
    Get k edges to attack based on Recalculated Degree (RD) Removal :cite:`holme2002attack`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack

    :return: a list of edge tuples to attack
    """
    graph_ = graph.copy()

    edges = []
    for _ in range(k):
        u, v = get_edge_id(graph_, k=1)[0]

        edges.append((u, v))
        graph_.remove_edge(u, v)

    return edges


def get_edge_ib(graph, k=3, approx=np.inf):
    """
    Get k edges to attack based on Initial Betweenness (IB) Removal :cite:`holme2002attack`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack
    :param approx: number of edges to approximate the betweenness centrality

    :return: a list of edge tuples to attack
    """

    centrality = nx.edge_betweenness_centrality(graph, k=min(len(graph), approx))
    edges = heapq.nlargest(k, centrality, key=centrality.get)

    return edges


def get_edge_rb(graph, k=3, approx=np.inf):
    """
    Get k edges to attack based on Recalculated Betweenness (RB) Removal :cite:`holme2002attack`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack
    :param approx: number of edges to approximate the betweenness centrality

    :return: a list of edge tuples to attack
    """

    top_edges = []
    graph_ = graph.copy()

    for _ in range(k):
        u, v = get_edge_ib(graph_, k=1, approx=approx)[0]

        top_edges.append((u, v))
        graph_.remove_edge(u, v)

    return top_edges


def get_edge_rnd(graph, k=3):
    """
    Randomly select k edges to attack

    :param graph: an undirected NetworkX graph
    :param k: number of edges to attack

    :return: a list of edge tuples to attack
    """
    edges = list(graph.edges)
    idx = np.random.choice(len(edges), k, replace=False)
    rnd_edges = [edges[i] for i in idx]

    return rnd_edges


categories = {
    'ns_node': 'node',
    'pr_node': 'node',
    'eig_node': 'node',
    'id_node': 'node',
    'rd_node': 'node',
    'ib_node': 'node',
    'rb_node': 'node',
    'rnd_node': 'node',
    None: None,

    'ns_line_edge': 'edge',
    'pr_line_edge': 'edge',
    'eig_line_edge': 'edge',
    'deg_line_edge': 'edge',
    'id_edge': 'edge',
    'rd_edge': 'edge',
    'ib_edge': 'edge',
    'rb_edge': 'edge',
    'rnd_edge': 'edge'
}

methods = {
    'ns_node': get_node_ns,
    'pr_node': get_node_pr,
    'eig_node': get_node_eig,
    'id_node': get_node_id,
    'rd_node': get_node_rd,
    'ib_node': get_node_ib,
    'rb_node': get_node_rb,
    'rnd_node': get_node_rnd,

    'ns_line_edge': get_edge_line_ns,
    'pr_line_edge': get_edge_line_pr,
    'eig_line_edge': get_edge_line_eig,
    'deg_line_edge': get_edge_line_deg,
    'id_edge': get_edge_id,
    'rd_edge': get_edge_rd,
    'ib_edge': get_edge_ib,
    'rb_edge': get_edge_rb,
    'rnd_edge': get_edge_rnd
}


class Attack(Simulation):
    """
    This class simulates a variety of attack strategies on an undirected NetworkX graph

    :param graph: an undirected NetworkX graph
    :param runs: an integer number of times to run the simulation
    :param steps: an integer number of steps to run a single simulation
    :param attack: a string representing the attack strategy to run
    :param defense: a string representing the defense strategy to run
    :param k_d: an integer number of nodes to defend
    :param **kwargs: see parent class Simulation for additional options
    """

    def __init__(self, graph, runs=10, steps=50, attack='id_node', defense=None, k_d=0, **kwargs):
        super().__init__(graph, runs, steps, **kwargs)
        self.graph = graph

        self.prm.update({
            'attack': attack,
            'attack_approx': None,

            'k_d': k_d,
            'defense': defense,

            'robust_measure': 'largest_connected_component',
        })

        self.prm.update(kwargs)

        if self.prm['plot_transition'] or self.prm['gif_animation']:
            self.node_pos, self.edge_pos = self.get_graph_coordinates()

        self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
        os.makedirs(self.save_dir, exist_ok=True)

        self.attacked = []
        self.protected = []
        self.connectivity = []

        self.reset_simulation()

    def reset_simulation(self):
        """
        Resets the simulation between each run
        """

        self.graph_ = self.graph.copy()
        self.attacked = []
        self.protected = []
        self.connectivity = []

        # attacked nodes or edges
        if self.prm['attack'] is not None and self.prm['steps'] > 0:
            self.attacked = run_attack_method(self.graph_, self.prm['attack'], self.prm['steps'], approx=self.prm['attack_approx'], seed=self.prm['seed'])

        elif self.prm['attack'] is not None:
            print(self.prm['attack'], "not available or k <= 0")

        # defended nodes or edges
        if self.prm['defense'] is not None and self.prm['k_d'] > 0:
            from defenses import get_defense_category, run_defense_method

            if get_defense_category(self.prm['defense']) == 'node':
                self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.prm['seed'])

            elif get_defense_category(self.prm['defense']) == 'edge':
                protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.prm['seed'])

                self.graph_.add_edges_from(protected['added'])
                if 'removed' in protected:
                    self.graph_.remove_edges_from(protected['removed'])

        elif self.prm['defense'] is not None:
            print(self.prm['defense'], "not available or k <= 0")

    def track_simulation(self, step):
        """
        Keeps track of important simulation information at each step of the simulation

        :param step: current simulation iteration
        """

        measure = run_measure(self.graph_, self.prm['robust_measure'])

        ccs = list(nx.connected_components(self.graph_))
        ccs.sort(key=len, reverse=True)

        m = interp1d([0, len(ccs)], [0.15, 1])

        status = {}
        for n in self.graph:
            for idx, cc in enumerate(ccs):
                if n in self.attacked[0:step]:
                    status[n] = 1
                    break
                elif n in cc:
                    status[n] = float(m(idx))
                    break
                else:
                    status[n] = 0

        self.sim_info[step] = {
            'status':  list(status.values()),
            'failed': len(self.attacked[0:step]),
            'measure': measure,
            'protected': self.protected
        }

    def run_single_sim(self):
        """
        Run the attack simulation
        """

        for step in range(self.prm['steps']):
            if step < len(self.attacked) and len(self.attacked) > 0:

                self.track_simulation(step)

                v = self.attacked[step]

                if get_attack_category(self.prm['attack']) == 'edge':
                    self.graph_.remove_edge(v[0], v[1])

                elif get_attack_category(self.prm['attack']) == 'node' and v not in self.protected:
                    self.graph_.remove_node(v)

            else:
                print("Ending attack simulation early, ran of {}s".format(get_attack_category(self.prm['attack'])))

        results = [v['measure'] if v['measure'] is not None else 0 for k, v in self.sim_info.items()]
        return results


def main():
    graph = graph_loader(graph_type='water', seed=1)

    params = {
        'runs': 1,
        'steps': 30,
        'seed': 1,

        'attack': 'rb_node',
        'attack_approx': int(0.1*len(graph)),

        'k_d': 0,
        'defense': None,

        'robust_measure': 'largest_connected_component',

        'plot_transition': True,
        'gif_animation': True,

        'edge_style': None,
        'node_style': 'spectral',
        'fa_iter': 2000,
    }

    cf = Attack(graph, **params)
    results = cf.run_simulation()
    cf.plot_results(results)


if __name__ == '__main__':
    main()
