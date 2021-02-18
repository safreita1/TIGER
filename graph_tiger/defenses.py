import os
import heapq
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.interpolate import interp1d

from graph_tiger.graphs import graph_loader
from graph_tiger.measures import run_measure
from graph_tiger.simulations import Simulation
from graph_tiger.attacks import get_node_ns as get_node_ns_attack
from graph_tiger.attacks import get_node_pr as get_node_pr_attack
from graph_tiger.attacks import get_node_eig as get_node_eig_attack
from graph_tiger.attacks import get_node_rnd as get_node_rnd_attack
from graph_tiger.attacks import get_node_ib as get_node_ib_attack
from graph_tiger.attacks import get_node_rb as get_node_rb_attack
from graph_tiger.attacks import get_node_id as get_node_id_attack
from graph_tiger.attacks import get_node_rd as get_node_rd_attack
from graph_tiger.attacks import get_attack_category, run_attack_method


def run_defense_method(graph, method, k=3, seed=None):
    """
    Runs a specified defense on an undirected graph, returning a list of nodes to defend.

    :param graph: an undirected NetworkX graph
    :param method: a string representing one of the attack methods
    :param k: number of nodes or edges to attack
    :param seed: sets the seed in order to obtain reproducible defense runs
    :return: a list of nodes or edge tuples to defend
    """

    protected = []
    if method in methods and k > 0:
        if seed is not None: np.random.seed(seed)
        protected = methods[method](graph, k)
    else:
        print("{} not implemented or k <= 0".format(method))

    return protected


def get_defense_methods():
    """
    Gets a list of available defense methods as a list of functions.

    :return: a list of all defense functions
    """

    return methods.keys()


def get_defense_category(method):
    """
    Gets the defense category e.g., 'node', 'edge' defense.

    :param method: a string representing the defense method
    :return: a string representing the defense type ('node' or 'edge')
    """

    category = None

    if method in categories:
        category = categories[method]

    return category


def get_node_ns(graph, k=3):
    """
    Get k nodes to defend based on the Netshield algorithm :cite:`tong2010vulnerability,chen2015node`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend

    :return: a list of nodes to defend
    """

    return get_node_ns_attack(graph, k)


def get_node_pr(graph, k=3):
    """
    Get k nodes to defend based on top PageRank entries :cite:`page1999pagerank`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend

    :return: a list of nodes to defend
    """

    return get_node_pr_attack(graph, k)


def get_node_eig(graph, k=3):
    """
    Get k nodes to defend based on top eigenvector centrality entries

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend
    :return: a list of nodes to defend
    """

    return get_node_eig_attack(graph, k)


def get_node_ib(graph, k=3, approx=np.inf):
    """
    Get k nodes to defend based on Initial Betweenness (IB) Removal :cite:`holme2002attack`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend
    :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
    is the number of nodes in the graph

    :return: a list of nodes to defend
    """

    return get_node_ib_attack(graph, k, approx)


def get_node_rb(graph, k=3, approx=np.inf):
    """
    Get k nodes to defend based on Recalculated Betweenness (RB) Removal :cite:`holme2002attack`

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend
    :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
    is the number of nodes in the graph

    :return: a list of nodes to defend
    """

    return get_node_rb_attack(graph, k, approx)


def get_node_id(graph, k=3):
    """
    Get k nodes to defend based on Initial Degree (ID) Removal :cite:`holme2002attack`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend

    :return: a list of nodes to defend
    """

    return get_node_id_attack(graph, k)


def get_node_rd(graph, k=3):
    """
    Get k nodes to defend based on Recalculated Degree (RD) Removal :cite:`holme2002attack`.

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend

    :return: a list of nodes to defend
    """

    return get_node_rd_attack(graph, k)


def get_node_rnd(graph, k=3):
    """
    Randomly select k distinct nodes to defend

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend

    :return: a list of nodes to defend
    """
    return get_node_rnd_attack(graph, k)


def get_central_edges(graph, k, method='eig'):
    """
    Internal function to compute edge PageRank, eigenvector centrality and degree centrality

    :param graph: undirected NetworkX graph
    :param k: int number of nodes to defend
    :param method: string representing defense method
    :return: list of edges to add
    """
    max_deg = max([d[1] for d in graph.degree])
    top_nodes = get_node_id(graph, k=max_deg + k)

    if method == 'pr':
        centrality = nx.pagerank(graph)
    elif method == 'eig':
        centrality = nx.eigenvector_centrality(graph)
    elif method == 'deg':
        centrality = dict(graph.degree)

    score = {}
    tried = set()

    for u in top_nodes:
        for v in top_nodes:
            if u != v and not graph.has_edge(u, v) and (u, v) not in tried:
                tried.add((u, v))
                tried.add((v, u))

                score[(u, v)] = centrality[u] * centrality[v]

    nodes = heapq.nlargest(k, score, key=score.get)

    return nodes


def add_edge_pr(graph, k=3):
    """
    Get k edges to defend based on top edge PageRank entries :cite:`tong2012gelling`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to add
    :return: a dictionary of the edges to be 'added'
    """

    info = defaultdict(list)
    info['added'] = get_central_edges(graph, k, method='pr')

    return info


def add_edge_eig(graph, k=3):
    """
    Get k edges to defend based on top edge eigenvector centrality entries :cite:`tong2012gelling`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to add
    :return: a dictionary of the edges to be 'added'
    """

    info = defaultdict(list)
    info['added'] = get_central_edges(graph, k, method='eig')

    return info


def add_edge_degree(graph, k=3):
    """
    Add k edges to defend based on top edge degree centrality entries :cite:`tong2012gelling`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to add
    :return: a list of edges to add
    """

    info = defaultdict(list)
    info['added'] = get_central_edges(graph, k, method='deg')

    return info


def add_edge_rnd(graph, k=3):
    """
    Add k random edges to the graph

    :param graph: an undirected NetworkX graph
    :param k: number of edges to add
    :return: a dictionary of the edges to be 'added'
    """

    graph_ = graph.copy()
    info = defaultdict(list)

    for _ in range(k):
        nodes = graph_.nodes
        u, v = np.random.choice(nodes, 2, replace=False)

        while graph_.has_edge(u, v) or u == v:
            u, v = np.random.choice(nodes, 2, replace=False)

        graph_.add_edge(u, v)
        info['added'].append((u, v))

    return info


def add_edge_pref(graph, k=3):
    """
    Adds an edge connecting two nodes with the lowest degrees :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to add
    :return: a dictionary of the edges to be 'added'
    """

    info = defaultdict(list)
    deg = dict(graph.degree)

    edges_tried = set()
    for _ in range(k):
        u = min(deg, key=deg.get)

        u_d = deg[u] + 1
        deg.pop(u)

        v = min(deg, key=deg.get)

        deg[v] += 1
        deg[u] = u_d

        if (u, v) not in edges_tried and (v, u) not in edges_tried:
            info['added'].append((u, v))
            edges_tried.update([(u, v), (v, u)])

    return info


def rewire_edge_rnd(graph, k=3):
    """
    Removes a random edge and adds one randomly :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    info = defaultdict(list)
    edges = list(graph.edges)

    m = len(edges)
    k = min(k, m)
    idx = np.random.choice(m, k, replace=False)

    info['removed'] = [edges[i] for i in idx]
    info['added'] = add_edge_rnd(graph, k=k)['added']

    return info


def rewire_edge_rnd_neighbor(graph, k=3):
    """
    Randomly selects a neighbor of a node and removes the edge; then adds a random edge :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    info = defaultdict(list)

    edges_seen = set()
    nodes = [n for n in graph.nodes if len(list(graph.neighbors(n))) > 0]  # get non-isolated nodes
    nodes = np.random.choice(nodes, min(k, len(nodes)), replace=False)

    for u in nodes:
        v = np.random.choice(list(graph.neighbors(u)))
        removed = (u, v)

        added = add_edge_rnd(graph, k=1)['added'][0]
        if added not in edges_seen and removed not in edges_seen:
            info['added'].append(added)
            info['removed'].append(removed)

            edges_seen.update([added, added[::-1], removed, removed[::-1]])

    return info


def rewire_edge_pref(graph, k=3):
    """
    Selects node with highest degree, randomly removes a neighbor; adds edge to random node in graph :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    graph_ = graph.copy()
    info = defaultdict(list)

    for _ in range(k):
        u = max(dict(graph_.degree), key=dict(graph_.degree).get)
        nbr = np.random.choice(list(graph_.neighbors(u)))

        graph_.remove_edge(u, nbr)
        info['removed'].append((u, nbr))

        v = np.random.choice(graph_.nodes)
        while graph_.has_edge(u, v) or u == v:
            v = np.random.choice(graph_.nodes)

        graph_.add_edge(nbr, v)
        info['added'].append((nbr, v))

    return info


def rewire_edge_pref_rnd(graph, k=3):
    """
    Selects an edge, disconnects the higher degree node, and reconnects to a random one :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    graph_ = graph.copy()
    info = defaultdict(list)

    edges = list(graph_.edges)

    k = min(len(edges), k)
    idx = np.random.choice(len(edges), k, replace=False)

    edges = [edges[i] for i in idx]
    for u, v in edges:
        info['removed'].append((u, v))

        rnd_node = np.random.choice(graph_.nodes)
        if graph_.degree(u) > graph_.degree(v):
            graph_.add_edge(v, rnd_node)
            info['added'].append((v, rnd_node))
        else:
            graph_.add_edge(u, rnd_node)
            info['added'].append((u, rnd_node))

    return info


categories = {
    'ns_node': 'node',
    'pr_node': 'node',
    'eig_node': 'node',
    'id_node': 'node',
    'ib_node': 'node',
    'rnd_node': 'node',

    'add_edge_pr': 'edge',
    'add_edge_eig': 'edge',
    'add_edge_deg': 'edge',
    'add_edge_random': 'edge',
    'add_edge_preferential': 'edge',
    'rewire_edge_random': 'edge',
    'rewire_edge_random_neighbor': 'edge',
    'rewire_edge_preferential': 'edge',
    'rewire_edge_preferential_random': 'edge'


}

methods = {
    'ns_node': get_node_ns,
    'pr_node': get_node_pr,
    'eig_node': get_node_eig,
    'id_node': get_node_id,
    'ib_node': get_node_ib,
    'rnd_node': get_node_rnd,

    'add_edge_pr': add_edge_pr,
    'add_edge_eig': add_edge_eig,
    'add_edge_deg': add_edge_degree,
    'add_edge_random': add_edge_rnd,
    'add_edge_preferential': add_edge_pref,
    'rewire_edge_random': rewire_edge_rnd,
    'rewire_edge_random_neighbor': rewire_edge_rnd_neighbor,
    'rewire_edge_preferential': rewire_edge_pref,
    'rewire_edge_preferential_random': rewire_edge_pref_rnd
}


class Defense(Simulation):
    """
    This class simulates a variety of defense techniques on an undirected NetworkX graph

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
        self.protected = defaultdict(list)
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
        if self.prm['attack'] is not None and self.prm['k_a'] > 0:
            self.attacked = run_attack_method(self.graph_, self.prm['attack'], self.prm['k_a'], approx=self.prm['attack_approx'], seed=self.prm['seed'])

            if get_attack_category(self.prm['attack']) == 'edge':
                self.graph_.remove_nodes_from(self.attacked)

        elif self.prm['attack'] is not None:
            print(self.prm['attack'], "not available or k <= 0")

        # defended nodes or edges
        if self.prm['defense'] is not None and self.prm['steps'] > 0:

            if get_defense_category(self.prm['defense']) == 'node':
                self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['steps'], seed=self.prm['seed'])

            elif get_defense_category(self.prm['defense']) == 'edge':
                self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['steps'], seed=self.prm['seed'])

        elif self.prm['defense'] is not None:
            print(self.prm['defense'], "not available or k <= 0")

        # remove attacked nodes after checking that they are not defended
        if get_attack_category(self.prm['attack']) == 'node':
            if get_defense_category(self.prm['defense']) == 'node':
                diff = set(self.protected) - set(self.attacked)
                self.graph_.remove_nodes_from(diff)
            else:
                self.graph_.remove_nodes_from(self.attacked)

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
            'failed': len(self.graph_) - len(max(ccs)),
            'measure': measure,
            'protected': self.protected,
            'edges_added': self.protected['added'][0:step] if 'added' in self.protected else [],
            'edges_removed': self.protected['removed'][0:step] if 'removed' in self.protected else []
        }

    def run_single_sim(self):
        """
        Run the defense simulation
        """

        for step in range(self.prm['steps']):
            if step < len(self.protected) and len(self.protected) > 0 and get_defense_category(self.prm['defense']) == 'edge':
                self.track_simulation(step)

                u, v = self.protected['added'][step]
                self.graph_.add_edge(u, v)

                if 'removed' in self.protected[step]:
                    u, v = self.protected['removed'][step]
                    self.graph.remove_edge(u, v)

            else:
                self.track_simulation(step)
                print("Ending defense simulation early, not an 'edge' defense or out of {}s".format(get_attack_category(self.prm['defense'])))

        results = [v['measure'] if v['measure'] is not None else 0 for k, v in self.sim_info.items()]
        return results


def main():
    graph = graph_loader(graph_type='water', seed=1)

    params = {
        'runs': 1,
        'steps': 30,
        'seed': 1,

        'attack': 'rb_node',
        'k_a': 30,
        'attack_approx': int(0.1*len(graph)),

        'defense': 'add_edge_random',
        'robust_measure': 'largest_connected_component',

        'plot_transition': True,
        'gif_animation': True,

        'edge_style': None,
        'node_style': 'spectral',
        'fa_iter': 2000,
    }

    cf = Defense(graph, **params)
    results = cf.run_simulation()
    cf.plot_results(results)


if __name__ == '__main__':
    main()
