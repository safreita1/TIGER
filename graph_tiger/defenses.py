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
    Runs a specified defense on an undirected graph.

    :param graph: an undirected NetworkX graph
    :param method: a string representing one of the defense methods
    :param k: number of nodes or edges to defend
    :param seed: sets the seed in order to obtain reproducible defense runs
    :return: a list of nodes or a dictionary of edge changes
    """

    if method not in methods:
        raise ValueError("defense method '{}' is not implemented".format(method))
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError('k must be a nonnegative integer')

    category = get_defense_category(method)
    if k == 0:
        return [] if category == 'node' else defaultdict(list)

    if category == 'node' and k > len(graph):
        raise ValueError('k exceeds the number of available nodes')

    if method.startswith('add_edge') and k > len(list(nx.non_edges(graph))):
        raise ValueError('k exceeds the number of available nonedges')

    if method.startswith('rewire_edge') and k > len(graph.edges):
        raise ValueError('k exceeds the number of available edges')

    rng = np.random.RandomState(seed)
    if method in ['rnd_node', 'add_edge_random', 'rewire_edge_random',
                  'rewire_edge_random_neighbor', 'rewire_edge_preferential',
                  'rewire_edge_preferential_random']:
        return methods[method](graph, k, rng=rng)

    return methods[method](graph, k)

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
    Get k nodes to defend based on the Netshield algorithm :cite:`tong2010vulnerability`.

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


def get_node_rnd(graph, k=3, rng=None):
    """
    Randomly select k distinct nodes to defend

    :param graph: an undirected NetworkX graph
    :param k: number of nodes to defend

    :return: a list of nodes to defend
    """
    rng = np.random if rng is None else rng
    return rng.choice(list(graph.nodes), k, replace=False).tolist()


def get_central_edges(graph, k, method='eig'):
    """
    Internal function to compute edge PageRank, eigenvector centrality and degree centrality.

    :param graph: undirected NetworkX graph
    :param k: int number of edges to add
    :param method: string representing defense method
    :return: list of edges to add
    """

    if method == 'pr':
        centrality = nx.pagerank(graph)
    elif method == 'eig':
        centrality = nx.eigenvector_centrality(graph)
    elif method == 'deg':
        centrality = dict(graph.degree)
    else:
        raise ValueError("central edge method '{}' is not implemented".format(method))

    score = {(u, v): centrality[u] * centrality[v] for u, v in nx.non_edges(graph)}

    return heapq.nlargest(k, score, key=score.get)


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
    :return: a dictionary of the edges to be 'added'
    """

    info = defaultdict(list)
    info['added'] = get_central_edges(graph, k, method='deg')

    return info


def add_edge_rnd(graph, k=3, rng=None):
    """
    Add k random nonedges to the graph.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to add
    :param rng: optional NumPy random generator
    :return: a dictionary of the edges to be 'added'
    """

    rng = np.random if rng is None else rng
    info = defaultdict(list)
    available = list(nx.non_edges(graph))

    if k > len(available):
        raise ValueError('k exceeds the number of available nonedges')

    idx = rng.choice(len(available), k, replace=False)
    info['added'] = [available[i] for i in np.atleast_1d(idx)]

    return info


def add_edge_pref(graph, k=3):
    """
    Adds edges between the lowest-degree feasible pairs :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to add
    :return: a dictionary of the edges to be 'added'
    """

    graph_ = graph.copy()
    info = defaultdict(list)
    order = {n: idx for idx, n in enumerate(graph_.nodes)}

    for _ in range(k):
        available = list(nx.non_edges(graph_))
        if len(available) == 0:
            raise ValueError('k exceeds the number of available nonedges')

        degree = dict(graph_.degree)
        edge = min(available, key=lambda e: (degree[e[0]] + degree[e[1]],
                                             max(degree[e[0]], degree[e[1]]),
                                             order[e[0]], order[e[1]]))
        graph_.add_edge(*edge)
        info['added'].append(edge)

    return info


def get_random_nonedges(graph, k, rng, excluded=None):
    """
    Return k random nonedges, excluding specified unordered node pairs.
    """

    excluded = set() if excluded is None else excluded
    available = [edge for edge in nx.non_edges(graph) if frozenset(edge) not in excluded]

    if k > len(available):
        raise ValueError('not enough nonedges are available for rewiring')

    idx = rng.choice(len(available), k, replace=False)
    return [available[i] for i in np.atleast_1d(idx)]


def rewire_edge_rnd(graph, k=3, rng=None):
    """
    Removes k random edges and adds k different random nonedges :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :param rng: optional NumPy random generator
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    rng = np.random if rng is None else rng
    graph_ = graph.copy()
    info = defaultdict(list)
    edges = list(graph_.edges)

    if k > len(edges):
        raise ValueError('k exceeds the number of available edges')

    idx = rng.choice(len(edges), k, replace=False)
    info['removed'] = [edges[i] for i in np.atleast_1d(idx)]
    graph_.remove_edges_from(info['removed'])

    excluded = {frozenset(edge) for edge in info['removed']}
    info['added'] = get_random_nonedges(graph_, k, rng, excluded=excluded)

    return info


def rewire_edge_rnd_neighbor(graph, k=3, rng=None):
    """
    Randomly removes a neighbor edge and adds a different random edge :cite:`beygelzimer2005improving`.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :param rng: optional NumPy random generator
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    rng = np.random if rng is None else rng
    graph_ = graph.copy()
    info = defaultdict(list)
    removed_seen = set()

    for _ in range(k):
        candidates = [(u, v) for u in graph_.nodes for v in graph_.neighbors(u)
                      if frozenset((u, v)) not in removed_seen]

        if len(candidates) == 0:
            raise ValueError('not enough distinct neighbor edges are available')

        removed = candidates[int(rng.choice(len(candidates)))]
        graph_.remove_edge(*removed)
        removed_seen.add(frozenset(removed))

        added = get_random_nonedges(graph_, 1, rng, excluded=removed_seen)[0]
        graph_.add_edge(*added)

        info['removed'].append(removed)
        info['added'].append(added)

    return info

def rewire_edge_pref(graph, k=3, rng=None):
    """
    Detaches a neighbor of a highest-degree node and reconnects that neighbor elsewhere.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :param rng: optional NumPy random generator
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    rng = np.random if rng is None else rng
    graph_ = graph.copy()
    info = defaultdict(list)

    for _ in range(k):
        nodes = [n for n in graph_.nodes if graph_.degree(n) > 0]
        if len(nodes) == 0:
            raise ValueError('not enough edges are available for rewiring')

        u = max(nodes, key=dict(graph_.degree).get)
        nbrs = list(graph_.neighbors(u))
        nbr = nbrs[int(rng.choice(len(nbrs)))]
        removed = (u, nbr)

        graph_.remove_edge(*removed)
        excluded = {frozenset(removed)}
        candidates = [(nbr, v) for v in graph_.nodes
                      if nbr != v and not graph_.has_edge(nbr, v)
                      and frozenset((nbr, v)) not in excluded]

        if len(candidates) == 0:
            raise ValueError('not enough nonedges are available for rewiring')

        added = candidates[int(rng.choice(len(candidates)))]
        graph_.add_edge(*added)

        info['removed'].append(removed)
        info['added'].append(added)

    return info


def rewire_edge_pref_rnd(graph, k=3, rng=None):
    """
    Disconnects the higher-degree endpoint and reconnects the other endpoint randomly.

    :param graph: an undirected NetworkX graph
    :param k: number of edges to rewire
    :param rng: optional NumPy random generator
    :return: a dictionary of the edges to be 'removed' and edges to be 'added'
    """

    rng = np.random if rng is None else rng
    graph_ = graph.copy()
    info = defaultdict(list)
    edges = list(graph_.edges)

    if k > len(edges):
        raise ValueError('k exceeds the number of available edges')

    idx = rng.choice(len(edges), k, replace=False)
    selected = [edges[i] for i in np.atleast_1d(idx)]

    for u, v in selected:
        anchor = v if graph_.degree(u) > graph_.degree(v) else u
        removed = (u, v)
        graph_.remove_edge(*removed)

        excluded = {frozenset(removed)}
        candidates = [(anchor, n) for n in graph_.nodes
                      if anchor != n and not graph_.has_edge(anchor, n)
                      and frozenset((anchor, n)) not in excluded]

        if len(candidates) == 0:
            raise ValueError('not enough nonedges are available for rewiring')

        added = candidates[int(rng.choice(len(candidates)))]
        graph_.add_edge(*added)

        info['removed'].append(removed)
        info['added'].append(added)

    return info

categories = {
    'ns_node': 'node',
    'pr_node': 'node',
    'eig_node': 'node',
    'id_node': 'node',
    'rd_node': 'node',
    'ib_node': 'node',
    'rb_node': 'node',
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
    'rd_node': get_node_rd,
    'ib_node': get_node_ib,
    'rb_node': get_node_rb,
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
        self.graph = self.graph_og.copy()

        self.prm.update({
            'attack': attack,
            'attack_approx': None,
            'k_a': 0,

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
            self.attacked = run_attack_method(self.graph_, self.prm['attack'], self.prm['k_a'], approx=self.prm['attack_approx'], seed=self.get_random_seed())

            if get_attack_category(self.prm['attack']) == 'edge':
                self.graph_.remove_edges_from(self.attacked)

        elif self.prm['attack'] is not None:
            print(self.prm['attack'], "not available or k <= 0")

        # defended nodes or edges
        if self.prm['defense'] is not None and self.prm['k_d'] > 0:
            self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.get_random_seed())

        elif self.prm['defense'] is not None:
            print(self.prm['defense'], "not available or k <= 0")

        # remove attacked nodes after checking that they are not defended
        if get_attack_category(self.prm['attack']) == 'node':
            if get_defense_category(self.prm['defense']) == 'node':
                diff = set(self.attacked) - set(self.protected)
                self.graph_.remove_nodes_from(diff)
            else:
                self.graph_.remove_nodes_from(self.attacked)

        self.track_simulation(step=0)

    def track_simulation(self, step):
        """
         Keeps track of important simulation information at each step of the simulation

         :param step: current simulation iteration
         """

        measure = run_measure(self.graph_, self.prm['robust_measure'])

        ccs = list(nx.connected_components(self.graph_))
        ccs.sort(key=len, reverse=True)
        m = interp1d([0, len(ccs)], [0.15, 1]) if len(ccs) > 0 else None

        status = {}
        for n in self.graph:
            status[n] = 0
            for idx, cc in enumerate(ccs):
                if n in self.attacked and n not in self.protected:
                    status[n] = 1
                    break
                elif n in cc:
                    status[n] = float(m(idx))
                    break

        lcc = len(ccs[0]) if len(ccs) > 0 else 0
        self.sim_info[step] = {
            'status':  list(status.values()),
            'failed': len(self.graph) - lcc,
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
            if get_defense_category(self.prm['defense']) == 'edge' and step < len(self.protected['added']):
                if 'removed' in self.protected and step < len(self.protected['removed']):
                    u, v = self.protected['removed'][step]
                    self.graph_.remove_edge(u, v)

                u, v = self.protected['added'][step]
                self.graph_.add_edge(u, v)

            else:
                print("Ending defense simulation early, not an 'edge' defense or out of {}s".format(get_defense_category(self.prm['defense'])))

            self.track_simulation(step + 1)

        results = [self.sim_info[step]['measure'] if self.sim_info[step]['measure'] is not None else 0
                   for step in range(self.prm['steps'] + 1)]
        return results


def main():
    graph = graph_loader(graph_type='ky2', seed=1)

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
