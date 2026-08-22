import numpy as np
from collections import defaultdict

from graph_tiger.simulations import Simulation
from graph_tiger.graphs import *
from graph_tiger.measures import run_measure
from graph_tiger.attacks import run_attack_method, get_attack_category
from graph_tiger.defenses import run_defense_method, get_defense_category


class Cascading(Simulation):
    """
    This class simulates Motter-Lai node-overload cascading failures on a network
    :cite:`motter2002cascade`.

    The initial load of a node is its unnormalized shortest-path betweenness and its
    capacity is ``(1 + r)`` times that initial load. After an attack, loads are
    recomputed on the functioning graph and overloaded nodes fail synchronously.

    The historical TIGER redistribution rule remains available as
    ``model='legacy_redistribution'``. It is a TIGER-specific model and is not the
    Crucitti cascading-failure model.

    :param graph: an undirected NetworkX graph
    :param model: cascading model (``motter_lai`` or ``legacy_redistribution``)
    :param runs: an integer number of times to run the simulation
    :param steps: an integer number of steps to run a single simulation
    :param l: a float representing the maximum initial load in the legacy model
    :param r: a float representing the amount of redundancy in the network
    :param **kwargs: see parent class Simulation for additional options
    """

    def __init__(self, graph, model='motter_lai', runs=10, steps=100, l=0.8, r=0.2, **kwargs):
        super().__init__(graph, runs, steps, **kwargs)

        self.prm.update({
            'model': model,
            'l': l,
            'r': r,
            'c': len(graph),

            'robust_measure': 'largest_connected_component',

            'k_a': 10,
            'attack': 'id_node',
            'attack_approx': None,

            'k_d': 0,
            'defense': None
        })

        self.prm.update(kwargs)
        self.validate_parameters()

        self.graph = self.graph_og.copy()

        if self.prm['plot_transition'] or self.prm['gif_animation']:
            self.node_pos, self.edge_pos = self.get_graph_coordinates()

        self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
        os.makedirs(self.save_dir, exist_ok=True)

        self.capacity_og = self.get_load(self.graph_og)
        self.max_val = max(self.capacity_og.values(), default=0) * (1.0 + self.prm['r'])

        self.protected = set()
        self.failed = set()
        self.failed_edges = set()
        self.processed = set()
        self.load = {}
        self.sim_info = defaultdict()

        self.reset_simulation()

    def validate_parameters(self):
        """
        Validate cascading-model parameters.
        """

        if self.prm['model'] not in ['motter_lai', 'legacy_redistribution']:
            raise ValueError("model must be 'motter_lai' or 'legacy_redistribution'")
        if self.prm['l'] < 0 or self.prm['l'] > 1:
            raise ValueError('l must satisfy 0 <= l <= 1')
        if self.prm['r'] < 0:
            raise ValueError('r must be nonnegative')
        if len(self.graph_og) > 0 and self.prm['c'] is not None and self.prm['c'] <= 0:
            raise ValueError('c must be positive')

    def get_load(self, graph):
        """
        Compute unnormalized shortest-path node load.

        :param graph: functioning NetworkX graph
        :return: dictionary mapping node labels to load
        """

        if len(graph) == 0:
            return {}

        if self.prm['c'] is None or self.prm['c'] >= len(graph):
            return nx.betweenness_centrality(graph, normalized=False, endpoints=False)

        return nx.betweenness_centrality(graph, k=int(self.prm['c']), normalized=False,
                                         endpoints=False, seed=self.prm['seed'])

    def reset_simulation(self):
        """
        Resets the simulation between each run.
        """

        self.graph = self.graph_og.copy()
        self.protected = set()
        self.failed = set()
        self.failed_edges = set()
        self.processed = set()
        self.sim_info = defaultdict()
        self.capacity = {n: (1.0 + self.prm['r']) * value for n, value in self.capacity_og.items()}

        if self.prm['model'] == 'motter_lai':
            self.load = self.capacity_og.copy()
        else:
            self.load = {}
            for n in self.graph.nodes:
                self.load[n] = self.capacity_og[n] * self.rng.uniform(0, self.prm['l'])

        # attacked nodes or edges
        if self.prm['attack'] is not None and self.prm['k_a'] > 0:
            attacked = run_attack_method(self.graph, self.prm['attack'], self.prm['k_a'],
                                         approx=self.prm['attack_approx'], seed=self.prm['seed'])

            if get_attack_category(self.prm['attack']) == 'node':
                self.failed = set(attacked)

            elif get_attack_category(self.prm['attack']) == 'edge':
                self.failed_edges = set(attacked)
                self.graph.remove_edges_from(self.failed_edges)

        elif self.prm['attack'] is not None:
            print(self.prm['attack'], "not available or k <= 0")

        # defended nodes or edges
        if self.prm['defense'] is not None and self.prm['k_d'] > 0:

            if get_defense_category(self.prm['defense']) == 'node':
                self.protected = set(run_defense_method(self.graph, self.prm['defense'],
                                                        self.prm['k_d'], seed=self.prm['seed']))
                for n in self.protected:
                    self.capacity[n] = 2 * self.capacity[n]

            elif get_defense_category(self.prm['defense']) == 'edge':
                edge_info = run_defense_method(self.graph, self.prm['defense'],
                                               self.prm['k_d'], seed=self.prm['seed'])

                if 'removed' in edge_info:
                    self.graph.remove_edges_from(edge_info['removed'])
                self.graph.add_edges_from(edge_info['added'])

        elif self.prm['defense'] is not None:
            print(self.prm['defense'], "not available or k <= 0")

        self.track_simulation(step=0)

    def track_simulation(self, step):
        """
        Keeps track of important simulation information at each step of the simulation.

        :param step: current simulation iteration
        """

        nodes_functioning = set(self.graph.nodes).difference(self.failed)

        measure = 0
        if len(nodes_functioning) > 0:
            measure = run_measure(self.graph.subgraph(nodes_functioning), self.prm['robust_measure'])

        self.sim_info[step] = {
            'status': [self.load.get(n, 0) for n in self.graph_og.nodes],
            'failed': len(self.failed),
            'failed_edges': self.failed_edges,
            'measure': measure,
            'protected': self.protected
        }

    def run_motter_lai_step(self):
        """
        Recompute load and synchronously fail overloaded functioning nodes.

        :return: set of nodes that fail in this step
        """

        nodes_functioning = set(self.graph.nodes).difference(self.failed)
        load = self.get_load(self.graph.subgraph(nodes_functioning).copy())

        self.load = {n: load.get(n, 0) for n in self.graph_og.nodes}
        failed_new = {n for n in nodes_functioning if self.load[n] > self.capacity[n]}
        self.failed.update(failed_new)

        return failed_new

    def run_legacy_step(self):
        """
        Redistribute each newly failed node's load once among functioning neighbors.

        :return: set of nodes that fail in this step
        """

        sources = self.failed.difference(self.processed)

        for n in sources:
            nbrs = set(self.graph.neighbors(n)).difference(self.failed)

            if len(nbrs) > 0:
                share = self.load[n] / len(nbrs)
                for nb in nbrs:
                    self.load[nb] += share

        self.processed.update(sources)

        nodes_functioning = set(self.graph.nodes).difference(self.failed)
        failed_new = {n for n in nodes_functioning if self.load[n] > self.capacity[n]}
        self.failed.update(failed_new)

        return failed_new

    def run_single_sim(self):
        """
        Run the cascading-failure simulation.
        """

        stable = False
        for step in range(self.prm['steps']):
            if not stable:
                if self.prm['model'] == 'motter_lai':
                    failed_new = self.run_motter_lai_step()
                else:
                    failed_new = self.run_legacy_step()

                stable = len(failed_new) == 0

            self.track_simulation(step + 1)

        robustness = [self.sim_info[step]['measure'] for step in range(self.prm['steps'] + 1)]
        return robustness


def main():
    graph = electrical()

    params = {
        'model': 'motter_lai',
        'runs': 1,
        'steps': 100,
        'seed': 1,

        'l': 0.8,
        'r': 0.2,
        'c': int(0.1 * len(graph)),

        'k_a': 5,
        'attack': 'id_node',
        'attack_approx': None,

        'k_d': 0,
        'defense': None,

        'robust_measure': 'largest_connected_component',

        'plot_transition': False,
        'gif_animation': True,
        'gif_snaps': True,

        'edge_style': None,
        'node_style': 'spectral',
        'fa_iter': 2000,
    }

    cf = Cascading(graph, **params)
    results = cf.run_simulation()
    cf.plot_results(results)


if __name__ == '__main__':
    main()
