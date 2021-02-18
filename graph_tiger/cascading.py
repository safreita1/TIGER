import numpy as np
from collections import defaultdict

from graph_tiger.simulations import Simulation
from graph_tiger.graphs import *
from graph_tiger.measures import run_measure
from graph_tiger.attacks import run_attack_method, get_attack_category
from graph_tiger.defenses import run_defense_method, get_defense_category


class Cascading(Simulation):
    """
    This class simulates cascading failures on a network

    :param graph: an undirected NetworkX graph
    :param runs: an integer number of times to run the simulation
    :param steps: an integer number of steps to run a single simulation
    :param l: a float representing the maximum initial load for each node
    :param r: a float representing the amount of redundancy in the network
    :param **kwargs: see parent class Simulation for additional options
    """

    def __init__(self, graph, runs=10, steps=100, l=0.8, r=0.2, **kwargs):
        super().__init__(graph, runs, steps, **kwargs)

        self.prm.update({
            'l': l,
            'r': r,
            'c': len(graph),

            'robust_measure': 'largest_connected_component',

            'k_a': 10,
            'attack': 'id_node',
            'attack_approx': None,

            'k_d': None,
            'defense': None
        })

        self.prm.update(kwargs)

        if self.prm['plot_transition'] or self.prm['gif_animation']:
            self.node_pos, self.edge_pos = self.get_graph_coordinates()

        self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
        os.makedirs(self.save_dir, exist_ok=True)

        self.capacity_og = nx.betweenness_centrality(self.graph, k=self.prm['c'], normalized=True, endpoints=True)
        self.max_val = max(self.capacity_og.values()) * (1.0 + self.prm['r'])

        self.protected = set()
        self.failed = set()
        self.load = defaultdict()
        self.sim_info = defaultdict()

        self.reset_simulation()

    def reset_simulation(self):
        """
         Resets the simulation between each run
         """

        self.protected = set()
        self.failed = set()
        self.load = defaultdict()
        self.sim_info = defaultdict()
        self.capacity = self.capacity_og.copy()

        for n in self.graph.nodes:
            self.load[n] = self.capacity[n] * np.random.uniform(0, self.prm['l'])  # self.capacity[n] * np.random.uniform(0, self.prm['l'])  # self.capacity[n] *
            self.capacity[n] = self.capacity[n] * (1.0 + self.prm['r'])

        self.track_simulation(step=0)

        # attacked nodes or edges
        if self.prm['attack'] is not None and self.prm['k_a'] > 0:
            self.failed = set(run_attack_method(self.graph, self.prm['attack'], self.prm['k_a'],
                                                approx=self.prm['attack_approx'], seed=self.prm['seed']))

            if get_attack_category(self.prm['attack']) == 'node':
                for n in self.failed:
                    self.load[n] = 2 * self.load[n]  # increase load by 2x when attacked

            elif get_attack_category(self.prm['attack']) == 'edge':
                self.graph.remove_edges_from(self.failed)

        # defended nodes or edges
        if self.prm['defense'] is not None and self.prm['k_d'] > 0:

            if get_defense_category(self.prm['defense']) == 'node':
                self.protected = run_defense_method(self.graph, self.prm['defense'], self.prm['k_d'], seed=self.prm['seed'])
                for n in self.protected:
                    self.capacity[n] = 2 * self.capacity[n]  # double the capacity when defended

            elif get_defense_category(self.prm['defense']) == 'edge':
                edge_info = run_defense_method(self.graph, self.prm['defense'], self.prm['k_d'], seed=self.prm['seed'])

                self.graph.add_edges_from(edge_info['added'])

                if 'removed' in edge_info:
                    self.graph.remove_edges_from(edge_info['removed'])

        elif self.prm['defense'] is not None:
            print(self.prm['defense'], "not available or k <= 0")

        self.track_simulation(step=1)

    def track_simulation(self, step):
        """
        Keeps track of important simulation information at each step of the simulation

        :param step: current simulation iteration
        """

        nodes_functioning = set(self.graph.nodes).difference(self.failed)

        measure = 0
        if len(nodes_functioning) > 0:
            measure = run_measure(self.graph.subgraph(nodes_functioning), self.prm['robust_measure'])

        self.sim_info[step] = {
            'status': [self.load[n] for n in self.graph.nodes],
            'failed': len(self.failed),
            'measure': measure,
            'protected': self.protected
        }

    def run_single_sim(self):
        """
        Run the attack simulation
        """

        for step in range(self.prm['steps']):
            self.track_simulation(step+2)

            failed_new = set()
            for n in self.failed:
                if self.load[n] > self.capacity[n]:
                    nbrs = list(self.graph.neighbors(n))

                    for nb in self.graph.neighbors(n):

                        if nb not in self.failed and nb not in failed_new:
                            self.load[nb] += self.load[n] / len(nbrs)

                            if self.load[nb] > self.capacity[nb]:
                                failed_new.add(nb)

            self.failed = self.failed.union(failed_new)

        robustness = [v['measure'] if v['measure'] is not None else 0 for k, v in self.sim_info.items()]
        return robustness


def main():
    graph = electrical()

    params = {
        'runs': 1,
        'steps': 100,
        'seed': 1,

        'l': 0.8,
        'r': 0.2,
        'c': int(0.1 * len(graph)),

        'k_a': 5,
        'attack': 'id_node',
        'attack_approx': None,  # int(0.1 * len(graph)),
        
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
