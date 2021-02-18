import random
import numpy as np
from collections import defaultdict

from graph_tiger.simulations import Simulation
from graph_tiger.graphs import *
from graph_tiger.measures import spectral_radius
from graph_tiger.attacks import run_attack_method, get_attack_category
from graph_tiger.defenses import run_defense_method, get_defense_category


class Diffusion(Simulation):
    def __init__(self, graph, model='SIS', runs=10, steps=5000, b=0.00208, d=0.01, c=1, **kwargs):
        """
        Simulates the propagation of a virus using either the SIS or SIR model

        :param graph: contact network
        :param model: a string to set the model type (i.e., SIS or SIR)
        :param runs: an integer number of times to run the simulation
        :param steps: an integer number of steps to run a single simulation
        :param b: float representing birth rate of virus (probability of transmitting disease to each neighbor)
        :param d: float representing death rate of virus (probability of each infected node healing)
        :param c: fraction of initially infected nodes
        :param **kwargs: see parent class Simulation for additional options
        """
        super().__init__(graph, runs, steps, **kwargs)

        self.prm.update({
            'model': model,
            'b': b,
            'd': d,
            'c': c,

            'diffusion': None,
            'method': None,
            'k': None
        })

        self.prm.update(kwargs)

        self.vaccinated = set()
        self.infected = set()

        if self.prm['plot_transition'] or self.prm['gif_animation']:
            self.node_pos, self.edge_pos = self.get_graph_coordinates()

        self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
        os.makedirs(self.save_dir, exist_ok=True)

        self.reset_simulation()

    def get_effective_strength(self):
        """
        Gets the effective string of the virus. This is a factor of the spectral radius (first eigenvalue) of graph,
        the virus birth rate 'b' and the virus death rate 'd'

        :return: a float for virus effective strength
        """

        return round(spectral_radius(self.graph) * self.prm['b'] / self.prm['d'], 2)

    def reset_simulation(self):
        """
        Resets the simulation between each run
        """

        self.vaccinated = set()
        self.sim_info = defaultdict()

        self.infected = set(np.random.choice(list(self.graph.nodes), size=int(self.prm['c'] * len(self.graph)), replace=False).tolist())

        # decrease network diffusion
        if self.prm['diffusion'] == 'min' and self.prm['k'] > 0:

            if get_attack_category(self.prm['method']) == 'node':
                self.vaccinated = set(run_attack_method(self.graph, self.prm['method'], self.prm['k'], seed=self.prm['seed']))
                self.infected = self.infected.difference(self.vaccinated)

            elif get_attack_category(self.prm['method']) == 'edge':
                edge_info = run_attack_method(self.graph, self.prm['method'], self.prm['k'], seed=self.prm['seed'])
                self.graph.remove_edges_from(edge_info)
            else:
                print(self.prm['method'], 'not available')

        # increase network diffusion
        elif self.prm['diffusion'] == 'max' and self.prm['k'] > 0:

            if get_defense_category(self.prm['method']) == 'edge':
                edge_info = run_defense_method(self.graph, self.prm['method'], self.prm['k'], seed=self.prm['seed'])

                self.graph.add_edges_from(edge_info['added'])
                self.graph.remove_edges_from(edge_info['removed'])
            else:
                print(self.prm['method'], 'not available')

        elif self.prm['diffusion'] is not None:
            print(self.prm['diffusion'], "not available or k <= 0")

    def track_simulation(self, step):
        """
          Keeps track of important simulation information at each step of the simulation

          :param step: current simulation iteration
          """

        self.sim_info[step] = {
            'status': [1 if n in self.infected else 0 for n in self.graph.nodes],
            'failed': len(self.infected),
            'recovered': len(self.vaccinated),
            'protected': self.vaccinated
        }

    def run_single_sim(self):
        """
        The initially infected nodes are chosen uniformly at random. At each time step,
        every susceptible (i.e., non-infected) node has a probability 'b' of being
        infected by neighboring infected nodes. Every infected node has a probability 'd'
        of being cured and becoming susceptible again (or recovered for SIR model).
        """

        for step in range(self.prm['steps']):
            self.track_simulation(step)

            infected_new = set()
            for node in self.infected:
                nbrs = self.graph.neighbors(node)
                nbrs = set(nbrs).difference(self.infected).difference(self.vaccinated)

                nbrs_infected = set([n for n in nbrs if random.random() <= self.prm['b']])
                infected_new = infected_new.union(nbrs_infected)

            cured = set([n for n in self.infected if random.random() <= self.prm['d']])

            self.infected = self.infected.union(infected_new)  # infect
            self.infected = self.infected.difference(cured)  # cure

            if self.prm['model'] == 'SIR':
                self.vaccinated.update(cured)

        if self.prm['model'] == 'SIS':
            history = [v['failed'] for k, v in self.sim_info.items()]
        else:
            history = [v['recovered'] for k, v in self.sim_info.items()]

        return history


def main():
    graph = as_733()

    sis_params = {
        'model': 'SIS',
        'b': 0.001,  # karate=0.00208
        'd': 0.01,
        'c': 1,

        'runs': 1,
        'steps': 5000,
        'seed': 1,

        'diffusion': 'min',
        'method': 'ns_node',
        'k': 5,

        'plot_transition': True,
        'gif_animation': True,

        'edge_style': 'bundled',
        'node_style': 'force_atlas',
        'fa_iter': 20
    }

    ds = Diffusion(graph, **sis_params)
    results = ds.run_simulation()
    ds.plot_results(results)

    # sir_params = {
    #     'model': 'SIR',
    #     'b': 0.00208,
    #     'd': 0.01,
    #     'c': 0.1,
    #     'runs': 10,
    #     'steps': 5000,
    #
    #     'diffusion': 'min',
    #     'method': 'add_edge_random',
    #     'k': 15,
    #
    #     'seed': 1,
    #     'fast_plot': False,
    #     'plot_transition': True,
    #     'gif_animation': True
    # }

    # ds = Diffusion(graph, sir_params)
    # results = ds.run_simulation()
    # ds.plot_results(results)


if __name__ == '__main__':
    main()