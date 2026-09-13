import random
import numpy as np
from collections import defaultdict

from graph_tiger.simulations import Simulation
from graph_tiger.graphs import *
from graph_tiger.measures import spectral_radius
from graph_tiger.attacks import run_attack_method, get_attack_category
from graph_tiger.defenses import run_defense_method, get_defense_category
from graph_tiger.utils import counter_random, select_backend


class Diffusion(Simulation):
    """
    Simulates the propagation of a virus using either the SIS or SIR model :cite:`kermack1927contribution`.

    :param graph: contact network
    :param model: a string to set the model type (i.e., SIS or SIR)
    :param runs: an integer number of times to run the simulation
    :param steps: an integer number of steps to run a single simulation
    :param b: float representing birth rate of virus (probability of transmitting disease to each neighbor)
    :param d: float representing death rate of virus (probability of each infected node healing)
    :param c: fraction of initially infected nodes
    :param backend: cpu, gpu, or auto for synchronous state transitions
    :param kwargs: see parent class Simulation for additional options
    """

    def __init__(self, graph, model='SIS', runs=10, steps=5000, b=0.00208, d=0.01, c=1, **kwargs):
        super().__init__(graph, runs, steps, **kwargs)

        self.prm.update({
            'model': model,
            'b': b,
            'd': d,
            'c': c,

            'diffusion': None,
            'method': None,
            'k': None,
            'backend': 'cpu',
            'min_gpu_nodes': None
        })

        self.prm.update(kwargs)
        self.validate_parameters()

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

        if self.prm['d'] == 0:
            return np.inf

        return round(spectral_radius(
            self.graph, backend=self.prm['backend'],
            min_gpu_nodes=self.prm['min_gpu_nodes']
        ) * self.prm['b'] / self.prm['d'], 2)

    def reset_simulation(self):
        """
        Resets the simulation between each run
        """

        self.begin_reset()

        self.graph = self.graph_og.copy()
        self.vaccinated = set()
        self.sim_info = defaultdict()

        self.infected = set(self.rng.choice(list(self.graph.nodes), size=int(self.prm['c'] * len(self.graph)), replace=False).tolist())

        # decrease network diffusion
        if self.prm['diffusion'] == 'min' and self.prm['k'] > 0:

            if get_attack_category(self.prm['method']) == 'node':
                self.vaccinated = set(run_attack_method(
                    self.graph, self.prm['method'], self.prm['k'],
                    seed=self.get_random_seed(), backend=self.prm['backend'],
                    min_gpu_nodes=self.prm['min_gpu_nodes']
                ))
                self.infected = self.infected.difference(self.vaccinated)

            elif get_attack_category(self.prm['method']) == 'edge':
                edge_info = run_attack_method(
                    self.graph, self.prm['method'], self.prm['k'],
                    seed=self.get_random_seed(), backend=self.prm['backend'],
                    min_gpu_nodes=self.prm['min_gpu_nodes']
                )
                self.graph.remove_edges_from(edge_info)
            else:
                print(self.prm['method'], 'not available')

        # increase network diffusion
        elif self.prm['diffusion'] == 'max' and self.prm['k'] > 0:

            if get_defense_category(self.prm['method']) == 'edge':
                edge_info = run_defense_method(
                    self.graph, self.prm['method'], self.prm['k'],
                    seed=self.get_random_seed(), backend=self.prm['backend'],
                    min_gpu_nodes=self.prm['min_gpu_nodes']
                )

                self.graph.add_edges_from(edge_info['added'])
                if 'removed' in edge_info:
                    self.graph.remove_edges_from(edge_info['removed'])
            else:
                print(self.prm['method'], 'not available')

        elif self.prm['diffusion'] is not None:
            print(self.prm['diffusion'], "not available or k <= 0")

        self.track_simulation(step=0)

    def validate_parameters(self):
        """
        Validate discrete-time SIS/SIR parameters.
        """

        if self.prm['model'] not in ['SIS', 'SIR']:
            raise ValueError("model must be 'SIS' or 'SIR'")
        if self.prm['b'] < 0 or self.prm['b'] > 1:
            raise ValueError('b must satisfy 0 <= b <= 1')
        if self.prm['d'] < 0 or self.prm['d'] > 1:
            raise ValueError('d must satisfy 0 <= d <= 1')
        if self.prm['c'] < 0 or self.prm['c'] > 1:
            raise ValueError('c must satisfy 0 <= c <= 1')
        if self.prm['runs'] <= 0:
            raise ValueError('runs must be positive')
        if self.prm['steps'] < 0:
            raise ValueError('steps must be nonnegative')
        if self.prm['diffusion'] not in [None, 'min', 'max']:
            raise ValueError("diffusion must be None, 'min', or 'max'")
        if self.prm['diffusion'] is not None and (self.prm['k'] is None or self.prm['k'] < 0):
            raise ValueError('k must be nonnegative when diffusion is requested')
        if self.prm['backend'] not in ['auto', 'cpu', 'gpu']:
            raise ValueError("backend must be one of 'auto', 'cpu', or 'gpu'")
        if (self.prm['min_gpu_nodes'] is not None
                and self.prm['min_gpu_nodes'] < 0):
            raise ValueError('min_gpu_nodes must be nonnegative')

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

    def _edge_arrays(self):
        """Return stable source and target indices for every directed contact."""

        node_order = tuple(self.graph.nodes)
        node_index = {node: index for index, node in enumerate(node_order)}
        sources = []
        targets = []
        for source in node_order:
            for target in self.graph.neighbors(source):
                sources.append(node_index[source])
                targets.append(node_index[target])
        return node_order, np.asarray(sources), np.asarray(targets)

    def _run_array_simulation(self, selected):
        """Run synchronous SIS/SIR transitions with a NumPy or CuPy state."""

        if selected == 'gpu':
            import cupy as xp
        else:
            xp = np

        node_order, sources_host, targets_host = self._edge_arrays()
        sources = xp.asarray(sources_host, dtype=xp.int64)
        targets = xp.asarray(targets_host, dtype=xp.int64)
        infected = xp.asarray([node in self.infected for node in node_order])
        vaccinated = xp.asarray([node in self.vaccinated for node in node_order])
        random_span = len(sources_host) + len(node_order)

        self.track_simulation(step=0)
        for step in range(self.prm['steps']):
            edge_random = counter_random(
                xp, len(sources_host), self._run_seed, step * random_span
            )
            node_random = counter_random(
                xp, len(node_order), self._run_seed,
                step * random_span + len(sources_host)
            )
            successful = (infected[sources] & ~infected[targets]
                          & ~vaccinated[targets]
                          & (edge_random < self.prm['b']))
            infected_new = xp.zeros(len(node_order), dtype=xp.bool_)
            infected_new[targets[successful]] = True
            cured = infected & (node_random < self.prm['d'])
            infected = (infected | infected_new) & ~cured
            if self.prm['model'] == 'SIR':
                vaccinated = vaccinated | cured

            if selected == 'gpu':
                infected_host = xp.asnumpy(infected)
                vaccinated_host = xp.asnumpy(vaccinated)
            else:
                infected_host = infected
                vaccinated_host = vaccinated
            self.infected = {
                node for node, active in zip(node_order, infected_host) if active
            }
            self.vaccinated = {
                node for node, active in zip(node_order, vaccinated_host) if active
            }
            self.track_simulation(step + 1)

    def run_single_sim(self):
        """
        The initially infected nodes are chosen uniformly at random. At each time step,
        every infected-susceptible edge transmits independently with probability 'b'.
        Every node infected at the start of the step recovers with probability 'd' and
        becomes susceptible again in SIS or permanently recovered in SIR. NumPy and
        CuPy use the same counter-based random values for exact seeded parity.
        """

        selected = select_backend(
            self.graph, backend=self.prm['backend'], k=1,
            min_gpu_nodes=self.prm['min_gpu_nodes'],
            operation=self.prm['model'].lower()
        )['selected']
        self._run_array_simulation(selected)

        if self.prm['model'] == 'SIS':
            history = [self.sim_info[step]['failed'] for step in range(self.prm['steps'] + 1)]
        else:
            history = [self.sim_info[step]['recovered'] for step in range(self.prm['steps'] + 1)]

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
