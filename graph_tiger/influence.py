import os
from collections import Counter, defaultdict
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from graph_tiger.simulations import Simulation


class Influence(Simulation):
    """
    Simulate progressive or reversible influence on a NetworkX graph.

    Independent-cascade and linear-threshold conventions follow :cite:`kempe2003maximizing`.
    The voter model follows :cite:`clifford1973model`; the competitive process is
    based on :cite:`bharathi2007competitive` with an explicit same-round tie rule.

    :param graph: simple NetworkX graph or directed graph
    :param model: independent_cascade, linear_threshold, voter, or competitive_cascade
    :param runs: number of simulation realizations
    :param steps: number of synchronous rounds or asynchronous voter updates
    :param seeds: initially active nodes for independent cascade or linear threshold
    :param probability: scalar probability, edge-attribute name, or message mapping
    :param weight: edge attribute used by linear threshold; missing values default to one
    :param threshold: scalar, node-attribute name, or node-to-threshold mapping
    :param initial_state: complete node-to-state mapping for the voter model
    :param message_seeds: message-to-seed-set mapping for competitive cascade
    :param tracked_state: state or message counted in the returned trajectory
    :param tie_break: random or priority for simultaneous competitive-cascade arrivals
    :param priority: message order used when tie_break is priority
    :param kwargs: see parent class Simulation for random seed and plotting options
    """

    models = {
        'independent_cascade',
        'linear_threshold',
        'voter',
        'competitive_cascade'
    }

    def __init__(self, graph, model='independent_cascade', runs=10, steps=100,
                 seeds=None, probability=0.1, weight='weight', threshold='threshold',
                 initial_state=None, message_seeds=None, tracked_state=1,
                 tie_break='random', priority=None, **kwargs):
        super().__init__(graph, runs, steps, **kwargs)

        self.prm.update({
            'model': model,
            'seeds': set() if seeds is None else set(seeds),
            'probability': probability,
            'weight': weight,
            'threshold': threshold,
            'initial_state': None if initial_state is None else dict(initial_state),
            'message_seeds': None if message_seeds is None else {
                message: set(nodes) for message, nodes in message_seeds.items()
            },
            'tracked_state': tracked_state,
            'tie_break': tie_break,
            'priority': None if priority is None else tuple(priority)
        })
        self.prm.update(kwargs)

        self.node_order = tuple(self.graph_og.nodes)
        self.node_index = {node: index for index, node in enumerate(self.node_order)}
        self.validate_parameters()

        if self.prm['plot_transition'] or self.prm['gif_animation']:
            self.node_pos, self.edge_pos = self.get_graph_coordinates()

        self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
        os.makedirs(self.save_dir, exist_ok=True)
        self.reset_simulation()

    def validate_parameters(self):
        """Validate graph, model, initial-state, and model-specific parameters."""
        if self.prm['model'] not in self.models:
            raise ValueError('unknown influence model')
        if self.graph_og.is_multigraph():
            raise ValueError('influence models require a simple graph')

        nodes = set(self.graph_og.nodes)
        model = self.prm['model']

        if model in {'independent_cascade', 'linear_threshold'}:
            if not self.prm['seeds'].issubset(nodes):
                raise ValueError('seeds must belong to the graph')

        if model == 'voter':
            state = self.prm['initial_state']
            if state is None or set(state) != nodes:
                raise ValueError('initial_state must assign every graph node')
            if self.graph_og.is_directed():
                raise ValueError('the voter model currently requires an undirected graph')
            try:
                state_values = set(state.values())
            except TypeError as error:
                raise ValueError('voter states must be hashable') from error
            if self.prm['tracked_state'] not in state_values:
                raise ValueError('tracked_state must occur in initial_state')

        if model == 'competitive_cascade':
            message_seeds = self.prm['message_seeds']
            if not message_seeds:
                raise ValueError('message_seeds must contain at least one message')
            assigned = set()
            for seeds in message_seeds.values():
                if not seeds.issubset(nodes):
                    raise ValueError('message seeds must belong to the graph')
                if assigned.intersection(seeds):
                    raise ValueError('message seed sets must be disjoint')
                assigned.update(seeds)
            if self.prm['tracked_state'] not in message_seeds:
                raise ValueError('tracked_state must name a competitive message')
            if self.prm['tie_break'] not in {'random', 'priority'}:
                raise ValueError("tie_break must be 'random' or 'priority'")
            if self.prm['tie_break'] == 'priority':
                if set(self.prm['priority'] or ()) != set(message_seeds):
                    raise ValueError('priority must contain every message exactly once')

        if model in {'independent_cascade', 'competitive_cascade'}:
            self._validate_probabilities()
        if model == 'linear_threshold':
            self._validate_linear_threshold()

    def _validate_probabilities(self):
        probability = self.prm['probability']
        if self.prm['model'] == 'competitive_cascade' and isinstance(probability, dict):
            if set(probability) != set(self.prm['message_seeds']):
                raise ValueError('probability mapping must contain every message')
            values = probability.values()
        else:
            values = [probability]

        for value in values:
            if isinstance(value, str):
                for _, _, data in self.graph_og.edges(data=True):
                    edge_probability = data.get(value)
                    if (not isinstance(edge_probability, Real)
                            or not np.isfinite(edge_probability)
                            or not 0 <= edge_probability <= 1):
                        raise ValueError('edge probabilities must be in [0, 1]')
            elif (not isinstance(value, Real) or not np.isfinite(value)
                  or not 0 <= value <= 1):
                raise ValueError('probability must be in [0, 1]')

    def _validate_linear_threshold(self):
        weight = self.prm['weight']
        if weight is not None and not isinstance(weight, str):
            raise ValueError('weight must be an edge-attribute name or None')
        if weight is not None:
            for _, _, data in self.graph_og.edges(data=True):
                value = data.get(weight, 1.0)
                if (not isinstance(value, Real) or not np.isfinite(value)
                        or value < 0):
                    raise ValueError('linear-threshold weights must be nonnegative')

        threshold = self.prm['threshold']
        if isinstance(threshold, dict):
            if set(threshold) != set(self.graph_og.nodes):
                raise ValueError('threshold mapping must contain every graph node')
            values = threshold.values()
        elif isinstance(threshold, str):
            values = []
            for _, data in self.graph_og.nodes(data=True):
                if threshold not in data:
                    raise ValueError('every node must have the threshold attribute')
                values.append(data[threshold])
        else:
            values = [threshold]
        if any(not isinstance(value, Real) or not np.isfinite(value) or value <= 0
               for value in values):
            raise ValueError('linear-threshold values must be positive')

    def reset_simulation(self):
        """Reset graph state and advance to the next reproducible random stream."""
        self.begin_reset()
        self.graph = self.graph_og.copy()
        self.sim_info = defaultdict()
        self.changed = set()
        self.influence = dict.fromkeys(self.node_order, 0.0)

        model = self.prm['model']
        if model in {'independent_cascade', 'linear_threshold'}:
            self.active = set(self.prm['seeds'])
            self.frontier = set(self.active)
            self.state = {node: int(node in self.active) for node in self.node_order}
            self.changed = set(self.active)
            self.state_values = [0, 1]
        elif model == 'voter':
            self.state = dict(self.prm['initial_state'])
            self.state_values = list(dict.fromkeys(self.state.values()))
        else:
            self.messages = tuple(self.prm['message_seeds'])
            self.state = dict.fromkeys(self.node_order)
            self.frontiers = {}
            for message, seeds in self.prm['message_seeds'].items():
                self.frontiers[message] = set(seeds)
                for node in seeds:
                    self.state[node] = message
                    self.changed.add(node)
            self.state_values = [None, *self.messages]

        self.prm['max_val'] = max(1, len(self.state_values) - 1)
        self.track_simulation(0)

    def _ordered(self, nodes):
        return sorted(nodes, key=self.node_index.__getitem__)

    def _targets(self, node):
        if self.graph.is_directed():
            return self._ordered(self.graph.successors(node))
        return self._ordered(self.graph.neighbors(node))

    def _edge_probability(self, source, target, message=None):
        probability = self.prm['probability']
        if isinstance(probability, dict):
            probability = probability[message]
        if isinstance(probability, str):
            return self.graph[source][target][probability]
        return probability

    def _threshold(self, node):
        threshold = self.prm['threshold']
        if isinstance(threshold, dict):
            return threshold[node]
        if isinstance(threshold, str):
            return self.graph.nodes[node][threshold]
        return threshold

    def run_independent_cascade_step(self):
        """Advance one synchronous independent-cascade frontier.

        :return: set of nodes activated in this round
        """
        activated = set()
        for source in self._ordered(self.frontier):
            for target in self._targets(source):
                if target in self.active:
                    continue
                if self.random.random() < self._edge_probability(source, target):
                    activated.add(target)
        self.active.update(activated)
        self.frontier = activated
        for node in activated:
            self.state[node] = 1
        return activated

    def run_linear_threshold_step(self):
        """Advance one synchronous linear-threshold frontier.

        :return: set of nodes activated in this round
        """
        for source in self._ordered(self.frontier):
            for target in self._targets(source):
                if target in self.active:
                    continue
                self.influence[target] += self.graph[source][target].get(
                    self.prm['weight'], 1.0
                ) if self.prm['weight'] is not None else 1.0

        activated = {
            node for node in self.node_order
            if node not in self.active and self.influence[node] >= self._threshold(node)
        }
        self.active.update(activated)
        self.frontier = activated
        for node in activated:
            self.state[node] = 1
        return activated

    def run_voter_step(self):
        """Perform one asynchronous node-copying event.

        :return: the changed node as a set, or an empty set
        """
        if not self.node_order:
            return set()
        node = self.random.choice(self.node_order)
        neighbors = tuple(self.graph.neighbors(node))
        if not neighbors:
            return set()
        source = self.random.choice(neighbors)
        if self.state[node] == self.state[source]:
            return set()
        self.state[node] = self.state[source]
        return {node}

    def run_competitive_cascade_step(self):
        """Advance all competitive-message frontiers by one synchronous round.

        :return: set of nodes that received at least one successful proposal
        """
        proposals = defaultdict(set)
        for message in self.messages:
            for source in self._ordered(self.frontiers[message]):
                for target in self._targets(source):
                    if self.state[target] is not None:
                        continue
                    if self.random.random() < self._edge_probability(source, target, message):
                        proposals[target].add(message)

        next_frontiers = {message: set() for message in self.messages}
        for target in self._ordered(proposals):
            candidates = proposals[target]
            if len(candidates) == 1:
                message = next(iter(candidates))
            elif self.prm['tie_break'] == 'priority':
                message = next(item for item in self.prm['priority'] if item in candidates)
            else:
                message = self.random.choice([
                    item for item in self.messages if item in candidates
                ])
            self.state[target] = message
            next_frontiers[message].add(target)

        self.frontiers = next_frontiers
        return set(proposals)

    def _absorbed(self):
        model = self.prm['model']
        if model in {'independent_cascade', 'linear_threshold'}:
            return not self.frontier
        if model == 'competitive_cascade':
            return not any(self.frontiers.values())
        return not any(
            self.state[source] != self.state[target]
            for source, target in self.graph.edges
        )

    def track_simulation(self, step):
        """Store node states, counts, changes, and frontiers for one step."""
        counts = Counter(self.state.values())
        model = self.prm['model']
        if model in {'independent_cascade', 'linear_threshold'}:
            active = len(self.active)
            frontier = set(self.frontier)
        elif model == 'competitive_cascade':
            active = len(self.graph) - counts.get(None, 0)
            frontier = {
                message: set(nodes) for message, nodes in self.frontiers.items()
            }
        else:
            active = counts.get(self.prm['tracked_state'], 0)
            frontier = set()

        self.sim_info[step] = {
            'status': [self.state[node] for node in self.node_order],
            'counts': dict(counts),
            'changed': set(self.changed),
            'frontier': frontier,
            'active': active,
            'tracked': counts.get(self.prm['tracked_state'], 0)
        }

    def run_single_sim(self):
        """Run one realization for the configured number of steps.

        :return: active counts for progressive models or tracked-state counts otherwise
        """
        methods = {
            'independent_cascade': self.run_independent_cascade_step,
            'linear_threshold': self.run_linear_threshold_step,
            'voter': self.run_voter_step,
            'competitive_cascade': self.run_competitive_cascade_step
        }

        for step in range(self.prm['steps']):
            self.changed = set() if self._absorbed() else methods[self.prm['model']]()
            self.track_simulation(step + 1)

        if self.prm['model'] in {'independent_cascade', 'linear_threshold'}:
            field = 'active'
        else:
            field = 'tracked'
        return [self.sim_info[step][field] for step in range(self.prm['steps'] + 1)]

    def get_plot_title(self, step):
        """Return a stable filename stem for an influence plot."""
        return 'Influence--model={},step={}'.format(self.prm['model'], step)

    def plot_results(self, results):
        """Plot a normalized active-state or tracked-state trajectory."""
        values = np.asarray(results, dtype=float)
        if len(self.graph_og):
            values /= len(self.graph_og)
        plt.figure(figsize=(6.4, 4.8))
        plt.plot(values)
        plt.xlabel('Steps')
        plt.ylabel('Active fraction' if self.prm['model'] in {
            'independent_cascade', 'linear_threshold'
        } else 'Tracked-state fraction')
        plt.ylim(0, 1)
        plt.title(self.prm['model'].replace('_', ' ').title())
        plt.savefig(os.path.join(self.save_dir, self.get_plot_title(self.prm['steps']) + '_results.pdf'))
        plt.clf()

    def get_visual_settings(self, step):
        """Return node and edge styles for a stored influence state."""
        colors = ['#d9e0e3', '#cf6636', '#5765b0', '#157a79', '#a65f9e']
        if len(self.state_values) > len(colors):
            tab20 = plt.get_cmap('tab20')
            colors.extend(
                tab20(index % tab20.N)
                for index in range(len(self.state_values) - len(colors))
            )
        state_index = {state: index for index, state in enumerate(self.state_values)}
        node_colors = np.asarray([
            state_index[state] for state in self.sim_info[step]['status']
        ])
        node_sizes = np.asarray([
            120 if node in self.sim_info[step]['changed'] else 45
            for node in self.node_order
        ])
        cmap = ListedColormap(colors[:len(self.state_values)])
        return node_colors, node_sizes, '#9aa8ae', 1, cmap

    def plot_graph_transition(self, sim_info):
        """Save representative snapshots from one influence realization."""
        steps = sorted(sim_info)
        if not steps:
            return
        selected = {steps[0], steps[-1]}
        selected.update(steps[1:3])
        selected.add(steps[len(steps) // 2])
        for step in sorted(selected):
            self.plot_network(step)
