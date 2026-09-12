graph_tiger.cascading
=====================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.7.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-cascading"><details class="api-entry" id="api-cascading-Cascading"><summary><code>Cascading(graph, model='motter_lai', runs=10, steps=100, l=0.8, r=0.2, beta=0, allocation='degree', initial_load=None, capacities=None, initial_failures=None, **kwargs)</code></summary><div class="api-body"><p>Overload simulation supporting global shortest-path rerouting, edge-efficiency degradation, and local load sharing. See the cascading-failure guide for equations and stopping rules.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>Undirected simple NetworkX graph. See Network inputs for node ordering, conversions, and weight conventions.</td></tr><tr><td><code>model</code></td><td>motter_lai (default), crucitti, local_load_sharing, or legacy_redistribution.</td></tr><tr><td><code>runs</code></td><td>Positive integer; number of realizations for run_simulation, default 10 in simulation subclasses.</td></tr><tr><td><code>steps</code></td><td>Nonnegative transition limit. Returned trajectories have steps+1 entries, including index 0.</td></tr><tr><td><code>l</code></td><td>Legacy initial-load upper fraction, in [0,1], default 0.8. Not used by the three other cascade models.</td></tr><tr><td><code>r</code></td><td>Nonnegative spare-capacity fraction; capacity is (1+r) times initial load. Must be positive for Crucitti. Default 0.2.</td></tr><tr><td><code>beta</code></td><td>Nonnegative intact-degree preference exponent, default 0; used only by local_load_sharing.</td></tr><tr><td><code>allocation</code></td><td>local policy: degree (default), greedy, proportional, or
       max_flow. Greedy ties follow graph node insertion order. Maximum flow
       uses Edmonds-Karp with that order; ties can affect subsequent failures.</td></tr><tr><td><code>initial_load</code></td><td>optional complete node-to-load mapping for local sharing;
       finite nonnegative values, default intact unweighted degrees</td></tr><tr><td><code>capacities</code></td><td>optional complete node-to-capacity mapping for local sharing;
       finite nonnegative values, default (1+r) times initial load. Explicit
       capacities are used directly, without another redundancy multiplier.</td></tr><tr><td><code>initial_failures</code></td><td>optional iterable of initially failed nodes for local
       sharing; replaces attack selection and its budget, including when empty</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization.</td></tr><tr><td><code>plot_transition</code></td><td>Boolean, default False. Save selected first-run snapshots.</td></tr><tr><td><code>gif_animation</code></td><td>Boolean, default False. Write an MP4 with FFmpeg on supported non-Windows platforms.</td></tr><tr><td><code>gif_snaps</code></td><td>Boolean, default False. Save animation frames when that workflow runs.</td></tr><tr><td><code>node_style</code></td><td>None (dataset coordinates or spectral layout) or force_atlas (optional dependency).</td></tr><tr><td><code>edge_style</code></td><td>None (straight edges) or bundled (optional dependency).</td></tr><tr><td><code>fa_iter</code></td><td>Positive integer, default 200; ForceAtlas2 iterations.</td></tr><tr><td><code>robust_measure</code></td><td>Measure key; default largest_connected_component for attack, defense and non-Crucitti cascades. Crucitti overrides it with weighted efficiency.</td></tr><tr><td><code>attack</code></td><td>Attack-selection key or None; simulation subclasses default to id_node. Budgets determine whether an initial attack is performed.</td></tr><tr><td><code>attack_approx</code></td><td>Sampled source-node count for betweenness attack selection; default None means exact. Not a fraction or edge count.</td></tr><tr><td><code>k_a</code></td><td>Nonnegative initial attack budget. Default 10 for Cascading and 0 for Defense; must not exceed available components.</td></tr><tr><td><code>defense</code></td><td>None or a supported defense key. Cascade node defense doubles selected capacities, but does not cancel initial failures.</td></tr><tr><td><code>k_d</code></td><td>Nonnegative node/edge defense budget, default 0. Attack: number selected for protection/intervention. Defense: number of proposed edge changes. Cascading: defense budget.</td></tr><tr><td><code>c</code></td><td>Cascading: number of source nodes for load approximation, default graph order (exact). Diffusion: initially infected fraction in [0,1], default 1.</td></tr></tbody></table></div><p><strong>Returns</strong> A configured Cascading instance. Construction initializes the graph, state and random generators.</p><p><a href="reproducibility.html">Output shapes, state lifecycle, stopping and random-seed conventions</a></p><p>References: <a href="references.html#ref-motter2002cascade">Cascade-based attacks on complex networks (2002)</a>; <a href="references.html#ref-crucitti2004model">Model for cascading failures in complex networks (2004)</a>; <a href="references.html#ref-wei2012analysis">Analysis of cascading failure in complex power networks under the load local preferential redistribution rule (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>class Cascading(Simulation):
       &quot;&quot;&quot;
       This class simulates cascading failures on a network.

       The default Motter-Lai model uses unnormalized shortest-path betweenness as
       node load and fixes capacity at ``(1 + r)`` times the initial load. After an
       attack, overloaded nodes fail synchronously :cite:`motter2002cascade`.

       In the Crucitti model, ``1 + r`` is the paper's tolerance parameter. An
       overloaded node remains in the network while the efficiencies of its incident
       edges decrease by its capacity-to-load ratio. Traffic is synchronously rerouted
       over the most efficient weighted paths and results report average network
       efficiency :cite:`crucitti2004model`.

       The local-load-sharing model gives node ``i`` initial load equal to its degree
       and fixed capacity ``(1 + r) L_i(0)``. A failed node redistributes its complete
       current load among functioning neighbors with weights proportional to
       ``degree ** beta``. Thus ``beta=0`` gives equal sharing, while larger values
       increasingly favor high-degree recipients :cite:`wei2012analysis`.
       Alternatively, ``allocation`` selects greedy or proportional spare-capacity
       sharing, or coordinated maximum flow. Every policy transfers the full load
       when a recipient exists. Only work with no recipient becomes lost service.

       The historical TIGER redistribution rule remains available as
       ``model='legacy_redistribution'``. It is a TIGER-specific model.

       :param graph: an undirected NetworkX graph
       :param model: cascading model (``motter_lai``, ``crucitti``, ``local_load_sharing``,
           or ``legacy_redistribution``)
       :param runs: an integer number of times to run the simulation
       :param steps: an integer number of steps to run a single simulation
       :param l: a float representing the maximum initial load in the legacy model
       :param r: a float representing the amount of redundancy in the network
       :param beta: a nonnegative degree-preference exponent for local load sharing
       :param allocation: local policy: degree (default), greedy, proportional, or
           max_flow. Greedy ties follow graph node insertion order. Maximum flow
           uses Edmonds-Karp with that order; ties can affect subsequent failures.
       :param initial_load: optional complete node-to-load mapping for local sharing;
           finite nonnegative values, default intact unweighted degrees
       :param capacities: optional complete node-to-capacity mapping for local sharing;
           finite nonnegative values, default (1+r) times initial load. Explicit
           capacities are used directly, without another redundancy multiplier.
       :param initial_failures: optional iterable of initially failed nodes for local
           sharing; replaces attack selection and its budget, including when empty
       :param kwargs: see parent class Simulation for additional options
       &quot;&quot;&quot;

       def __init__(self, graph, model='motter_lai', runs=10, steps=100, l=0.8, r=0.2,
                    beta=0, allocation='degree', initial_load=None, capacities=None,
                    initial_failures=None, **kwargs):
           super().__init__(graph, runs, steps, **kwargs)

           self.prm.update({
               'model': model,
               'l': l,
               'r': r,
               'beta': beta,
               'allocation': allocation,
               'initial_load': initial_load,
               'capacities': capacities,
               'initial_failures': initial_failures,
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

           if self.prm['model'] == 'crucitti':
               self.prm['robust_measure'] = 'network_efficiency'

           self.graph = self.graph_og.copy()

           if self.prm['plot_transition'] or self.prm['gif_animation']:
               self.node_pos, self.edge_pos = self.get_graph_coordinates()

           self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
           os.makedirs(self.save_dir, exist_ok=True)

           if self.prm['model'] == 'local_load_sharing':
               self.capacity_og = (dict(self.graph_og.degree()) if self.prm['initial_load'] is None
                                   else self.prm['initial_load'].copy())
           else:
               self.capacity_og = self.get_load(self.graph_og)
           self.capacity_initial = {n: (1.0 + self.prm['r']) * value
                                    for n, value in self.capacity_og.items()}
           if self.prm['capacities'] is not None:
               self.capacity_initial = self.prm['capacities'].copy()
           self.max_val = max(self.capacity_initial.values(), default=0)
           self.prm['max_val'] = self.max_val

           self.protected = set()
           self.failed = set()
           self.failed_edges = set()
           self.processed = set()
           self.overloaded = set()
           self.shed_load = 0
           self.load = {}
           self.sim_info = defaultdict()

           self.reset_simulation()

       def validate_parameters(self):
           &quot;&quot;&quot;
           Validate cascading-model parameters.
           &quot;&quot;&quot;

           models = ['motter_lai', 'crucitti', 'local_load_sharing', 'legacy_redistribution']
           if self.prm['model'] not in models:
               raise ValueError('unknown cascading model')
           if self.prm['l'] &lt; 0 or self.prm['l'] &gt; 1:
               raise ValueError('l must satisfy 0 &lt;= l &lt;= 1')
           if not np.isfinite(self.prm['r']) or self.prm['r'] &lt; 0:
               raise ValueError('r must be nonnegative')
           if not np.isfinite(self.prm['beta']) or self.prm['beta'] &lt; 0:
               raise ValueError('beta must be nonnegative')
           if self.prm['allocation'] not in ['degree', 'greedy', 'proportional', 'max_flow']:
               raise ValueError('unknown local allocation policy')
           local_options = ['initial_load', 'capacities', 'initial_failures']
           if self.prm['model'] != 'local_load_sharing':
               if self.prm['allocation'] != 'degree' or any(self.prm[k] is not None for k in local_options):
                   raise ValueError('allocation and supplied load/capacity/failure inputs require local_load_sharing')
           for key in ['initial_load', 'capacities']:
               values = self.prm[key]
               if values is not None:
                   if not isinstance(values, Mapping) or set(values) != set(self.graph_og):
                       raise ValueError(key + ' must map every graph node exactly once')
                   if any(not isinstance(v, Real) or not np.isfinite(v) or v &lt; 0 for v in values.values()):
                       raise ValueError(key + ' must contain finite nonnegative numbers')
                   self.prm[key] = {n: float(values[n]) for n in self.graph_og}
           if self.prm['initial_failures'] is not None:
               if isinstance(self.prm['initial_failures'], (str, bytes)):
                   raise ValueError('initial_failures must be an iterable of node labels')
               failed = set(self.prm['initial_failures'])
               if not failed.issubset(self.graph_og):
                   raise ValueError('initial_failures contains unknown nodes')
               self.prm['initial_failures'] = tuple(n for n in self.graph_og if n in failed)
           if self.prm['model'] == 'crucitti' and self.prm['r'] == 0:
               raise ValueError('r must be positive for the Crucitti model')
           if self.prm['model'] in ['crucitti', 'local_load_sharing']:
               if self.prm['attack'] is not None and self.prm['initial_failures'] is None:
                   if get_attack_category(self.prm['attack']) != 'node':
                       raise ValueError('the selected cascading model requires a node attack')
           if len(self.graph_og) &gt; 0 and self.prm['c'] is not None and self.prm['c'] &lt;= 0:
               raise ValueError('c must be positive')

       def get_load(self, graph, weight=None):
           &quot;&quot;&quot;
           Compute unnormalized shortest-path node load.

           :param graph: functioning NetworkX graph
           :param weight: optional edge-distance attribute used for weighted routing
           :return: dictionary mapping node labels to load
           &quot;&quot;&quot;

           if len(graph) == 0:
               return {}

           if self.prm['c'] is None or self.prm['c'] &gt;= len(graph):
               return nx.betweenness_centrality(graph, normalized=False, endpoints=False, weight=weight)

           return nx.betweenness_centrality(graph, k=int(self.prm['c']), normalized=False,
                                            endpoints=False, weight=weight, seed=self.get_random_seed())

       @property
       def shed_load(self):
           &quot;&quot;&quot;Backward-compatible alias for lost_load; no deliberate shedding.&quot;&quot;&quot;
           return self.lost_load

       @shed_load.setter
       def shed_load(self, value):
           self.lost_load = value

       @staticmethod
       def get_efficiency_graph(graph):
           &quot;&quot;&quot;
           Convert edge efficiencies into positive routing distances.

           Zero-efficiency edges remain in the simulation graph but are unavailable
           to the most-efficient-path calculation.

           :param graph: current Crucitti simulation graph
           :return: graph copy with reciprocal edge distances
           &quot;&quot;&quot;

           graph_ = graph.copy()

           for u, v, data in list(graph_.edges(data=True)):
               efficiency = data.get('efficiency', 1)

               if efficiency &lt;= 0:
                   graph_.remove_edge(u, v)
               else:
                   graph_[u][v]['distance'] = 1 / efficiency

           return graph_

       def get_efficiency(self, graph):
           &quot;&quot;&quot;
           Compute average weighted network efficiency from the Crucitti model.

           Path efficiency is the reciprocal of the sum of reciprocal edge
           efficiencies, matching the harmonic path composition in
           :cite:`crucitti2004model`.

           :param graph: current functioning graph
           :return: average efficiency over ordered node pairs
           &quot;&quot;&quot;

           if len(graph) &lt;= 1:
               return 0

           graph_ = self.get_efficiency_graph(graph)
           efficiency = 0

           for source, distances in nx.all_pairs_dijkstra_path_length(graph_, weight='distance'):
               for target, distance in distances.items():
                   if source != target and distance &gt; 0:
                       efficiency += 1 / distance

           return efficiency / (len(graph) * (len(graph) - 1))

       def reset_simulation(self):
           &quot;&quot;&quot;
           Resets the simulation between each run.
           &quot;&quot;&quot;

           self.begin_reset()

           self.graph = self.graph_og.copy()
           self.protected = set()
           self.failed = set()
           self.failed_edges = set()
           self.processed = set()
           self.overloaded = set()
           self.shed_load = 0
           self.sim_info = defaultdict()
           self.last_transfers = {}
           self.capacity = self.capacity_initial.copy()

           if self.prm['model'] in ['motter_lai', 'crucitti', 'local_load_sharing']:
               self.load = self.capacity_og.copy()
           else:
               self.load = {}
               for n in self.graph.nodes:
                   self.load[n] = self.capacity_og[n] * self.rng.uniform(0, self.prm['l'])

           # attacked nodes or edges
           if self.prm['initial_failures'] is not None:
               self.failed = set(self.prm['initial_failures'])
           elif self.prm['attack'] is not None and self.prm['k_a'] &gt; 0:
               attacked = run_attack_method(self.graph, self.prm['attack'], self.prm['k_a'],
                                            approx=self.prm['attack_approx'], seed=self.get_random_seed())

               if get_attack_category(self.prm['attack']) == 'node':
                   self.failed = set(attacked)

               elif get_attack_category(self.prm['attack']) == 'edge':
                   self.failed_edges = set(attacked)
                   self.graph.remove_edges_from(self.failed_edges)

           elif self.prm['attack'] is not None:
               print(self.prm['attack'], &quot;not available or k &lt;= 0&quot;)

           # defended nodes or edges
           if self.prm['defense'] is not None and self.prm['k_d'] &gt; 0:

               if get_defense_category(self.prm['defense']) == 'node':
                   self.protected = set(run_defense_method(self.graph, self.prm['defense'],
                                                           self.prm['k_d'], seed=self.get_random_seed()))
                   for n in self.protected:
                       self.capacity[n] = 2 * self.capacity[n]

               elif get_defense_category(self.prm['defense']) == 'edge':
                   edge_info = run_defense_method(self.graph, self.prm['defense'],
                                                  self.prm['k_d'], seed=self.get_random_seed())

                   if 'removed' in edge_info:
                       self.graph.remove_edges_from(edge_info['removed'])
                   self.graph.add_edges_from(edge_info['added'])

           elif self.prm['defense'] is not None:
               print(self.prm['defense'], &quot;not available or k &lt;= 0&quot;)

           if self.prm['model'] == 'crucitti':
               for u, v in self.graph.edges:
                   self.graph[u][v]['efficiency'] = 1.0

               nodes_functioning = set(self.graph.nodes).difference(self.failed)
               graph_ = self.get_efficiency_graph(self.graph.subgraph(nodes_functioning).copy())
               load = self.get_load(graph_, weight='distance')
               self.load = {n: load.get(n, 0) for n in self.graph_og.nodes}
               self.overloaded = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}

           self.track_simulation(step=0)

       def track_simulation(self, step):
           &quot;&quot;&quot;
           Keeps track of important simulation information at each step of the simulation.

           :param step: current simulation iteration
           &quot;&quot;&quot;

           nodes_functioning = set(self.graph.nodes).difference(self.failed)

           measure = 0
           if len(nodes_functioning) &gt; 0:
               graph_ = self.graph.subgraph(nodes_functioning)

               if self.prm['model'] == 'crucitti':
                   measure = self.get_efficiency(graph_)
               else:
                   measure = run_measure(graph_, self.prm['robust_measure'])

           self.sim_info[step] = {
               'status': [self.load.get(n, 0) for n in self.graph_og.nodes],
               'failed': len(self.failed),
               'failed_edges': self.failed_edges,
               'overloaded': self.overloaded,
               'edge_efficiency': {(u, v): data.get('efficiency', 1)
                                   for u, v, data in self.graph.edges(data=True)},
               'shed_load': self.shed_load,
               'lost_load': self.lost_load,
               'measure': measure,
               'protected': self.protected
           }

       def run_motter_lai_step(self):
           &quot;&quot;&quot;
           Recompute load and synchronously fail overloaded functioning nodes.

           :return: set of nodes that fail in this step
           &quot;&quot;&quot;

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           load = self.get_load(self.graph.subgraph(nodes_functioning).copy())

           self.load = {n: load.get(n, 0) for n in self.graph_og.nodes}
           failed_new = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}
           self.failed.update(failed_new)

           return failed_new

       def run_crucitti_step(self):
           &quot;&quot;&quot;
           Update incident edge efficiencies without removing overloaded nodes.

           All edge updates use the loads at the beginning of the step. If both
           endpoints are overloaded, the lower capacity-to-load ratio determines
           the undirected edge efficiency.

           :return: whether any edge efficiency changed
           &quot;&quot;&quot;

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           factor = {}

           for n in nodes_functioning:
               if self.load.get(n, 0) &gt; self.capacity[n]:
                   factor[n] = self.capacity[n] / self.load[n]
               else:
                   factor[n] = 1

           changed = False
           for u, v in self.graph.subgraph(nodes_functioning).edges:
               efficiency = min(factor[u], factor[v])

               if not np.isclose(self.graph[u][v].get('efficiency', 1), efficiency):
                   changed = True

               self.graph[u][v]['efficiency'] = efficiency

           graph_ = self.get_efficiency_graph(self.graph.subgraph(nodes_functioning).copy())
           load = self.get_load(graph_, weight='distance')
           self.load = {n: load.get(n, 0) for n in self.graph_og.nodes}
           self.overloaded = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}

           return changed

       def run_local_load_sharing_step(self):
           &quot;&quot;&quot;
           Transfer complete failed workloads, then synchronously fail overloads.

           Degree weights use the intact graph. Greedy and proportional allocations
           use pre-round headroom independently for each source. Maximum flow
           coordinates the capacity-fitting pass across all sources. Greedy and
           maximum-flow leftovers are divided equally among eligible recipients;
           proportional sharing uses equal shares when all headroom is zero.
           last_transfers maps (failed source, recipient) to the completed transfer.
           lost_load accumulates work with no functioning neighbor; shed_load is
           retained as a compatibility alias, not a deliberate shedding control.

           :return: set of nodes that fail in this step
           &quot;&quot;&quot;

           sources = [n for n in self.graph_og if n in self.failed and n not in self.processed]
           order = {n: i for i, n in enumerate(self.graph_og)}
           eligible = {n: sorted((j for j in self.graph.neighbors(n) if j not in self.failed),
                                 key=order.__getitem__) for n in sources}
           spare = {n: max(0.0, self.capacity[n] - self.load[n])
                    for n in self.graph if n not in self.failed}
           policy = self.prm['allocation']
           flow = self._local_capacity_flow(sources, eligible, spare) if policy == 'max_flow' else {}
           transfers = defaultdict(float)
           self.last_transfers = {}

           for n in sources:
               nbrs = eligible[n]
               demand = self.load[n]

               if len(nbrs) == 0:
                   self.lost_load += demand
               else:
                   if policy == 'degree':
                       degrees = {j: self.graph_og.degree(j) for j in nbrs}
                       scale = max(degrees.values())
                       weights = {j: (degrees[j] / scale) ** self.prm['beta']
                                  if scale else 1.0 for j in nbrs}
                       total = sum(weights.values())
                       assigned = {j: demand * (weights[j] / total) for j in nbrs}
                   elif policy == 'proportional':
                       scale = max(spare[j] for j in nbrs)
                       weights = {j: spare[j] / scale if scale else 1.0 for j in nbrs}
                       total = sum(weights.values())
                       assigned = {j: demand * (weights[j] / total) for j in nbrs}
                   else:
                       assigned = {j: 0.0 for j in nbrs}
                       remaining = demand
                       if policy == 'greedy':
                           for j in sorted(nbrs, key=lambda j: -spare[j]):
                               assigned[j] = min(remaining, spare[j])
                               remaining -= assigned[j]
                       else:
                           assigned = {j: flow.get(n, {}).get(j, 0.0) for j in nbrs}
                           remaining = max(0.0, demand - sum(assigned.values()))
                       for j in nbrs:
                           assigned[j] += remaining / len(nbrs)
                   for nb in nbrs:
                       self.last_transfers[n, nb] = assigned[nb]
                       transfers[nb] += assigned[nb]

               self.load[n] = 0

           for n, value in transfers.items():
               self.load[n] += value

           self.processed.update(sources)

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           failed_new = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}
           self.failed.update(failed_new)

           return failed_new

       def _local_capacity_flow(self, sources, eligible, spare):
           &quot;&quot;&quot;Joint headroom-fitting pass with deterministic insertion-order ties.&quot;&quot;&quot;
           network = nx.DiGraph()
           source, sink = object(), object()
           network.add_nodes_from([source, sink])
           for n in sources:
               network.add_edge(source, ('failed', n), capacity=self.load[n])
               for j in eligible[n]:
                   network.add_edge(('failed', n), ('recipient', j), capacity=self.load[n])
           for j in self.graph_og:
               if ('recipient', j) in network:
                   network.add_edge(('recipient', j), sink, capacity=spare[j])
           _, flows = nx.maximum_flow(network, source, sink,
                                      flow_func=nx.algorithms.flow.edmonds_karp)
           return {n: {j: flows[('failed', n)].get(('recipient', j), 0.0)
                       for j in eligible[n]} for n in sources}

       def run_legacy_step(self):
           &quot;&quot;&quot;
           Redistribute each newly failed node's load once among functioning neighbors.

           :return: set of nodes that fail in this step
           &quot;&quot;&quot;

           sources = self.failed.difference(self.processed)

           for n in sources:
               nbrs = set(self.graph.neighbors(n)).difference(self.failed)

               if len(nbrs) &gt; 0:
                   share = self.load[n] / len(nbrs)
                   for nb in nbrs:
                       self.load[nb] += share

           self.processed.update(sources)

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           failed_new = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}
           self.failed.update(failed_new)

           return failed_new

       def run_single_sim(self):
           &quot;&quot;&quot;
           Run the cascading-failure simulation.
           &quot;&quot;&quot;

           if 0 not in self.sim_info:
               self.track_simulation(step=0)

           stable = False
           for step in range(self.prm['steps']):
               if not stable:
                   if self.prm['model'] == 'motter_lai':
                       failed_new = self.run_motter_lai_step()
                       stable = len(failed_new) == 0

                   elif self.prm['model'] == 'crucitti':
                       stable = not self.run_crucitti_step()

                   elif self.prm['model'] == 'local_load_sharing':
                       failed_new = self.run_local_load_sharing_step()
                       stable = len(failed_new) == 0

                   else:
                       failed_new = self.run_legacy_step()
                       stable = len(failed_new) == 0

               self.track_simulation(step + 1)

           robustness = [self.sim_info[step]['measure'] for step in range(self.prm['steps'] + 1)]
           return robustness</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L13">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-validate_parameters"><summary><code>Cascading.validate_parameters()</code></summary><div class="api-body"><p>Validate cascading-model parameters.</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def validate_parameters(self):
           &quot;&quot;&quot;
           Validate cascading-model parameters.
           &quot;&quot;&quot;

           models = ['motter_lai', 'crucitti', 'local_load_sharing', 'legacy_redistribution']
           if self.prm['model'] not in models:
               raise ValueError('unknown cascading model')
           if self.prm['l'] &lt; 0 or self.prm['l'] &gt; 1:
               raise ValueError('l must satisfy 0 &lt;= l &lt;= 1')
           if not np.isfinite(self.prm['r']) or self.prm['r'] &lt; 0:
               raise ValueError('r must be nonnegative')
           if not np.isfinite(self.prm['beta']) or self.prm['beta'] &lt; 0:
               raise ValueError('beta must be nonnegative')
           if self.prm['allocation'] not in ['degree', 'greedy', 'proportional', 'max_flow']:
               raise ValueError('unknown local allocation policy')
           local_options = ['initial_load', 'capacities', 'initial_failures']
           if self.prm['model'] != 'local_load_sharing':
               if self.prm['allocation'] != 'degree' or any(self.prm[k] is not None for k in local_options):
                   raise ValueError('allocation and supplied load/capacity/failure inputs require local_load_sharing')
           for key in ['initial_load', 'capacities']:
               values = self.prm[key]
               if values is not None:
                   if not isinstance(values, Mapping) or set(values) != set(self.graph_og):
                       raise ValueError(key + ' must map every graph node exactly once')
                   if any(not isinstance(v, Real) or not np.isfinite(v) or v &lt; 0 for v in values.values()):
                       raise ValueError(key + ' must contain finite nonnegative numbers')
                   self.prm[key] = {n: float(values[n]) for n in self.graph_og}
           if self.prm['initial_failures'] is not None:
               if isinstance(self.prm['initial_failures'], (str, bytes)):
                   raise ValueError('initial_failures must be an iterable of node labels')
               failed = set(self.prm['initial_failures'])
               if not failed.issubset(self.graph_og):
                   raise ValueError('initial_failures contains unknown nodes')
               self.prm['initial_failures'] = tuple(n for n in self.graph_og if n in failed)
           if self.prm['model'] == 'crucitti' and self.prm['r'] == 0:
               raise ValueError('r must be positive for the Crucitti model')
           if self.prm['model'] in ['crucitti', 'local_load_sharing']:
               if self.prm['attack'] is not None and self.prm['initial_failures'] is None:
                   if get_attack_category(self.prm['attack']) != 'node':
                       raise ValueError('the selected cascading model requires a node attack')
           if len(self.graph_og) &gt; 0 and self.prm['c'] is not None and self.prm['c'] &lt;= 0:
               raise ValueError('c must be positive')</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L123">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-get_load"><summary><code>Cascading.get_load(graph, weight=None)</code></summary><div class="api-body"><p>Compute unnormalized shortest-path node load.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>functioning NetworkX graph</td></tr><tr><td><code>weight</code></td><td>optional edge-distance attribute used for weighted routing</td></tr></tbody></table></div><p><strong>Returns</strong> dictionary mapping node labels to load</p><details class="source-code"><summary>View source code</summary><pre><code>def get_load(self, graph, weight=None):
           &quot;&quot;&quot;
           Compute unnormalized shortest-path node load.

           :param graph: functioning NetworkX graph
           :param weight: optional edge-distance attribute used for weighted routing
           :return: dictionary mapping node labels to load
           &quot;&quot;&quot;

           if len(graph) == 0:
               return {}

           if self.prm['c'] is None or self.prm['c'] &gt;= len(graph):
               return nx.betweenness_centrality(graph, normalized=False, endpoints=False, weight=weight)

           return nx.betweenness_centrality(graph, k=int(self.prm['c']), normalized=False,
                                            endpoints=False, weight=weight, seed=self.get_random_seed())</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L167">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-shed_load"><summary><code>Cascading.shed_load</code></summary><div class="api-body"><p>Backward-compatible alias for lost_load; no deliberate shedding.</p><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def shed_load(self):
           &quot;&quot;&quot;Backward-compatible alias for lost_load; no deliberate shedding.&quot;&quot;&quot;
           return self.lost_load</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L186">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-get_efficiency_graph"><summary><code>Cascading.get_efficiency_graph(graph)</code></summary><div class="api-body"><p>Convert edge efficiencies into positive routing distances.

   Zero-efficiency edges remain in the simulation graph but are unavailable
   to the most-efficient-path calculation.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>current Crucitti simulation graph</td></tr></tbody></table></div><p><strong>Returns</strong> graph copy with reciprocal edge distances</p><details class="source-code"><summary>View source code</summary><pre><code>def get_efficiency_graph(graph):
           &quot;&quot;&quot;
           Convert edge efficiencies into positive routing distances.

           Zero-efficiency edges remain in the simulation graph but are unavailable
           to the most-efficient-path calculation.

           :param graph: current Crucitti simulation graph
           :return: graph copy with reciprocal edge distances
           &quot;&quot;&quot;

           graph_ = graph.copy()

           for u, v, data in list(graph_.edges(data=True)):
               efficiency = data.get('efficiency', 1)

               if efficiency &lt;= 0:
                   graph_.remove_edge(u, v)
               else:
                   graph_[u][v]['distance'] = 1 / efficiency

           return graph_</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L195">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-get_efficiency"><summary><code>Cascading.get_efficiency(graph)</code></summary><div class="api-body"><p>Compute average weighted network efficiency from the Crucitti model.

   Path efficiency is the reciprocal of the sum of reciprocal edge
   efficiencies, matching the harmonic path composition in
   .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>current functioning graph</td></tr></tbody></table></div><p><strong>Returns</strong> average efficiency over ordered node pairs</p><p>References: <a href="references.html#ref-crucitti2004model">Model for cascading failures in complex networks (2004)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_efficiency(self, graph):
           &quot;&quot;&quot;
           Compute average weighted network efficiency from the Crucitti model.

           Path efficiency is the reciprocal of the sum of reciprocal edge
           efficiencies, matching the harmonic path composition in
           :cite:`crucitti2004model`.

           :param graph: current functioning graph
           :return: average efficiency over ordered node pairs
           &quot;&quot;&quot;

           if len(graph) &lt;= 1:
               return 0

           graph_ = self.get_efficiency_graph(graph)
           efficiency = 0

           for source, distances in nx.all_pairs_dijkstra_path_length(graph_, weight='distance'):
               for target, distance in distances.items():
                   if source != target and distance &gt; 0:
                       efficiency += 1 / distance

           return efficiency / (len(graph) * (len(graph) - 1))</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L218">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-reset_simulation"><summary><code>Cascading.reset_simulation()</code></summary><div class="api-body"><p>Resets the simulation between each run.</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def reset_simulation(self):
           &quot;&quot;&quot;
           Resets the simulation between each run.
           &quot;&quot;&quot;

           self.begin_reset()

           self.graph = self.graph_og.copy()
           self.protected = set()
           self.failed = set()
           self.failed_edges = set()
           self.processed = set()
           self.overloaded = set()
           self.shed_load = 0
           self.sim_info = defaultdict()
           self.last_transfers = {}
           self.capacity = self.capacity_initial.copy()

           if self.prm['model'] in ['motter_lai', 'crucitti', 'local_load_sharing']:
               self.load = self.capacity_og.copy()
           else:
               self.load = {}
               for n in self.graph.nodes:
                   self.load[n] = self.capacity_og[n] * self.rng.uniform(0, self.prm['l'])

           # attacked nodes or edges
           if self.prm['initial_failures'] is not None:
               self.failed = set(self.prm['initial_failures'])
           elif self.prm['attack'] is not None and self.prm['k_a'] &gt; 0:
               attacked = run_attack_method(self.graph, self.prm['attack'], self.prm['k_a'],
                                            approx=self.prm['attack_approx'], seed=self.get_random_seed())

               if get_attack_category(self.prm['attack']) == 'node':
                   self.failed = set(attacked)

               elif get_attack_category(self.prm['attack']) == 'edge':
                   self.failed_edges = set(attacked)
                   self.graph.remove_edges_from(self.failed_edges)

           elif self.prm['attack'] is not None:
               print(self.prm['attack'], &quot;not available or k &lt;= 0&quot;)

           # defended nodes or edges
           if self.prm['defense'] is not None and self.prm['k_d'] &gt; 0:

               if get_defense_category(self.prm['defense']) == 'node':
                   self.protected = set(run_defense_method(self.graph, self.prm['defense'],
                                                           self.prm['k_d'], seed=self.get_random_seed()))
                   for n in self.protected:
                       self.capacity[n] = 2 * self.capacity[n]

               elif get_defense_category(self.prm['defense']) == 'edge':
                   edge_info = run_defense_method(self.graph, self.prm['defense'],
                                                  self.prm['k_d'], seed=self.get_random_seed())

                   if 'removed' in edge_info:
                       self.graph.remove_edges_from(edge_info['removed'])
                   self.graph.add_edges_from(edge_info['added'])

           elif self.prm['defense'] is not None:
               print(self.prm['defense'], &quot;not available or k &lt;= 0&quot;)

           if self.prm['model'] == 'crucitti':
               for u, v in self.graph.edges:
                   self.graph[u][v]['efficiency'] = 1.0

               nodes_functioning = set(self.graph.nodes).difference(self.failed)
               graph_ = self.get_efficiency_graph(self.graph.subgraph(nodes_functioning).copy())
               load = self.get_load(graph_, weight='distance')
               self.load = {n: load.get(n, 0) for n in self.graph_og.nodes}
               self.overloaded = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}

           self.track_simulation(step=0)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L243">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-track_simulation"><summary><code>Cascading.track_simulation(step)</code></summary><div class="api-body"><p>Keeps track of important simulation information at each step of the simulation.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>current simulation iteration</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def track_simulation(self, step):
           &quot;&quot;&quot;
           Keeps track of important simulation information at each step of the simulation.

           :param step: current simulation iteration
           &quot;&quot;&quot;

           nodes_functioning = set(self.graph.nodes).difference(self.failed)

           measure = 0
           if len(nodes_functioning) &gt; 0:
               graph_ = self.graph.subgraph(nodes_functioning)

               if self.prm['model'] == 'crucitti':
                   measure = self.get_efficiency(graph_)
               else:
                   measure = run_measure(graph_, self.prm['robust_measure'])

           self.sim_info[step] = {
               'status': [self.load.get(n, 0) for n in self.graph_og.nodes],
               'failed': len(self.failed),
               'failed_edges': self.failed_edges,
               'overloaded': self.overloaded,
               'edge_efficiency': {(u, v): data.get('efficiency', 1)
                                   for u, v, data in self.graph.edges(data=True)},
               'shed_load': self.shed_load,
               'lost_load': self.lost_load,
               'measure': measure,
               'protected': self.protected
           }</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L317">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-run_motter_lai_step"><summary><code>Cascading.run_motter_lai_step()</code></summary><div class="api-body"><p>Recompute load and synchronously fail overloaded functioning nodes.</p><p><strong>Returns</strong> set of nodes that fail in this step</p><details class="source-code"><summary>View source code</summary><pre><code>def run_motter_lai_step(self):
           &quot;&quot;&quot;
           Recompute load and synchronously fail overloaded functioning nodes.

           :return: set of nodes that fail in this step
           &quot;&quot;&quot;

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           load = self.get_load(self.graph.subgraph(nodes_functioning).copy())

           self.load = {n: load.get(n, 0) for n in self.graph_og.nodes}
           failed_new = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}
           self.failed.update(failed_new)

           return failed_new</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L348">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-run_crucitti_step"><summary><code>Cascading.run_crucitti_step()</code></summary><div class="api-body"><p>Update incident edge efficiencies without removing overloaded nodes.

   All edge updates use the loads at the beginning of the step. If both
   endpoints are overloaded, the lower capacity-to-load ratio determines
   the undirected edge efficiency.</p><p><strong>Returns</strong> whether any edge efficiency changed</p><details class="source-code"><summary>View source code</summary><pre><code>def run_crucitti_step(self):
           &quot;&quot;&quot;
           Update incident edge efficiencies without removing overloaded nodes.

           All edge updates use the loads at the beginning of the step. If both
           endpoints are overloaded, the lower capacity-to-load ratio determines
           the undirected edge efficiency.

           :return: whether any edge efficiency changed
           &quot;&quot;&quot;

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           factor = {}

           for n in nodes_functioning:
               if self.load.get(n, 0) &gt; self.capacity[n]:
                   factor[n] = self.capacity[n] / self.load[n]
               else:
                   factor[n] = 1

           changed = False
           for u, v in self.graph.subgraph(nodes_functioning).edges:
               efficiency = min(factor[u], factor[v])

               if not np.isclose(self.graph[u][v].get('efficiency', 1), efficiency):
                   changed = True

               self.graph[u][v]['efficiency'] = efficiency

           graph_ = self.get_efficiency_graph(self.graph.subgraph(nodes_functioning).copy())
           load = self.get_load(graph_, weight='distance')
           self.load = {n: load.get(n, 0) for n in self.graph_og.nodes}
           self.overloaded = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}

           return changed</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L364">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-run_local_load_sharing_step"><summary><code>Cascading.run_local_load_sharing_step()</code></summary><div class="api-body"><p>Transfer complete failed workloads, then synchronously fail overloads.

   Degree weights use the intact graph. Greedy and proportional allocations
   use pre-round headroom independently for each source. Maximum flow
   coordinates the capacity-fitting pass across all sources. Greedy and
   maximum-flow leftovers are divided equally among eligible recipients;
   proportional sharing uses equal shares when all headroom is zero.
   last_transfers maps (failed source, recipient) to the completed transfer.
   lost_load accumulates work with no functioning neighbor; shed_load is
   retained as a compatibility alias, not a deliberate shedding control.</p><p><strong>Returns</strong> set of nodes that fail in this step</p><details class="source-code"><summary>View source code</summary><pre><code>def run_local_load_sharing_step(self):
           &quot;&quot;&quot;
           Transfer complete failed workloads, then synchronously fail overloads.

           Degree weights use the intact graph. Greedy and proportional allocations
           use pre-round headroom independently for each source. Maximum flow
           coordinates the capacity-fitting pass across all sources. Greedy and
           maximum-flow leftovers are divided equally among eligible recipients;
           proportional sharing uses equal shares when all headroom is zero.
           last_transfers maps (failed source, recipient) to the completed transfer.
           lost_load accumulates work with no functioning neighbor; shed_load is
           retained as a compatibility alias, not a deliberate shedding control.

           :return: set of nodes that fail in this step
           &quot;&quot;&quot;

           sources = [n for n in self.graph_og if n in self.failed and n not in self.processed]
           order = {n: i for i, n in enumerate(self.graph_og)}
           eligible = {n: sorted((j for j in self.graph.neighbors(n) if j not in self.failed),
                                 key=order.__getitem__) for n in sources}
           spare = {n: max(0.0, self.capacity[n] - self.load[n])
                    for n in self.graph if n not in self.failed}
           policy = self.prm['allocation']
           flow = self._local_capacity_flow(sources, eligible, spare) if policy == 'max_flow' else {}
           transfers = defaultdict(float)
           self.last_transfers = {}

           for n in sources:
               nbrs = eligible[n]
               demand = self.load[n]

               if len(nbrs) == 0:
                   self.lost_load += demand
               else:
                   if policy == 'degree':
                       degrees = {j: self.graph_og.degree(j) for j in nbrs}
                       scale = max(degrees.values())
                       weights = {j: (degrees[j] / scale) ** self.prm['beta']
                                  if scale else 1.0 for j in nbrs}
                       total = sum(weights.values())
                       assigned = {j: demand * (weights[j] / total) for j in nbrs}
                   elif policy == 'proportional':
                       scale = max(spare[j] for j in nbrs)
                       weights = {j: spare[j] / scale if scale else 1.0 for j in nbrs}
                       total = sum(weights.values())
                       assigned = {j: demand * (weights[j] / total) for j in nbrs}
                   else:
                       assigned = {j: 0.0 for j in nbrs}
                       remaining = demand
                       if policy == 'greedy':
                           for j in sorted(nbrs, key=lambda j: -spare[j]):
                               assigned[j] = min(remaining, spare[j])
                               remaining -= assigned[j]
                       else:
                           assigned = {j: flow.get(n, {}).get(j, 0.0) for j in nbrs}
                           remaining = max(0.0, demand - sum(assigned.values()))
                       for j in nbrs:
                           assigned[j] += remaining / len(nbrs)
                   for nb in nbrs:
                       self.last_transfers[n, nb] = assigned[nb]
                       transfers[nb] += assigned[nb]

               self.load[n] = 0

           for n, value in transfers.items():
               self.load[n] += value

           self.processed.update(sources)

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           failed_new = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}
           self.failed.update(failed_new)

           return failed_new</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L400">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-run_legacy_step"><summary><code>Cascading.run_legacy_step()</code></summary><div class="api-body"><p>Redistribute each newly failed node's load once among functioning neighbors.</p><p><strong>Returns</strong> set of nodes that fail in this step</p><details class="source-code"><summary>View source code</summary><pre><code>def run_legacy_step(self):
           &quot;&quot;&quot;
           Redistribute each newly failed node's load once among functioning neighbors.

           :return: set of nodes that fail in this step
           &quot;&quot;&quot;

           sources = self.failed.difference(self.processed)

           for n in sources:
               nbrs = set(self.graph.neighbors(n)).difference(self.failed)

               if len(nbrs) &gt; 0:
                   share = self.load[n] / len(nbrs)
                   for nb in nbrs:
                       self.load[nb] += share

           self.processed.update(sources)

           nodes_functioning = set(self.graph.nodes).difference(self.failed)
           failed_new = {n for n in nodes_functioning if self.load[n] &gt; self.capacity[n]}
           self.failed.update(failed_new)

           return failed_new</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L492">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-cascading-Cascading-run_single_sim"><summary><code>Cascading.run_single_sim()</code></summary><div class="api-body"><p>Run the cascading-failure simulation.</p><p><strong>Returns</strong> List of steps+1 values. Attack/Defense and non-Crucitti cascades: selected measure. Crucitti: weighted efficiency. SIS: infected count. SIR: recovered count. run_simulation averages these across runs.</p><details class="source-code"><summary>View source code</summary><pre><code>def run_single_sim(self):
           &quot;&quot;&quot;
           Run the cascading-failure simulation.
           &quot;&quot;&quot;

           if 0 not in self.sim_info:
               self.track_simulation(step=0)

           stable = False
           for step in range(self.prm['steps']):
               if not stable:
                   if self.prm['model'] == 'motter_lai':
                       failed_new = self.run_motter_lai_step()
                       stable = len(failed_new) == 0

                   elif self.prm['model'] == 'crucitti':
                       stable = not self.run_crucitti_step()

                   elif self.prm['model'] == 'local_load_sharing':
                       failed_new = self.run_local_load_sharing_step()
                       stable = len(failed_new) == 0

                   else:
                       failed_new = self.run_legacy_step()
                       stable = len(failed_new) == 0

               self.track_simulation(step + 1)

           robustness = [self.sim_info[step]['measure'] for step in range(self.prm['steps'] + 1)]
           return robustness</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/cascading.py#L517">View this source on GitHub</a>.</p></div></details></div>
