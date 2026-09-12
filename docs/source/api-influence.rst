graph_tiger.influence
=====================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.7.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-influence"><details class="api-entry" id="api-influence-Influence"><summary><code>Influence(graph, model='independent_cascade', runs=10, steps=100, seeds=None, probability=0.1, weight='weight', threshold='threshold', initial_state=None, message_seeds=None, tracked_state=1, tie_break='random', priority=None, **kwargs)</code></summary><div class="api-body"><p>Independent cascade, linear threshold, asynchronous voter, and competitive-cascade dynamics. See the information-diffusion guide for update and tie-breaking conventions.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>Simple NetworkX Graph or DiGraph. Independent, threshold, and competitive cascades follow outgoing edges; voter requires an undirected graph.</td></tr><tr><td><code>model</code></td><td>independent_cascade (default), linear_threshold, voter, or competitive_cascade.</td></tr><tr><td><code>runs</code></td><td>Positive integer; number of realizations for run_simulation, default 10 in simulation subclasses.</td></tr><tr><td><code>steps</code></td><td>Nonnegative transition limit. Returned trajectories have steps+1 entries, including index 0.</td></tr><tr><td><code>seeds</code></td><td>Initially active graph nodes for independent cascade or linear threshold; default empty.</td></tr><tr><td><code>probability</code></td><td>Independent cascade: scalar edge-attempt probability or edge-attribute name. Competitive cascade: also accepts a message-to-probability mapping. Values must lie in [0,1].</td></tr><tr><td><code>weight</code></td><td>Linear-threshold edge-attribute name; missing edge values default to 1. Use None for one unit of influence per edge.</td></tr><tr><td><code>threshold</code></td><td>Linear-threshold activation level as a positive scalar, complete node mapping, or node-attribute name.</td></tr><tr><td><code>initial_state</code></td><td>Voter model only: complete node-to-state mapping. States must be hashable.</td></tr><tr><td><code>message_seeds</code></td><td>Competitive cascade only: message-to-seed-set mapping. Seed sets must be nonempty collectively and disjoint.</td></tr><tr><td><code>tracked_state</code></td><td>Voter state or competitive message counted in the returned trajectory.</td></tr><tr><td><code>tie_break</code></td><td>Competitive cascade collision rule: random (default) or priority.</td></tr><tr><td><code>priority</code></td><td>Complete message order used when tie_break is priority.</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization.</td></tr><tr><td><code>plot_transition</code></td><td>Boolean, default False. Save selected first-run snapshots.</td></tr><tr><td><code>gif_animation</code></td><td>Boolean, default False. Write an MP4 with FFmpeg on supported non-Windows platforms.</td></tr><tr><td><code>gif_snaps</code></td><td>Boolean, default False. Save animation frames when that workflow runs.</td></tr><tr><td><code>node_style</code></td><td>None (dataset coordinates or spectral layout) or force_atlas (optional dependency).</td></tr><tr><td><code>edge_style</code></td><td>None (straight edges) or bundled (optional dependency).</td></tr><tr><td><code>fa_iter</code></td><td>Positive integer, default 200; ForceAtlas2 iterations.</td></tr></tbody></table></div><p><strong>Returns</strong> A configured Influence instance. Construction initializes the graph, state and random generators.</p><p><a href="reproducibility.html">Output shapes, state lifecycle, stopping and random-seed conventions</a></p><p>References: <a href="references.html#ref-kempe2003maximizing">Maximizing the Spread of Influence through a Social Network (2003)</a>; <a href="references.html#ref-clifford1973model">A Model for Spatial Conflict (1973)</a>; <a href="references.html#ref-bharathi2007competitive">Competitive Influence Maximization in Social Networks (2007)</a></p><details class="source-code"><summary>View source code</summary><pre><code>class Influence(Simulation):
       &quot;&quot;&quot;
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
       &quot;&quot;&quot;

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
           &quot;&quot;&quot;Validate graph, model, initial-state, and model-specific parameters.&quot;&quot;&quot;
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
                   raise ValueError(&quot;tie_break must be 'random' or 'priority'&quot;)
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
                               or not 0 &lt;= edge_probability &lt;= 1):
                           raise ValueError('edge probabilities must be in [0, 1]')
               elif (not isinstance(value, Real) or not np.isfinite(value)
                     or not 0 &lt;= value &lt;= 1):
                   raise ValueError('probability must be in [0, 1]')

       def _validate_linear_threshold(self):
           weight = self.prm['weight']
           if weight is not None and not isinstance(weight, str):
               raise ValueError('weight must be an edge-attribute name or None')
           if weight is not None:
               for _, _, data in self.graph_og.edges(data=True):
                   value = data.get(weight, 1.0)
                   if (not isinstance(value, Real) or not np.isfinite(value)
                           or value &lt; 0):
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
           if any(not isinstance(value, Real) or not np.isfinite(value) or value &lt;= 0
                  for value in values):
               raise ValueError('linear-threshold values must be positive')

       def reset_simulation(self):
           &quot;&quot;&quot;Reset graph state and advance to the next reproducible random stream.&quot;&quot;&quot;
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
           &quot;&quot;&quot;Advance one synchronous independent-cascade frontier.

           :return: set of nodes activated in this round
           &quot;&quot;&quot;
           activated = set()
           for source in self._ordered(self.frontier):
               for target in self._targets(source):
                   if target in self.active:
                       continue
                   if self.random.random() &lt; self._edge_probability(source, target):
                       activated.add(target)
           self.active.update(activated)
           self.frontier = activated
           for node in activated:
               self.state[node] = 1
           return activated

       def run_linear_threshold_step(self):
           &quot;&quot;&quot;Advance one synchronous linear-threshold frontier.

           :return: set of nodes activated in this round
           &quot;&quot;&quot;
           for source in self._ordered(self.frontier):
               for target in self._targets(source):
                   if target in self.active:
                       continue
                   self.influence[target] += self.graph[source][target].get(
                       self.prm['weight'], 1.0
                   ) if self.prm['weight'] is not None else 1.0

           activated = {
               node for node in self.node_order
               if node not in self.active and self.influence[node] &gt;= self._threshold(node)
           }
           self.active.update(activated)
           self.frontier = activated
           for node in activated:
               self.state[node] = 1
           return activated

       def run_voter_step(self):
           &quot;&quot;&quot;Perform one asynchronous node-copying event.

           :return: the changed node as a set, or an empty set
           &quot;&quot;&quot;
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
           &quot;&quot;&quot;Advance all competitive-message frontiers by one synchronous round.

           :return: set of nodes that received at least one successful proposal
           &quot;&quot;&quot;
           proposals = defaultdict(set)
           for message in self.messages:
               for source in self._ordered(self.frontiers[message]):
                   for target in self._targets(source):
                       if self.state[target] is not None:
                           continue
                       if self.random.random() &lt; self._edge_probability(source, target, message):
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
           &quot;&quot;&quot;Store node states, counts, changes, and frontiers for one step.&quot;&quot;&quot;
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
           &quot;&quot;&quot;Run one realization for the configured number of steps.

           :return: active counts for progressive models or tracked-state counts otherwise
           &quot;&quot;&quot;
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
           &quot;&quot;&quot;Return a stable filename stem for an influence plot.&quot;&quot;&quot;
           return 'Influence--model={},step={}'.format(self.prm['model'], step)

       def plot_results(self, results):
           &quot;&quot;&quot;Plot a normalized active-state or tracked-state trajectory.&quot;&quot;&quot;
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
           &quot;&quot;&quot;Return node and edge styles for a stored influence state.&quot;&quot;&quot;
           colors = ['#d9e0e3', '#cf6636', '#5765b0', '#157a79', '#a65f9e']
           if len(self.state_values) &gt; len(colors):
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
           &quot;&quot;&quot;Save representative snapshots from one influence realization.&quot;&quot;&quot;
           steps = sorted(sim_info)
           if not steps:
               return
           selected = {steps[0], steps[-1]}
           selected.update(steps[1:3])
           selected.add(steps[len(steps) // 2])
           for step in sorted(selected):
               self.plot_network(step)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L12">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-validate_parameters"><summary><code>Influence.validate_parameters()</code></summary><div class="api-body"><p>Validate graph, model, initial-state, and model-specific parameters.</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def validate_parameters(self):
           &quot;&quot;&quot;Validate graph, model, initial-state, and model-specific parameters.&quot;&quot;&quot;
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
                   raise ValueError(&quot;tie_break must be 'random' or 'priority'&quot;)
               if self.prm['tie_break'] == 'priority':
                   if set(self.prm['priority'] or ()) != set(message_seeds):
                       raise ValueError('priority must contain every message exactly once')

           if model in {'independent_cascade', 'competitive_cascade'}:
               self._validate_probabilities()
           if model == 'linear_threshold':
               self._validate_linear_threshold()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L76">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-reset_simulation"><summary><code>Influence.reset_simulation()</code></summary><div class="api-body"><p>Reset graph state and advance to the next reproducible random stream.</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def reset_simulation(self):
           &quot;&quot;&quot;Reset graph state and advance to the next reproducible random stream.&quot;&quot;&quot;
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
           self.track_simulation(0)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L176">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-run_independent_cascade_step"><summary><code>Influence.run_independent_cascade_step()</code></summary><div class="api-body"><p>Advance one synchronous independent-cascade frontier.</p><p><strong>Returns</strong> set of nodes activated in this round</p><details class="source-code"><summary>View source code</summary><pre><code>def run_independent_cascade_step(self):
           &quot;&quot;&quot;Advance one synchronous independent-cascade frontier.

           :return: set of nodes activated in this round
           &quot;&quot;&quot;
           activated = set()
           for source in self._ordered(self.frontier):
               for target in self._targets(source):
                   if target in self.active:
                       continue
                   if self.random.random() &lt; self._edge_probability(source, target):
                       activated.add(target)
           self.active.update(activated)
           self.frontier = activated
           for node in activated:
               self.state[node] = 1
           return activated</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L232">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-run_linear_threshold_step"><summary><code>Influence.run_linear_threshold_step()</code></summary><div class="api-body"><p>Advance one synchronous linear-threshold frontier.</p><p><strong>Returns</strong> set of nodes activated in this round</p><details class="source-code"><summary>View source code</summary><pre><code>def run_linear_threshold_step(self):
           &quot;&quot;&quot;Advance one synchronous linear-threshold frontier.

           :return: set of nodes activated in this round
           &quot;&quot;&quot;
           for source in self._ordered(self.frontier):
               for target in self._targets(source):
                   if target in self.active:
                       continue
                   self.influence[target] += self.graph[source][target].get(
                       self.prm['weight'], 1.0
                   ) if self.prm['weight'] is not None else 1.0

           activated = {
               node for node in self.node_order
               if node not in self.active and self.influence[node] &gt;= self._threshold(node)
           }
           self.active.update(activated)
           self.frontier = activated
           for node in activated:
               self.state[node] = 1
           return activated</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L250">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-run_voter_step"><summary><code>Influence.run_voter_step()</code></summary><div class="api-body"><p>Perform one asynchronous node-copying event.</p><p><strong>Returns</strong> the changed node as a set, or an empty set</p><details class="source-code"><summary>View source code</summary><pre><code>def run_voter_step(self):
           &quot;&quot;&quot;Perform one asynchronous node-copying event.

           :return: the changed node as a set, or an empty set
           &quot;&quot;&quot;
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
           return {node}</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L273">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-run_competitive_cascade_step"><summary><code>Influence.run_competitive_cascade_step()</code></summary><div class="api-body"><p>Advance all competitive-message frontiers by one synchronous round.</p><p><strong>Returns</strong> set of nodes that received at least one successful proposal</p><details class="source-code"><summary>View source code</summary><pre><code>def run_competitive_cascade_step(self):
           &quot;&quot;&quot;Advance all competitive-message frontiers by one synchronous round.

           :return: set of nodes that received at least one successful proposal
           &quot;&quot;&quot;
           proposals = defaultdict(set)
           for message in self.messages:
               for source in self._ordered(self.frontiers[message]):
                   for target in self._targets(source):
                       if self.state[target] is not None:
                           continue
                       if self.random.random() &lt; self._edge_probability(source, target, message):
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
           return set(proposals)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L290">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-track_simulation"><summary><code>Influence.track_simulation(step)</code></summary><div class="api-body"><p>Store node states, counts, changes, and frontiers for one step.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>Argument used by the implementation shown below. Default: required.</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def track_simulation(self, step):
           &quot;&quot;&quot;Store node states, counts, changes, and frontiers for one step.&quot;&quot;&quot;
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
           }</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L332">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-run_single_sim"><summary><code>Influence.run_single_sim()</code></summary><div class="api-body"><p>Run one realization for the configured number of steps.</p><p><strong>Returns</strong> List of steps+1 node counts. Independent cascade and linear threshold count all active nodes; voter and competitive cascade count tracked_state. run_simulation averages these values across runs.</p><details class="source-code"><summary>View source code</summary><pre><code>def run_single_sim(self):
           &quot;&quot;&quot;Run one realization for the configured number of steps.

           :return: active counts for progressive models or tracked-state counts otherwise
           &quot;&quot;&quot;
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
           return [self.sim_info[step][field] for step in range(self.prm['steps'] + 1)]</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L357">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-get_plot_title"><summary><code>Influence.get_plot_title(step)</code></summary><div class="api-body"><p>Return a stable filename stem for an influence plot.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>Argument used by the implementation shown below. Default: required.</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def get_plot_title(self, step):
           &quot;&quot;&quot;Return a stable filename stem for an influence plot.&quot;&quot;&quot;
           return 'Influence--model={},step={}'.format(self.prm['model'], step)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L379">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-plot_results"><summary><code>Influence.plot_results(results)</code></summary><div class="api-body"><p>Plot a normalized active-state or tracked-state trajectory.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>results</code></td><td>Argument used by the implementation shown below. Default: required.</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def plot_results(self, results):
           &quot;&quot;&quot;Plot a normalized active-state or tracked-state trajectory.&quot;&quot;&quot;
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
           plt.clf()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L383">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-get_visual_settings"><summary><code>Influence.get_visual_settings(step)</code></summary><div class="api-body"><p>Return node and edge styles for a stored influence state.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>Argument used by the implementation shown below. Default: required.</td></tr></tbody></table></div><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def get_visual_settings(self, step):
           &quot;&quot;&quot;Return node and edge styles for a stored influence state.&quot;&quot;&quot;
           colors = ['#d9e0e3', '#cf6636', '#5765b0', '#157a79', '#a65f9e']
           if len(self.state_values) &gt; len(colors):
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
           return node_colors, node_sizes, '#9aa8ae', 1, cmap</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L399">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-influence-Influence-plot_graph_transition"><summary><code>Influence.plot_graph_transition(sim_info)</code></summary><div class="api-body"><p>Save representative snapshots from one influence realization.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>sim_info</code></td><td>Argument used by the implementation shown below. Default: required.</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def plot_graph_transition(self, sim_info):
           &quot;&quot;&quot;Save representative snapshots from one influence realization.&quot;&quot;&quot;
           steps = sorted(sim_info)
           if not steps:
               return
           selected = {steps[0], steps[-1]}
           selected.update(steps[1:3])
           selected.add(steps[len(steps) // 2])
           for step in sorted(selected):
               self.plot_network(step)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/influence.py#L419">View this source on GitHub</a>.</p></div></details></div>
