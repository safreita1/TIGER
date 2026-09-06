graph_tiger.diffusion
=====================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.6.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-diffusion"><details class="api-entry" id="api-diffusion-Diffusion"><summary><code>Diffusion(graph, model='SIS', runs=10, steps=5000, b=0.00208, d=0.01, c=1, **kwargs)</code></summary><div class="api-body"><p>Synchronous SIS/SIR dynamics on a contact graph. Transmission ignores weight attributes. SIS returns infected counts; SIR returns recovered counts.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>Undirected simple NetworkX graph. See Network inputs for node ordering, conversions, and weight conventions.</td></tr><tr><td><code>model</code></td><td>SIS (default) or SIR.</td></tr><tr><td><code>runs</code></td><td>Positive integer; number of realizations for run_simulation, default 10 in simulation subclasses.</td></tr><tr><td><code>steps</code></td><td>Nonnegative transition limit. Returned trajectories have steps+1 entries, including index 0.</td></tr><tr><td><code>b</code></td><td>Per-step transmission probability per infected–susceptible edge, in [0,1]; default 0.00208.</td></tr><tr><td><code>d</code></td><td>Per-step recovery probability for each initially infected node in the step, in [0,1]; default 0.01.</td></tr><tr><td><code>c</code></td><td>Initially infected fraction in [0,1], default 1. Uses floor(c*n) nodes, before vaccination.</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization.</td></tr><tr><td><code>plot_transition</code></td><td>Boolean, default False. Save selected first-run snapshots.</td></tr><tr><td><code>gif_animation</code></td><td>Boolean, default False. Write an MP4 with FFmpeg on supported non-Windows platforms.</td></tr><tr><td><code>gif_snaps</code></td><td>Boolean, default False. Save animation frames when that workflow runs.</td></tr><tr><td><code>node_style</code></td><td>None (dataset coordinates or spectral layout) or force_atlas (optional dependency).</td></tr><tr><td><code>edge_style</code></td><td>None (straight edges) or bundled (optional dependency).</td></tr><tr><td><code>fa_iter</code></td><td>Positive integer, default 200; ForceAtlas2 iterations.</td></tr><tr><td><code>diffusion</code></td><td>None, min (vaccination/contact deletion), or max (edge addition/rewiring); default None.</td></tr><tr><td><code>method</code></td><td>Selection key required when diffusion is min or max.</td></tr><tr><td><code>k</code></td><td>Nonnegative intervention budget when diffusion is requested; default None.</td></tr></tbody></table></div><p><strong>Returns</strong> A configured Diffusion instance. Construction initializes the graph, state and random generators.</p><p><a href="reproducibility.html">Output shapes, state lifecycle, stopping and random-seed conventions</a></p><p>References: <a href="references.html#ref-kermack1927contribution">A contribution to the mathematical theory of epidemics (1927)</a></p><details class="source-code"><summary>View source code</summary><pre><code>class Diffusion(Simulation):
       &quot;&quot;&quot;
       Simulates the propagation of a virus using either the SIS or SIR model :cite:`kermack1927contribution`.

       :param graph: contact network
       :param model: a string to set the model type (i.e., SIS or SIR)
       :param runs: an integer number of times to run the simulation
       :param steps: an integer number of steps to run a single simulation
       :param b: float representing birth rate of virus (probability of transmitting disease to each neighbor)
       :param d: float representing death rate of virus (probability of each infected node healing)
       :param c: fraction of initially infected nodes
       :param kwargs: see parent class Simulation for additional options
       &quot;&quot;&quot;

       def __init__(self, graph, model='SIS', runs=10, steps=5000, b=0.00208, d=0.01, c=1, **kwargs):
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
           self.validate_parameters()

           self.vaccinated = set()
           self.infected = set()

           if self.prm['plot_transition'] or self.prm['gif_animation']:
               self.node_pos, self.edge_pos = self.get_graph_coordinates()

           self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
           os.makedirs(self.save_dir, exist_ok=True)

           self.reset_simulation()

       def get_effective_strength(self):
           &quot;&quot;&quot;
           Gets the effective string of the virus. This is a factor of the spectral radius (first eigenvalue) of graph,
           the virus birth rate 'b' and the virus death rate 'd'

           :return: a float for virus effective strength
           &quot;&quot;&quot;

           if self.prm['d'] == 0:
               return np.inf

           return round(spectral_radius(self.graph) * self.prm['b'] / self.prm['d'], 2)

       def reset_simulation(self):
           &quot;&quot;&quot;
           Resets the simulation between each run
           &quot;&quot;&quot;

           self.begin_reset()

           self.graph = self.graph_og.copy()
           self.vaccinated = set()
           self.sim_info = defaultdict()

           self.infected = set(self.rng.choice(list(self.graph.nodes), size=int(self.prm['c'] * len(self.graph)), replace=False).tolist())

           # decrease network diffusion
           if self.prm['diffusion'] == 'min' and self.prm['k'] &gt; 0:

               if get_attack_category(self.prm['method']) == 'node':
                   self.vaccinated = set(run_attack_method(self.graph, self.prm['method'], self.prm['k'], seed=self.get_random_seed()))
                   self.infected = self.infected.difference(self.vaccinated)

               elif get_attack_category(self.prm['method']) == 'edge':
                   edge_info = run_attack_method(self.graph, self.prm['method'], self.prm['k'], seed=self.get_random_seed())
                   self.graph.remove_edges_from(edge_info)
               else:
                   print(self.prm['method'], 'not available')

           # increase network diffusion
           elif self.prm['diffusion'] == 'max' and self.prm['k'] &gt; 0:

               if get_defense_category(self.prm['method']) == 'edge':
                   edge_info = run_defense_method(self.graph, self.prm['method'], self.prm['k'], seed=self.get_random_seed())

                   self.graph.add_edges_from(edge_info['added'])
                   if 'removed' in edge_info:
                       self.graph.remove_edges_from(edge_info['removed'])
               else:
                   print(self.prm['method'], 'not available')

           elif self.prm['diffusion'] is not None:
               print(self.prm['diffusion'], &quot;not available or k &lt;= 0&quot;)

           self.track_simulation(step=0)

       def validate_parameters(self):
           &quot;&quot;&quot;
           Validate discrete-time SIS/SIR parameters.
           &quot;&quot;&quot;

           if self.prm['model'] not in ['SIS', 'SIR']:
               raise ValueError(&quot;model must be 'SIS' or 'SIR'&quot;)
           if self.prm['b'] &lt; 0 or self.prm['b'] &gt; 1:
               raise ValueError('b must satisfy 0 &lt;= b &lt;= 1')
           if self.prm['d'] &lt; 0 or self.prm['d'] &gt; 1:
               raise ValueError('d must satisfy 0 &lt;= d &lt;= 1')
           if self.prm['c'] &lt; 0 or self.prm['c'] &gt; 1:
               raise ValueError('c must satisfy 0 &lt;= c &lt;= 1')
           if self.prm['runs'] &lt;= 0:
               raise ValueError('runs must be positive')
           if self.prm['steps'] &lt; 0:
               raise ValueError('steps must be nonnegative')
           if self.prm['diffusion'] not in [None, 'min', 'max']:
               raise ValueError(&quot;diffusion must be None, 'min', or 'max'&quot;)
           if self.prm['diffusion'] is not None and (self.prm['k'] is None or self.prm['k'] &lt; 0):
               raise ValueError('k must be nonnegative when diffusion is requested')

       def track_simulation(self, step):
           &quot;&quot;&quot;
             Keeps track of important simulation information at each step of the simulation

             :param step: current simulation iteration
             &quot;&quot;&quot;

           self.sim_info[step] = {
               'status': [1 if n in self.infected else 0 for n in self.graph.nodes],
               'failed': len(self.infected),
               'recovered': len(self.vaccinated),
               'protected': self.vaccinated
           }

       def run_single_sim(self):
           &quot;&quot;&quot;
           The initially infected nodes are chosen uniformly at random. At each time step,
           every infected-susceptible edge transmits independently with probability 'b'.
           Every node infected at the start of the step recovers with probability 'd' and
           becomes susceptible again in SIS or permanently recovered in SIR.
           &quot;&quot;&quot;

           for step in range(self.prm['steps']):
               infected_new = set()
               for node in self.infected:
                   nbrs = self.graph.neighbors(node)
                   nbrs = set(nbrs).difference(self.infected).difference(self.vaccinated)

                   nbrs_infected = set([n for n in nbrs if self.random.random() &lt; self.prm['b']])
                   infected_new = infected_new.union(nbrs_infected)

               cured = set([n for n in self.infected if self.random.random() &lt; self.prm['d']])

               self.infected = self.infected.union(infected_new)
               self.infected = self.infected.difference(cured)

               if self.prm['model'] == 'SIR':
                   self.vaccinated.update(cured)

               self.track_simulation(step + 1)

           if self.prm['model'] == 'SIS':
               history = [self.sim_info[step]['failed'] for step in range(self.prm['steps'] + 1)]
           else:
               history = [self.sim_info[step]['recovered'] for step in range(self.prm['steps'] + 1)]

           return history</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/diffusion.py#L12">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-diffusion-Diffusion-get_effective_strength"><summary><code>Diffusion.get_effective_strength()</code></summary><div class="api-body"><p>Gets the effective string of the virus. This is a factor of the spectral radius (first eigenvalue) of graph,
   the virus birth rate 'b' and the virus death rate 'd'</p><p><strong>Returns</strong> a float for virus effective strength</p><details class="source-code"><summary>View source code</summary><pre><code>def get_effective_strength(self):
           &quot;&quot;&quot;
           Gets the effective string of the virus. This is a factor of the spectral radius (first eigenvalue) of graph,
           the virus birth rate 'b' and the virus death rate 'd'

           :return: a float for virus effective strength
           &quot;&quot;&quot;

           if self.prm['d'] == 0:
               return np.inf

           return round(spectral_radius(self.graph) * self.prm['b'] / self.prm['d'], 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/diffusion.py#L54">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-diffusion-Diffusion-reset_simulation"><summary><code>Diffusion.reset_simulation()</code></summary><div class="api-body"><p>Resets the simulation between each run</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def reset_simulation(self):
           &quot;&quot;&quot;
           Resets the simulation between each run
           &quot;&quot;&quot;

           self.begin_reset()

           self.graph = self.graph_og.copy()
           self.vaccinated = set()
           self.sim_info = defaultdict()

           self.infected = set(self.rng.choice(list(self.graph.nodes), size=int(self.prm['c'] * len(self.graph)), replace=False).tolist())

           # decrease network diffusion
           if self.prm['diffusion'] == 'min' and self.prm['k'] &gt; 0:

               if get_attack_category(self.prm['method']) == 'node':
                   self.vaccinated = set(run_attack_method(self.graph, self.prm['method'], self.prm['k'], seed=self.get_random_seed()))
                   self.infected = self.infected.difference(self.vaccinated)

               elif get_attack_category(self.prm['method']) == 'edge':
                   edge_info = run_attack_method(self.graph, self.prm['method'], self.prm['k'], seed=self.get_random_seed())
                   self.graph.remove_edges_from(edge_info)
               else:
                   print(self.prm['method'], 'not available')

           # increase network diffusion
           elif self.prm['diffusion'] == 'max' and self.prm['k'] &gt; 0:

               if get_defense_category(self.prm['method']) == 'edge':
                   edge_info = run_defense_method(self.graph, self.prm['method'], self.prm['k'], seed=self.get_random_seed())

                   self.graph.add_edges_from(edge_info['added'])
                   if 'removed' in edge_info:
                       self.graph.remove_edges_from(edge_info['removed'])
               else:
                   print(self.prm['method'], 'not available')

           elif self.prm['diffusion'] is not None:
               print(self.prm['diffusion'], &quot;not available or k &lt;= 0&quot;)

           self.track_simulation(step=0)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/diffusion.py#L67">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-diffusion-Diffusion-validate_parameters"><summary><code>Diffusion.validate_parameters()</code></summary><div class="api-body"><p>Validate discrete-time SIS/SIR parameters.</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def validate_parameters(self):
           &quot;&quot;&quot;
           Validate discrete-time SIS/SIR parameters.
           &quot;&quot;&quot;

           if self.prm['model'] not in ['SIS', 'SIR']:
               raise ValueError(&quot;model must be 'SIS' or 'SIR'&quot;)
           if self.prm['b'] &lt; 0 or self.prm['b'] &gt; 1:
               raise ValueError('b must satisfy 0 &lt;= b &lt;= 1')
           if self.prm['d'] &lt; 0 or self.prm['d'] &gt; 1:
               raise ValueError('d must satisfy 0 &lt;= d &lt;= 1')
           if self.prm['c'] &lt; 0 or self.prm['c'] &gt; 1:
               raise ValueError('c must satisfy 0 &lt;= c &lt;= 1')
           if self.prm['runs'] &lt;= 0:
               raise ValueError('runs must be positive')
           if self.prm['steps'] &lt; 0:
               raise ValueError('steps must be nonnegative')
           if self.prm['diffusion'] not in [None, 'min', 'max']:
               raise ValueError(&quot;diffusion must be None, 'min', or 'max'&quot;)
           if self.prm['diffusion'] is not None and (self.prm['k'] is None or self.prm['k'] &lt; 0):
               raise ValueError('k must be nonnegative when diffusion is requested')</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/diffusion.py#L110">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-diffusion-Diffusion-track_simulation"><summary><code>Diffusion.track_simulation(step)</code></summary><div class="api-body"><p>Keeps track of important simulation information at each step of the simulation</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>current simulation iteration</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def track_simulation(self, step):
           &quot;&quot;&quot;
             Keeps track of important simulation information at each step of the simulation

             :param step: current simulation iteration
             &quot;&quot;&quot;

           self.sim_info[step] = {
               'status': [1 if n in self.infected else 0 for n in self.graph.nodes],
               'failed': len(self.infected),
               'recovered': len(self.vaccinated),
               'protected': self.vaccinated
           }</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/diffusion.py#L132">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-diffusion-Diffusion-run_single_sim"><summary><code>Diffusion.run_single_sim()</code></summary><div class="api-body"><p>The initially infected nodes are chosen uniformly at random. At each time step,
   every infected-susceptible edge transmits independently with probability 'b'.
   Every node infected at the start of the step recovers with probability 'd' and
   becomes susceptible again in SIS or permanently recovered in SIR.</p><p><strong>Returns</strong> List of steps+1 values. Attack/Defense and non-Crucitti cascades: selected measure. Crucitti: weighted efficiency. SIS: infected count. SIR: recovered count. run_simulation averages these across runs.</p><details class="source-code"><summary>View source code</summary><pre><code>def run_single_sim(self):
           &quot;&quot;&quot;
           The initially infected nodes are chosen uniformly at random. At each time step,
           every infected-susceptible edge transmits independently with probability 'b'.
           Every node infected at the start of the step recovers with probability 'd' and
           becomes susceptible again in SIS or permanently recovered in SIR.
           &quot;&quot;&quot;

           for step in range(self.prm['steps']):
               infected_new = set()
               for node in self.infected:
                   nbrs = self.graph.neighbors(node)
                   nbrs = set(nbrs).difference(self.infected).difference(self.vaccinated)

                   nbrs_infected = set([n for n in nbrs if self.random.random() &lt; self.prm['b']])
                   infected_new = infected_new.union(nbrs_infected)

               cured = set([n for n in self.infected if self.random.random() &lt; self.prm['d']])

               self.infected = self.infected.union(infected_new)
               self.infected = self.infected.difference(cured)

               if self.prm['model'] == 'SIR':
                   self.vaccinated.update(cured)

               self.track_simulation(step + 1)

           if self.prm['model'] == 'SIS':
               history = [self.sim_info[step]['failed'] for step in range(self.prm['steps'] + 1)]
           else:
               history = [self.sim_info[step]['recovered'] for step in range(self.prm['steps'] + 1)]

           return history</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/diffusion.py#L146">View this source on GitHub</a>.</p></div></details></div>
