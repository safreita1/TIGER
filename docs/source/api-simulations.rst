graph_tiger.simulations
=======================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.7.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-simulations"><details class="api-entry" id="api-simulations-Simulation"><summary><code>Simulation(graph, runs, steps, **kwargs)</code></summary><div class="api-body"><p>The parent class for all simulation classes i.e., attack, defense, cascading failure and diffusion models.
   Provides a shared set of functions, largely for network visualization and plotting of results</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>Undirected simple NetworkX graph. See Network inputs for node ordering, conversions, and weight conventions.</td></tr><tr><td><code>runs</code></td><td>Positive integer; number of realizations for run_simulation, default 10 in simulation subclasses.</td></tr><tr><td><code>steps</code></td><td>Nonnegative transition limit. Returned trajectories have steps+1 entries, including index 0.</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization.</td></tr><tr><td><code>plot_transition</code></td><td>Boolean, default False. Save selected first-run snapshots.</td></tr><tr><td><code>gif_animation</code></td><td>Boolean, default False. Write an MP4 with FFmpeg on supported non-Windows platforms.</td></tr><tr><td><code>gif_snaps</code></td><td>Boolean, default False. Save animation frames when that workflow runs.</td></tr><tr><td><code>node_style</code></td><td>None (dataset coordinates or spectral layout) or force_atlas (optional dependency).</td></tr><tr><td><code>edge_style</code></td><td>None (straight edges) or bundled (optional dependency).</td></tr><tr><td><code>fa_iter</code></td><td>Positive integer, default 200; ForceAtlas2 iterations.</td></tr></tbody></table></div><p><strong>Returns</strong> A configured Simulation instance. Construction initializes the graph, state and random generators.</p><p><a href="reproducibility.html">Output shapes, state lifecycle, stopping and random-seed conventions</a></p><details class="source-code"><summary>View source code</summary><pre><code>class Simulation:
       &quot;&quot;&quot;
       The parent class for all simulation classes i.e., attack, defense, cascading failure and diffusion models.
       Provides a shared set of functions, largely for network visualization and plotting of results

       :param graph: undirected NetworkX graph
       :param runs: number of times to run the simulation
       :param steps: number of time steps to run each simulation
       :param kwargs: optional parameters to change visualization settings
       &quot;&quot;&quot;
       def __init__(self, graph, runs, steps, **kwargs):
           self.graph_og = graph.copy()
           self.graph = graph

           self.prm = {
               'runs': runs,
               'steps': steps,

               'seed': 1,
               'max_val': 1,

               'gif_animation': False,
               'gif_snaps': False,
               'plot_transition': False,

               'edge_style': None,
               'node_style': None,
               'fa_iter': 200
           }

           self.prm.update(kwargs)

           if self.prm['runs'] &lt;= 0:
               raise ValueError('runs must be positive')
           if self.prm['steps'] &lt; 0:
               raise ValueError('steps must be nonnegative')

           self.sim_info = defaultdict()
           self.sparse_graph = get_sparse_graph(self.graph)

           self._reset_rng = np.random.RandomState(self.prm['seed'])
           self.random = random.Random(self.prm['seed'])
           self.rng = np.random.RandomState(self.prm['seed'])

       def begin_reset(self):
           &quot;&quot;&quot;
           Start the next reproducible simulation run.
           &quot;&quot;&quot;

           seed = int(self._reset_rng.randint(0, np.iinfo(np.int32).max))
           self.random = random.Random(seed)
           self.rng = np.random.RandomState(seed)

       def get_random_seed(self):
           &quot;&quot;&quot;
           Return the next reproducible seed for a child operation.
           &quot;&quot;&quot;

           return int(self.rng.randint(0, np.iinfo(np.int32).max))

       def child_class(self):
           &quot;&quot;&quot;
           Gets the child class name
           :return: string
           &quot;&quot;&quot;
           return self.__class__.__name__

       def get_graph_coordinates(self):
           &quot;&quot;&quot;
           Gets the graph coordinates, which can be:
           (1) set in the graph itself with the 'pos' tag on the vertices,
           (2) positioned according to the force atlas2 algorithm,
           (3) positioned using a spectral layout.

           Then lays out the edges, can be curved, bundled, or straight

           :return: Tuple containing node and edge positions
           &quot;&quot;&quot;
           edge_pos = None

           node_pos = {k: v['pos'] for k, v in dict(self.graph.nodes).items() if 'pos' in v}   # check graph for coords
           node_pos = node_pos if len(node_pos) == len(self.graph) else None

           # node positions
           if self.prm['node_style'] == 'force_atlas' and node_pos is None:
               try:
                   from fa2 import ForceAtlas2
               except ImportError:
                   raise ImportError(&quot;ForceAtlas2 layout requires the 'visualization' extra&quot;)

               force = ForceAtlas2(outboundAttractionDistribution=True, edgeWeightInfluence=0, scalingRatio=6.0, verbose=False)
               node_pos = force.forceatlas2_networkx_layout(self.graph, pos=None, iterations=self.prm['fa_iter'])

           elif node_pos is None:
               node_pos = nx.spectral_layout(self.graph)

           # edge positions
           if self.prm['edge_style'] == 'bundled':
               try:
                   from datashader.bundling import hammer_bundle
               except ImportError:
                   raise ImportError(&quot;edge bundling requires the 'visualization' extra&quot;)

               pos = pd.DataFrame.from_dict(node_pos, orient='index', columns=['x', 'y']).rename_axis('name').reset_index()
               edge_pos = hammer_bundle(pos, nx.to_pandas_edgelist(self.graph))

           return node_pos, edge_pos

       def plot_results(self, results):
           &quot;&quot;&quot;
           Plots the compiled simulation results

           :param results: a list of floats representing each simulation output
           &quot;&quot;&quot;
           normalize = self.child_class() == 'Diffusion' or self.prm.get('robust_measure') == 'largest_connected_component'
           results_plot = [r / len(self.graph_og) for r in results] if normalize and len(self.graph_og) &gt; 0 else results

           plt.figure(figsize=(6.4, 4.8))

           if self.child_class() == 'Diffusion':
               plt.plot(results_plot, label=&quot;Effective strength: {}&quot;.format(self.get_effective_strength()))

               if self.prm['model'] == 'SIS':
                   plt.ylabel('Infected Nodes')
               else:
                   plt.ylabel('Recovered Nodes')

               plt.legend()
               plt.yscale('log')
               plt.ylim(0.001, 1)

           elif self.child_class() == 'Cascading' or self.child_class() == 'Attack' or self.child_class() == 'Defense':
               plt.plot(results_plot)
               plt.ylabel(self.prm['robust_measure'])
               if normalize:
                   plt.ylim(0, 1)

           plt.xlabel('Steps')
           plt.title(self.child_class())
           plt.savefig(os.path.join(self.save_dir, self.get_plot_title(self.prm['steps']) + '_results.pdf'))
           # plt.show()

           plt.clf()

       def get_plot_title(self, step):
           &quot;&quot;&quot;
           Gets the title for each plot

           :param step: the current simulation iteration
           :return: title string
           &quot;&quot;&quot;
           if self.child_class() == 'Diffusion':
               title = '{}_epidemic--step={},diffusion={},method={},k={}'.format(self.prm['model'], step, self.prm['diffusion'], self.prm['method'], self.prm['k'])

           elif self.child_class() == 'Cascading':
               title = 'Cascading--step={},l={},r={},k_a={},attack={},k_d={},defense={}'.format(step, self.prm['l'], self.prm['r'], self.prm['k_a'],
                                                                                               self.prm['attack'], self.prm['k_d'], self.prm['defense'])
           elif self.child_class() == 'Attack':
               title = 'Attack--step={},attack={},k_d={},defense={}'.format(step, self.prm['attack'], self.prm['k_d'], self.prm['defense'])

           elif self.child_class() == 'Defense':
               title = 'Defense--step={},attack={},k_a={},defense={}'.format(step, self.prm['attack'], self.prm['k_a'], self.prm['defense'])

           else:
               title = ''

           return title

       def plot_graph_transition(self, sim_info):
           &quot;&quot;&quot;
           Helper function to decide which snapshots to take for network visualization

           :param sim_info: the information stored at each step in the simulation
           &quot;&quot;&quot;
           history = [info['failed'] for step, info in sim_info.items()]

           start = history[0]
           end = history[-1]
           middle = start - int((start - end) / 2)
           mid_step, _ = min(enumerate(history), key=lambda x: abs(x[1] - middle))

           steps_to_plot = [0, 1, 2, mid_step, self.prm['steps']]

           for step in sorted(set([step for step in steps_to_plot if step in sim_info])):
               self.plot_network(step=step)

       def get_visual_settings(self, step):
           &quot;&quot;&quot;
           Sets the visual settings for the network visualization

           :param step: current iteration of the simulation
           :return: four lists, each containing a number corresponding to the size or color of each node in the graph + cmap representing color scheme
           &quot;&quot;&quot;
           if self.child_class() == 'Cascading':
               nc, ns = [], []
               ew = 1
               ec = 'gray'

               for node, load in zip(self.graph_og.nodes, self.sim_info[step]['status']):
                   capacity = self.capacity[node]
                   if self.prm['max_val'] &gt; 0:
                       cval = interp1d([0, self.prm['max_val']], [20, 1500])
                       ns.append(float(cval(capacity)))
                   else:
                       ns.append(20)

                   if capacity &gt; 0 and load &lt;= capacity:
                       cval = interp1d([0, capacity], [0, 0.8])
                       nc.append(float(cval(load)))
                   elif load &gt; capacity:
                       nc.append(1)
                   else:
                       nc.append(0)

               cmap = plt.get_cmap('jet', 5)

           elif self.child_class() == 'Diffusion':
               nc, ns = [], []
               ew = 0.1
               ec = '#1F76B4'

               for node, s in zip(self.graph_og.nodes, self.sim_info[step]['status']):
                   if node in self.sim_info[step]['protected']:
                       nc.append(0.5)
                       ns.append(200)
                   elif s == 1:
                       nc.append(s)
                       ns.append(40)
                   else:
                       nc.append(s)
                       ns.append(20)
               cmap = LinearSegmentedColormap.from_list('mycmap', ['#67CAFF', '#17255A', '#FF5964'])

           elif self.child_class() == 'Attack' or self.child_class() == 'Defense':
               ew = 5
               ec = 'gray'
               nc = self.sim_info[step]['status']
               ns = [120 if status == 1 else 40 for status in self.sim_info[step]['status']]
               cmap = plt.get_cmap('gist_rainbow_r')

           nc = np.array(nc)
           ns = np.array(ns)

           return nc, ns, ec, ew, cmap

       def draw_graph(self, step):
           &quot;&quot;&quot;
           Draws the graph

           :param step: current iteration of the simulation
           :return: matplotlib.collections.PathCollection PathCollection` of the nodes.
           &quot;&quot;&quot;
           nc, ns, ec, ew, cmap = self.get_visual_settings(step)

           if self.prm['edge_style'] == 'bundled':
               plt.plot(self.edge_pos.x, self.edge_pos.y, zorder=1, linewidth=ew, color=ec)

           else:
               nx.draw_networkx_edges(self.graph, pos=self.node_pos, width=ew, edge_color=ec)

           nodes = nx.draw_networkx_nodes(self.graph, pos=self.node_pos, cmap=cmap, vmin=0, vmax=self.prm['max_val'], node_size=ns, node_color=nc)

           return nodes

       def plot_network(self, step):
           &quot;&quot;&quot;
           Plots the compiled simulation results

           :param step: current iteration of the simulation
           &quot;&quot;&quot;
           fig = plt.figure(figsize=(20, 20))

           self.draw_graph(step)

           plt.axis('image')
           title = self.get_plot_title(step)
           plt.savefig(os.path.join(self.save_dir, title + '.pdf'))
           # plt.show()
           plt.clf()
           # plt.close(fig)


       def create_simulation_gif(self):
           &quot;&quot;&quot;
           Draws and saves the network simulation to an MP4 file
           &quot;&quot;&quot;
           fig = plt.figure(figsize=(20, 20))
           nodes = self.draw_graph(step=0)

           def update(step):
               nc, ns, _, _, _ = self.get_visual_settings(step)

               nodes.set_array(nc)
               nodes.set_sizes(ns)

               if self.prm['gif_snaps']:
                   snap_dir = os.path.join(self.save_dir, 'gif_snaps/')
                   os.makedirs(snap_dir, exist_ok=True)

                   plt.savefig(snap_dir + 'step_{}.pdf'.format(step))

               return nodes,

           if self.child_class() == 'Diffusion':
               frames = list(range(0, self.prm['steps'] + 1, 10))
               if self.prm['steps'] not in frames:
                   frames.append(self.prm['steps'])
               interval = 20
               fps = 5
           elif self.child_class() == 'Cascading':
               frames = self.prm['steps'] + 1
               interval = 20
               fps = 3
           else:
               frames = self.prm['steps'] + 1
               interval = 20
               fps = 1

           if platform.system() != 'Windows':
               anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=not self.prm['gif_snaps'], repeat=False)
               writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])

               title = self.get_plot_title(self.prm['steps'])
               gif_path = os.path.join(self.save_dir, title + '.mp4')
               anim.save(gif_path, writer=writer)
           else:
               print('Warning: Animated video functionality not supported on Windows; snapshot images are available.')

           plt.clf()

       def run_simulation(self):
           &quot;&quot;&quot;
           Averages the simulation over the number of 'runs'.

           :return: a list containing the average value at each 'step' of the simulation.
           &quot;&quot;&quot;
           print('Running simulation {} times'.format(self.prm['runs']))

           sim_results = list(range(self.prm['runs']))
           for r in range(self.prm['runs']):
               sim_results[r] = self.run_single_sim()

               if self.prm['plot_transition'] and r == 0:
                   self.plot_graph_transition(self.sim_info)

               if self.prm['gif_animation'] and r == 0:
                   self.create_simulation_gif()

               self.reset_simulation()

           result_length = len(sim_results[0])
           if any(len(result) != result_length for result in sim_results):
               raise ValueError('simulation runs returned different timeline lengths')

           avg_results = []
           for t in range(result_length):
               avg_results.append(np.mean([sim_results[r][t] for r in range(self.prm['runs'])]))

           return avg_results

       def reset_simulation(self):
           &quot;&quot;&quot;
           Implemented by child class
           &quot;&quot;&quot;
           pass

       def run_single_sim(self):
           &quot;&quot;&quot;
           Implemented by child class
           &quot;&quot;&quot;
           pass

       def get_effective_strength(self):
           &quot;&quot;&quot;
           Implemented by child class
           &quot;&quot;&quot;
           pass</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L16">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-begin_reset"><summary><code>Simulation.begin_reset()</code></summary><div class="api-body"><p>Start the next reproducible simulation run.</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def begin_reset(self):
           &quot;&quot;&quot;
           Start the next reproducible simulation run.
           &quot;&quot;&quot;

           seed = int(self._reset_rng.randint(0, np.iinfo(np.int32).max))
           self.random = random.Random(seed)
           self.rng = np.random.RandomState(seed)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L61">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-get_random_seed"><summary><code>Simulation.get_random_seed()</code></summary><div class="api-body"><p>Return the next reproducible seed for a child operation.</p><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def get_random_seed(self):
           &quot;&quot;&quot;
           Return the next reproducible seed for a child operation.
           &quot;&quot;&quot;

           return int(self.rng.randint(0, np.iinfo(np.int32).max))</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L70">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-child_class"><summary><code>Simulation.child_class()</code></summary><div class="api-body"><p>Gets the child class name</p><p><strong>Returns</strong> string</p><details class="source-code"><summary>View source code</summary><pre><code>def child_class(self):
           &quot;&quot;&quot;
           Gets the child class name
           :return: string
           &quot;&quot;&quot;
           return self.__class__.__name__</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L77">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-get_graph_coordinates"><summary><code>Simulation.get_graph_coordinates()</code></summary><div class="api-body"><p>Gets the graph coordinates, which can be:
   (1) set in the graph itself with the 'pos' tag on the vertices,
   (2) positioned according to the force atlas2 algorithm,
   (3) positioned using a spectral layout.

   Then lays out the edges, can be curved, bundled, or straight</p><p><strong>Returns</strong> Tuple containing node and edge positions</p><details class="source-code"><summary>View source code</summary><pre><code>def get_graph_coordinates(self):
           &quot;&quot;&quot;
           Gets the graph coordinates, which can be:
           (1) set in the graph itself with the 'pos' tag on the vertices,
           (2) positioned according to the force atlas2 algorithm,
           (3) positioned using a spectral layout.

           Then lays out the edges, can be curved, bundled, or straight

           :return: Tuple containing node and edge positions
           &quot;&quot;&quot;
           edge_pos = None

           node_pos = {k: v['pos'] for k, v in dict(self.graph.nodes).items() if 'pos' in v}   # check graph for coords
           node_pos = node_pos if len(node_pos) == len(self.graph) else None

           # node positions
           if self.prm['node_style'] == 'force_atlas' and node_pos is None:
               try:
                   from fa2 import ForceAtlas2
               except ImportError:
                   raise ImportError(&quot;ForceAtlas2 layout requires the 'visualization' extra&quot;)

               force = ForceAtlas2(outboundAttractionDistribution=True, edgeWeightInfluence=0, scalingRatio=6.0, verbose=False)
               node_pos = force.forceatlas2_networkx_layout(self.graph, pos=None, iterations=self.prm['fa_iter'])

           elif node_pos is None:
               node_pos = nx.spectral_layout(self.graph)

           # edge positions
           if self.prm['edge_style'] == 'bundled':
               try:
                   from datashader.bundling import hammer_bundle
               except ImportError:
                   raise ImportError(&quot;edge bundling requires the 'visualization' extra&quot;)

               pos = pd.DataFrame.from_dict(node_pos, orient='index', columns=['x', 'y']).rename_axis('name').reset_index()
               edge_pos = hammer_bundle(pos, nx.to_pandas_edgelist(self.graph))

           return node_pos, edge_pos</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L84">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-plot_results"><summary><code>Simulation.plot_results(results)</code></summary><div class="api-body"><p>Plots the compiled simulation results</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>results</code></td><td>a list of floats representing each simulation output</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def plot_results(self, results):
           &quot;&quot;&quot;
           Plots the compiled simulation results

           :param results: a list of floats representing each simulation output
           &quot;&quot;&quot;
           normalize = self.child_class() == 'Diffusion' or self.prm.get('robust_measure') == 'largest_connected_component'
           results_plot = [r / len(self.graph_og) for r in results] if normalize and len(self.graph_og) &gt; 0 else results

           plt.figure(figsize=(6.4, 4.8))

           if self.child_class() == 'Diffusion':
               plt.plot(results_plot, label=&quot;Effective strength: {}&quot;.format(self.get_effective_strength()))

               if self.prm['model'] == 'SIS':
                   plt.ylabel('Infected Nodes')
               else:
                   plt.ylabel('Recovered Nodes')

               plt.legend()
               plt.yscale('log')
               plt.ylim(0.001, 1)

           elif self.child_class() == 'Cascading' or self.child_class() == 'Attack' or self.child_class() == 'Defense':
               plt.plot(results_plot)
               plt.ylabel(self.prm['robust_measure'])
               if normalize:
                   plt.ylim(0, 1)

           plt.xlabel('Steps')
           plt.title(self.child_class())
           plt.savefig(os.path.join(self.save_dir, self.get_plot_title(self.prm['steps']) + '_results.pdf'))
           # plt.show()

           plt.clf()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L125">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-get_plot_title"><summary><code>Simulation.get_plot_title(step)</code></summary><div class="api-body"><p>Gets the title for each plot</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>the current simulation iteration</td></tr></tbody></table></div><p><strong>Returns</strong> title string</p><details class="source-code"><summary>View source code</summary><pre><code>def get_plot_title(self, step):
           &quot;&quot;&quot;
           Gets the title for each plot

           :param step: the current simulation iteration
           :return: title string
           &quot;&quot;&quot;
           if self.child_class() == 'Diffusion':
               title = '{}_epidemic--step={},diffusion={},method={},k={}'.format(self.prm['model'], step, self.prm['diffusion'], self.prm['method'], self.prm['k'])

           elif self.child_class() == 'Cascading':
               title = 'Cascading--step={},l={},r={},k_a={},attack={},k_d={},defense={}'.format(step, self.prm['l'], self.prm['r'], self.prm['k_a'],
                                                                                               self.prm['attack'], self.prm['k_d'], self.prm['defense'])
           elif self.child_class() == 'Attack':
               title = 'Attack--step={},attack={},k_d={},defense={}'.format(step, self.prm['attack'], self.prm['k_d'], self.prm['defense'])

           elif self.child_class() == 'Defense':
               title = 'Defense--step={},attack={},k_a={},defense={}'.format(step, self.prm['attack'], self.prm['k_a'], self.prm['defense'])

           else:
               title = ''

           return title</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L161">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-plot_graph_transition"><summary><code>Simulation.plot_graph_transition(sim_info)</code></summary><div class="api-body"><p>Helper function to decide which snapshots to take for network visualization</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>sim_info</code></td><td>the information stored at each step in the simulation</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def plot_graph_transition(self, sim_info):
           &quot;&quot;&quot;
           Helper function to decide which snapshots to take for network visualization

           :param sim_info: the information stored at each step in the simulation
           &quot;&quot;&quot;
           history = [info['failed'] for step, info in sim_info.items()]

           start = history[0]
           end = history[-1]
           middle = start - int((start - end) / 2)
           mid_step, _ = min(enumerate(history), key=lambda x: abs(x[1] - middle))

           steps_to_plot = [0, 1, 2, mid_step, self.prm['steps']]

           for step in sorted(set([step for step in steps_to_plot if step in sim_info])):
               self.plot_network(step=step)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L185">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-get_visual_settings"><summary><code>Simulation.get_visual_settings(step)</code></summary><div class="api-body"><p>Sets the visual settings for the network visualization</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>current iteration of the simulation</td></tr></tbody></table></div><p><strong>Returns</strong> four lists, each containing a number corresponding to the size or color of each node in the graph + cmap representing color scheme</p><details class="source-code"><summary>View source code</summary><pre><code>def get_visual_settings(self, step):
           &quot;&quot;&quot;
           Sets the visual settings for the network visualization

           :param step: current iteration of the simulation
           :return: four lists, each containing a number corresponding to the size or color of each node in the graph + cmap representing color scheme
           &quot;&quot;&quot;
           if self.child_class() == 'Cascading':
               nc, ns = [], []
               ew = 1
               ec = 'gray'

               for node, load in zip(self.graph_og.nodes, self.sim_info[step]['status']):
                   capacity = self.capacity[node]
                   if self.prm['max_val'] &gt; 0:
                       cval = interp1d([0, self.prm['max_val']], [20, 1500])
                       ns.append(float(cval(capacity)))
                   else:
                       ns.append(20)

                   if capacity &gt; 0 and load &lt;= capacity:
                       cval = interp1d([0, capacity], [0, 0.8])
                       nc.append(float(cval(load)))
                   elif load &gt; capacity:
                       nc.append(1)
                   else:
                       nc.append(0)

               cmap = plt.get_cmap('jet', 5)

           elif self.child_class() == 'Diffusion':
               nc, ns = [], []
               ew = 0.1
               ec = '#1F76B4'

               for node, s in zip(self.graph_og.nodes, self.sim_info[step]['status']):
                   if node in self.sim_info[step]['protected']:
                       nc.append(0.5)
                       ns.append(200)
                   elif s == 1:
                       nc.append(s)
                       ns.append(40)
                   else:
                       nc.append(s)
                       ns.append(20)
               cmap = LinearSegmentedColormap.from_list('mycmap', ['#67CAFF', '#17255A', '#FF5964'])

           elif self.child_class() == 'Attack' or self.child_class() == 'Defense':
               ew = 5
               ec = 'gray'
               nc = self.sim_info[step]['status']
               ns = [120 if status == 1 else 40 for status in self.sim_info[step]['status']]
               cmap = plt.get_cmap('gist_rainbow_r')

           nc = np.array(nc)
           ns = np.array(ns)

           return nc, ns, ec, ew, cmap</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L203">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-draw_graph"><summary><code>Simulation.draw_graph(step)</code></summary><div class="api-body"><p>Draws the graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>current iteration of the simulation</td></tr></tbody></table></div><p><strong>Returns</strong> matplotlib.collections.PathCollection PathCollection of the nodes.</p><details class="source-code"><summary>View source code</summary><pre><code>def draw_graph(self, step):
           &quot;&quot;&quot;
           Draws the graph

           :param step: current iteration of the simulation
           :return: matplotlib.collections.PathCollection PathCollection` of the nodes.
           &quot;&quot;&quot;
           nc, ns, ec, ew, cmap = self.get_visual_settings(step)

           if self.prm['edge_style'] == 'bundled':
               plt.plot(self.edge_pos.x, self.edge_pos.y, zorder=1, linewidth=ew, color=ec)

           else:
               nx.draw_networkx_edges(self.graph, pos=self.node_pos, width=ew, edge_color=ec)

           nodes = nx.draw_networkx_nodes(self.graph, pos=self.node_pos, cmap=cmap, vmin=0, vmax=self.prm['max_val'], node_size=ns, node_color=nc)

           return nodes</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L262">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-plot_network"><summary><code>Simulation.plot_network(step)</code></summary><div class="api-body"><p>Plots the compiled simulation results</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>current iteration of the simulation</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def plot_network(self, step):
           &quot;&quot;&quot;
           Plots the compiled simulation results

           :param step: current iteration of the simulation
           &quot;&quot;&quot;
           fig = plt.figure(figsize=(20, 20))

           self.draw_graph(step)

           plt.axis('image')
           title = self.get_plot_title(step)
           plt.savefig(os.path.join(self.save_dir, title + '.pdf'))
           # plt.show()
           plt.clf()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L281">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-create_simulation_gif"><summary><code>Simulation.create_simulation_gif()</code></summary><div class="api-body"><p>Draws and saves the network simulation to an MP4 file</p><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def create_simulation_gif(self):
           &quot;&quot;&quot;
           Draws and saves the network simulation to an MP4 file
           &quot;&quot;&quot;
           fig = plt.figure(figsize=(20, 20))
           nodes = self.draw_graph(step=0)

           def update(step):
               nc, ns, _, _, _ = self.get_visual_settings(step)

               nodes.set_array(nc)
               nodes.set_sizes(ns)

               if self.prm['gif_snaps']:
                   snap_dir = os.path.join(self.save_dir, 'gif_snaps/')
                   os.makedirs(snap_dir, exist_ok=True)

                   plt.savefig(snap_dir + 'step_{}.pdf'.format(step))

               return nodes,

           if self.child_class() == 'Diffusion':
               frames = list(range(0, self.prm['steps'] + 1, 10))
               if self.prm['steps'] not in frames:
                   frames.append(self.prm['steps'])
               interval = 20
               fps = 5
           elif self.child_class() == 'Cascading':
               frames = self.prm['steps'] + 1
               interval = 20
               fps = 3
           else:
               frames = self.prm['steps'] + 1
               interval = 20
               fps = 1

           if platform.system() != 'Windows':
               anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=not self.prm['gif_snaps'], repeat=False)
               writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])

               title = self.get_plot_title(self.prm['steps'])
               gif_path = os.path.join(self.save_dir, title + '.mp4')
               anim.save(gif_path, writer=writer)
           else:
               print('Warning: Animated video functionality not supported on Windows; snapshot images are available.')

           plt.clf()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L299">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-run_simulation"><summary><code>Simulation.run_simulation()</code></summary><div class="api-body"><p>Run repeated realizations and return their mean trajectory. A reset occurs after every run, including the last; inspect individual histories with run_single_sim before resetting.</p><p><strong>Returns</strong> List of steps+1 values. Attack/Defense and non-Crucitti cascades: selected measure. Crucitti: weighted efficiency. SIS: infected count. SIR: recovered count. run_simulation averages these across runs.</p><details class="source-code"><summary>View source code</summary><pre><code>def run_simulation(self):
           &quot;&quot;&quot;
           Averages the simulation over the number of 'runs'.

           :return: a list containing the average value at each 'step' of the simulation.
           &quot;&quot;&quot;
           print('Running simulation {} times'.format(self.prm['runs']))

           sim_results = list(range(self.prm['runs']))
           for r in range(self.prm['runs']):
               sim_results[r] = self.run_single_sim()

               if self.prm['plot_transition'] and r == 0:
                   self.plot_graph_transition(self.sim_info)

               if self.prm['gif_animation'] and r == 0:
                   self.create_simulation_gif()

               self.reset_simulation()

           result_length = len(sim_results[0])
           if any(len(result) != result_length for result in sim_results):
               raise ValueError('simulation runs returned different timeline lengths')

           avg_results = []
           for t in range(result_length):
               avg_results.append(np.mean([sim_results[r][t] for r in range(self.prm['runs'])]))

           return avg_results</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L347">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-reset_simulation"><summary><code>Simulation.reset_simulation()</code></summary><div class="api-body"><p>Implemented by child class</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def reset_simulation(self):
           &quot;&quot;&quot;
           Implemented by child class
           &quot;&quot;&quot;
           pass</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L377">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-run_single_sim"><summary><code>Simulation.run_single_sim()</code></summary><div class="api-body"><p>Implemented by child class</p><p><strong>Returns</strong> List of steps+1 values. Attack/Defense and non-Crucitti cascades: selected measure. Crucitti: weighted efficiency. SIS: infected count. SIR: recovered count. run_simulation averages these across runs.</p><details class="source-code"><summary>View source code</summary><pre><code>def run_single_sim(self):
           &quot;&quot;&quot;
           Implemented by child class
           &quot;&quot;&quot;
           pass</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L383">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-simulations-Simulation-get_effective_strength"><summary><code>Simulation.get_effective_strength()</code></summary><div class="api-body"><p>Implemented by child class</p><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def get_effective_strength(self):
           &quot;&quot;&quot;
           Implemented by child class
           &quot;&quot;&quot;
           pass</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/simulations.py#L389">View this source on GitHub</a>.</p></div></details></div>
