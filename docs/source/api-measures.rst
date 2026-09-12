graph_tiger.measures
====================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.7.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-measures"><details class="method-keys"><summary>Accepted method names (17)</summary><div class="table-scroll"><table><thead><tr><th>Method key</th><th>Function</th></tr></thead><tbody><tr><td><code>node_connectivity</code></td><td><a href="#api-measures-node_connectivity">node_connectivity</a></td></tr><tr><td><code>edge_connectivity</code></td><td><a href="#api-measures-edge_connectivity">edge_connectivity</a></td></tr><tr><td><code>diameter</code></td><td><a href="#api-measures-diameter">diameter</a></td></tr><tr><td><code>average_distance</code></td><td><a href="#api-measures-avg_distance">avg_distance</a></td></tr><tr><td><code>average_inverse_distance</code></td><td><a href="#api-measures-avg_inverse_distance">avg_inverse_distance</a></td></tr><tr><td><code>average_vertex_betweenness</code></td><td><a href="#api-measures-avg_vertex_betweenness">avg_vertex_betweenness</a></td></tr><tr><td><code>average_edge_betweenness</code></td><td><a href="#api-measures-avg_edge_betweenness">avg_edge_betweenness</a></td></tr><tr><td><code>average_clustering_coefficient</code></td><td><a href="#api-measures-average_clustering_coefficient">average_clustering_coefficient</a></td></tr><tr><td><code>largest_connected_component</code></td><td><a href="#api-measures-largest_connected_component">largest_connected_component</a></td></tr><tr><td><code>spectral_radius</code></td><td><a href="#api-measures-spectral_radius">spectral_radius</a></td></tr><tr><td><code>spectral_gap</code></td><td><a href="#api-measures-spectral_gap">spectral_gap</a></td></tr><tr><td><code>natural_connectivity</code></td><td><a href="#api-measures-natural_connectivity">natural_connectivity</a></td></tr><tr><td><code>spectral_scaling</code></td><td><a href="#api-measures-spectral_scaling">spectral_scaling</a></td></tr><tr><td><code>generalized_robustness_index</code></td><td><a href="#api-measures-generalized_robustness_index">generalized_robustness_index</a></td></tr><tr><td><code>algebraic_connectivity</code></td><td><a href="#api-measures-algebraic_connectivity">algebraic_connectivity</a></td></tr><tr><td><code>number_spanning_trees</code></td><td><a href="#api-measures-num_spanning_trees">num_spanning_trees</a></td></tr><tr><td><code>effective_resistance</code></td><td><a href="#api-measures-effective_resistance">effective_resistance</a></td></tr></tbody></table></div></details><details class="api-entry" id="api-measures-run_measure"><summary><code>run_measure(graph, measure, k=np.inf, use_gpu=False, timeout=None)</code></summary><div class="api-body"><p>Dispatch one structural measure by its accepted key. k controls sampling/truncation only for supported measures. Validate the measure on the graph states expected in your study.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph to measure</td></tr><tr><td><code>measure</code></td><td>string containing the robustness measure to evaluate</td></tr><tr><td><code>k</code></td><td>Sampled sources or retained eigenpairs for supported measures; default infinity. Small k does not guarantee faster execution.</td></tr><tr><td><code>timeout</code></td><td>optional number of seconds to wait for the measure.</td></tr><tr><td><code>use_gpu</code></td><td>Boolean, default False. Optional adjacency partial-spectrum GPU request; Laplacian GPU support is unavailable. Default: False.</td></tr></tbody></table></div><p><strong>Returns</strong> a float representing the robustness of the graph, or None if it times out or a NetworkX error occurs</p><details class="source-code"><summary>View source code</summary><pre><code>def run_measure(graph, measure, k=np.inf, use_gpu=False, timeout=None):
       &quot;&quot;&quot;
       Evaluates graph robustness according to a specified measure

       :param graph: undirected NetworkX graph to measure
       :param measure: string containing the robustness measure to evaluate
       :param k: an integer for fast approximation of certain robustness measures. small k = fast, large k = precise
       :param timeout: optional number of seconds to wait for the measure.
       :return: a float representing the robustness of the graph, or None if it times out or a NetworkX error occurs
       &quot;&quot;&quot;

       if measure not in measures:
           raise ValueError(&quot;measure '{}' is not implemented&quot;.format(measure))
       if timeout is not None and timeout &lt; 0:
           raise ValueError('timeout must be nonnegative')

       executor = None

       try:
           if timeout is None:
               return measures[measure](graph, k=k, use_gpu=use_gpu)

           executor = ThreadPoolExecutor(max_workers=1)
           result = executor.submit(measures[measure], graph, k=k, use_gpu=use_gpu)
           return result.result(timeout=timeout)

       except FutureTimeoutError:
           print('timed out', measure)
           return None

       except nx.NetworkXException as e:
           print('error', e, measure)
           return None

       finally:
           if executor is not None:
               executor.shutdown(wait=False)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L10">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-get_measures"><summary><code>get_measures()</code></summary><div class="api-body"><p>Returns a list of strings representing all of the available graph robustness measures</p><p><strong>Returns</strong> list of strings</p><details class="source-code"><summary>View source code</summary><pre><code>def get_measures():
       &quot;&quot;&quot;
       Returns a list of strings representing all of the available graph robustness measures

       :return: list of strings
       &quot;&quot;&quot;

       return list(measures.keys())</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L49">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-node_connectivity"><summary><code>node_connectivity(graph, **kwargs)</code></summary><div class="api-body"><p>Minimum node-removal count needed to disconnect the graph or reduce it to a trivial graph. A worst-case structural cut measure, not the expected damage under a random process.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> an integer</p><p>References: <a href="references.html#ref-esfahanian2013connectivity">Connectivity algorithms (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def node_connectivity(graph, **kwargs):
       &quot;&quot;&quot;
       Measures the minimal number of vertices that can be removed to disconnect the graph.
       Larger vertex (node) connectivity --&gt; harder to disconnect graph
       --&gt; more robust graph :cite:`esfahanian2013connectivity`.

       :param graph: undirected NetworkX graph
       :return: an integer
       &quot;&quot;&quot;

       return nx.algorithms.node_connectivity(graph)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L76">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-edge_connectivity"><summary><code>edge_connectivity(graph, **kwargs)</code></summary><div class="api-body"><p>Minimum edge-removal count needed to disconnect the graph. This is a worst-case structural cut measure.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> an integer</p><p>References: <a href="references.html#ref-esfahanian2013connectivity">Connectivity algorithms (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def edge_connectivity(graph, **kwargs):
       &quot;&quot;&quot;
       Measures the minimal number of edges that can be removed to disconnect the graph.
       Larger edge connectivity --&gt; harder to disconnect graph --&gt;
       more robust graph :cite:`esfahanian2013connectivity`.

       :param graph: undirected NetworkX graph
       :return: an integer
       &quot;&quot;&quot;

       return nx.algorithms.edge_connectivity(graph)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L89">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-avg_distance"><summary><code>avg_distance(graph, **kwargs)</code></summary><div class="api-body"><p>Mean shortest-path hop distance. Defined only for connected graphs; a smaller value alone does not establish greater robustness.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def avg_distance(graph, **kwargs):
       &quot;&quot;&quot;
       The average distance between all pairs of nodes in the graph.
       The smaller the average shortest path distance, the more robust the graph.
       This can be viewed through the lens of network connectivity i.e.,
       smaller avg. distance --&gt; better connected graph :cite:`ellens2013graph`.

       Undefined for disconnected graphs.

       :param graph: undirected NetworkX graph
       :return: a float
       &quot;&quot;&quot;

       return round(nx.average_shortest_path_length(graph), 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L102">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-avg_inverse_distance"><summary><code>avg_inverse_distance(graph, **kwargs)</code></summary><div class="api-body"><p>Mean reciprocal shortest-path hop distance. Unreachable pairs contribute zero. Node removal changes the averaging population unless you apply an original-node normalization.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def avg_inverse_distance(graph, **kwargs):
       &quot;&quot;&quot;
       The average inverse distance between all pairs of nodes in the graph.
       The larger the average inverse shortest path distance, the more robust the graph.
       This can be viewed through the lens of network connectivity i.e., larger average inverse distance
       --&gt; better connected graph --&gt; more robust graph :cite:`ellens2013graph`.

       Resolves the issue of not working for disconnected graphs in the avg_distance() function.

       :param graph: undirected NetworkX graph
       :return: a float
       &quot;&quot;&quot;

       return round(nx.global_efficiency(graph), 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L118">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-diameter"><summary><code>diameter(graph, **kwargs)</code></summary><div class="api-body"><p>Maximum shortest-path hop distance of a connected graph. Fragmentation can remove distant nodes without improving operational service.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> an integer</p><p>References: <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def diameter(graph, **kwargs):
       &quot;&quot;&quot;
       The diameter of a connected graph is the longest shortest path between all pairs of nodes.
       The smaller the diameter the more robust the graph i.e., smaller diameter --&gt;
       better connected graph --&gt; more robust graph :cite:`ellens2013graph`.

       :param graph: undirected NetworkX graph
       :return: an integer
       &quot;&quot;&quot;

       return nx.diameter(graph)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L134">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-avg_vertex_betweenness"><summary><code>avg_vertex_betweenness(graph, k=np.inf, **kwargs)</code></summary><div class="api-body"><p>Mean unnormalized node betweenness, excluding endpoints. A mean does not describe the concentration of routing load.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>Number of sampled source nodes. Validate approximation error; no fixed fraction is universally adequate.</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def avg_vertex_betweenness(graph, k=np.inf, **kwargs):
       &quot;&quot;&quot;
       The average vertex betweenness of a graph is the summation of vertex betweenness for every node in the graph.
       The smaller the average vertex betweenness, the more robust the graph.
       We can view this as the load of the network being better distributed and
       less dependent on a few nodes :cite:`ellens2013graph`.

       :param graph: undirected NetworkX graph
       :param k: the number of nodes used to approximate betweenness centrality (k=10% of nodes is usually good)
       :return: a float
       &quot;&quot;&quot;

       if len(graph) == 0:
           return 0

       samples = None if np.isinf(k) or k &gt;= len(graph) else int(k)
       node_centralities = nx.betweenness_centrality(graph, k=samples, normalized=False, endpoints=False)
       avg_betw = sum(list(node_centralities.values())) / len(node_centralities)

       return round(avg_betw, 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L147">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-avg_edge_betweenness"><summary><code>avg_edge_betweenness(graph, k=np.inf, **kwargs)</code></summary><div class="api-body"><p>Mean unnormalized edge betweenness. Inspect individual values when studying concentrated brokerage.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>Number of sampled source nodes. Validate approximation error; no fixed fraction is universally adequate.</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def avg_edge_betweenness(graph, k=np.inf, **kwargs):
       &quot;&quot;&quot;
       Similar to vertex betweenness, edge betweenness is defined as the number of shortest paths
       that pass through an edge *e* out of the total possible shortest paths.
       The smaller the average edge betweenness, the more robust the graph. We can view this as the
       load of the network being better distributed and less dependent on a few edges :cite:`ellens2013graph`.

       :param graph: undirected NetworkX graph
       :param k: the number of nodes used to approximate betweenness centrality (k=10% of nodes is usually good)
       :return: a float
       &quot;&quot;&quot;

       samples = None if np.isinf(k) or k &gt;= len(graph) else int(k)
       edge_centralities = nx.edge_betweenness_centrality(graph, k=samples, normalized=False)

       if len(edge_centralities) &gt; 0:
           avg_betweenness = sum(list(edge_centralities.values())) / len(edge_centralities)

           return round(avg_betweenness, 2)
       else:
           return 0</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L169">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-average_clustering_coefficient"><summary><code>average_clustering_coefficient(graph, **kwargs)</code></summary><div class="api-body"><p>Mean local clustering coefficient, not global transitivity. More triangles need not improve every robustness outcome.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def average_clustering_coefficient(graph, **kwargs):
       &quot;&quot;&quot;
       The global clustering coefficient is based on the number of triplets of nodes in the graph,
       and provides an indication of how well nodes tend to cluster together.
       The larger the average global clustering coefficient, the more robust the graph i.e., more triangles --&gt;
       better connected --&gt; more robust graph :cite:`ellens2013graph`.

       :param graph: undirected NetworkX graph
       :return: a float
       &quot;&quot;&quot;
       return round(nx.average_clustering(graph), 2) if len(graph) &gt; 0 else 0</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L192">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-largest_connected_component"><summary><code>largest_connected_component(graph, **kwargs)</code></summary><div class="api-body"><p>Number of nodes in the largest connected component; zero for an empty graph. Divide by the original node count for retained-connectivity curves.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> an integer</p><details class="source-code"><summary>View source code</summary><pre><code>def largest_connected_component(graph, **kwargs):
       &quot;&quot;&quot;
       This measure provides an indication of a graph's connectivity by measuring the number
       of nodes contained in the largest connected component. The larger the value, the more robust the graph.

       :param graph: undirected NetworkX graph
       :return: an integer
       &quot;&quot;&quot;
       if len(graph) == 0:
           return 0

       lcc = max(nx.connected_components(graph), key=len)
       return len(lcc)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L205">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-spectral_radius"><summary><code>spectral_radius(graph, use_gpu=False, **kwargs)</code></summary><div class="api-body"><p>Largest algebraic adjacency eigenvalue. Uses edge weights. Its effect depends on the process; larger values can facilitate epidemic spread.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>use_gpu</code></td><td>defaults to False; set to True to use GPU (if available)</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-chen2015node">Node immunization on large graphs: Theory and algorithms (2015)</a>; <a href="references.html#ref-tong2010vulnerability">On the vulnerability of large graphs (2010)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def spectral_radius(graph, use_gpu=False, **kwargs):
       &quot;&quot;&quot;
       The largest eigenvalue :math:`\lambda_1` of an adjacency matrix **A** is called the spectral radius.
       The larger the spectral radius, the more robust the graph. This can be viewed from its close relationship to the
       &quot;path&quot; or &quot;loop&quot; capacity in a network :cite:`chen2015node,tong2010vulnerability`.

       :param graph: undirected NetworkX graph
       :param use_gpu: defaults to False; set to True to use GPU (if available)
       :return: a float
       &quot;&quot;&quot;
       if len(graph) == 0:
           return 0

       lam = get_adjacency_spectrum(graph, k=1, which='LA', eigvals_only=True, use_gpu=use_gpu)

       idx = lam.argsort()[::-1]  # sort descending algebraic
       lam = lam[idx]

       return round(lam[0], 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L225">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-spectral_gap"><summary><code>spectral_gap(graph, use_gpu=False, **kwargs)</code></summary><div class="api-body"><p>Difference between the largest and second-largest algebraic adjacency eigenvalues. Not the normalized random-walk spectral gap, and not a universal robustness ranking.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>use_gpu</code></td><td>defaults to False; set to True to use GPU (if available)</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-chan2016optimizing">Optimizing network robustness by edge rewiring: a general framework (2016)</a>; <a href="references.html#ref-malliaros2012fast">Fast robustness estimation in large social graphs: Communities and anomaly detection (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def spectral_gap(graph, use_gpu=False, **kwargs):
       &quot;&quot;&quot;
       The difference between the largest and second largest eigenvalues of the adjacency matrix
       (:math:`\lambda_1 - \lambda_2`) is called the spectral gap :math:`\lambda_d`.
       The larger the spectral gap, the more robust the graph.
       Has an advantage over spectral radius since it accounts for
       undesirable bridges in the network :cite:`chan2016optimizing,malliaros2012fast`.

       :param graph: undirected NetworkX graph
       :param use_gpu: defaults to False; set to True to use GPU (if available)
       :return: a float
       &quot;&quot;&quot;
       if len(graph) &lt; 2:
           return 0

       lam = get_adjacency_spectrum(graph, k=2, which='LA', eigvals_only=True, use_gpu=use_gpu)

       idx = lam.argsort()[::-1]  # sort descending algebraic
       lam = lam[idx]

       return round(lam[0] - lam[1], 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L246">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-natural_connectivity"><summary><code>natural_connectivity(graph, k=np.inf, use_gpu=False, **kwargs)</code></summary><div class="api-body"><p>Logarithm of the mean exponentiated adjacency eigenvalue. Full-spectrum structural redundancy diagnostic; truncation omits contributions.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>use_gpu</code></td><td>defaults to False; set to True to use GPU (if available)</td></tr><tr><td><code>k</code></td><td>Nonnegative intervention budget when diffusion is requested; default None. Default: np.inf.</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-chan2014make">Make it or break it: Manipulating robustness in large networks (2014)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def natural_connectivity(graph, k=np.inf, use_gpu=False, **kwargs):
       &quot;&quot;&quot;
       Natural connectivity has a physical and structural interpretation that is tied to the connectivity properties
       of a network, identifying alternative pathways in a network through the weighted number of closed walks.
       The larger the natural connectivity (average eigenvalue of adjacency matrix), the more robust the graph :cite:`chan2014make`.

       :param graph: undirected NetworkX graph
       :param use_gpu: defaults to False; set to True to use GPU (if available)
       :return: a float
       &quot;&quot;&quot;
       if len(graph) == 0:
           return 0

       lam = get_adjacency_spectrum(graph, k=k, which='LA', eigvals_only=True, use_gpu=use_gpu)

       return round(logsumexp(lam.real) - math.log(len(graph)), 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L269">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-odd_subgraph_centrality"><summary><code>odd_subgraph_centrality(i, lam, u)</code></summary><div class="api-body"><p>Odd closed-walk contribution at a node from the supplied adjacency eigenpairs. Used internally by spectral scaling.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>i</code></td><td>node index</td></tr><tr><td><code>lam</code></td><td>largest eigenvalue</td></tr><tr><td><code>u</code></td><td>largest eigenvector</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-estrada2005spectral">Spectral measures of bipartivity in complex networks (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def odd_subgraph_centrality(i, lam, u):
       &quot;&quot;&quot;
       Calculates the number of odd length closed walks that a node participates in :cite:`estrada2005spectral`.
       Used in the calculation of spectral scaling and generalized robustness index.

       :param i: node index
       :param lam: largest eigenvalue
       :param u: largest eigenvector
       :return: a float
       &quot;&quot;&quot;

       sc = 0
       for j in range(len(lam)):
           sc += np.power(u[i, j], 2) * np.sinh(lam[j])

       return sc</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L287">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-spectral_scaling"><summary><code>spectral_scaling(graph, k=np.inf, use_gpu=False, **kwargs)</code></summary><div class="api-body"><p>Root-mean-square log residual linking a leading eigenvector to odd subgraph centrality. Undefined logarithms can produce None; do not interpret that as zero vulnerability.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>use_gpu</code></td><td>defaults to False; set to True to use GPU (if available)</td></tr><tr><td><code>k</code></td><td>Nonnegative intervention budget when diffusion is requested; default None. Default: np.inf.</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-estrada2006network">Network robustness to targeted attacks. The interplay of expansibility and degree distribution (2006)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def spectral_scaling(graph, k=np.inf, use_gpu=False, **kwargs):
       &quot;&quot;&quot;
       Spectral scaling is a combination of the spectral gap and subgraph centrality. Spectral scaling takes into account
       if a graph has many bridges. The smaller the value, the more robust the graph :cite:`estrada2006network`.

       :param graph: undirected NetworkX graph
       :param use_gpu: defaults to False; set to True to use GPU (if available)
       :return: a float
       &quot;&quot;&quot;
       lam, u = get_adjacency_spectrum(graph, k=k, which='LM', eigvals_only=False, use_gpu=use_gpu)

       idx = np.abs(lam).argsort()[::-1]  # sort descending magnitude
       lam = lam[idx]
       u = u[:, idx]

       u[:, 0] = np.abs(u[:, 0])  # first eigenvector should be positive

       sc = 0
       for i in range(len(graph)):
           sc += np.power(np.log10(u[:, 0][i]) - (np.log10(np.power(np.sinh(lam[0]), -0.5)) + 0.5 * np.log10(odd_subgraph_centrality(i, lam, u))), 2)

       sc = np.sqrt(sc / len(graph))
       if np.isnan(sc) or np.isinf(sc): sc = None

       return sc</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L305">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-generalized_robustness_index"><summary><code>generalized_robustness_index(graph, k=30, use_gpu=False, **kwargs)</code></summary><div class="api-body"><p>Spectral scaling using a truncated set of eigenpairs. Direct calls default to k=30; run_measure supplies its own k default.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>use_gpu</code></td><td>defaults to False; set to True to use GPU (if available)</td></tr><tr><td><code>k</code></td><td>Nonnegative intervention budget when diffusion is requested; default None. Default: 30.</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-malliaros2012fast">Fast robustness estimation in large social graphs: Communities and anomaly detection (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def generalized_robustness_index(graph, k=30, use_gpu=False, **kwargs):
       &quot;&quot;&quot;
       This can be considered a fast approximation of spectral scaling. The smaller the value, the more robust the graph.
       Also helps determine if a graph has many bridges (bad for robustness) :cite:`malliaros2012fast`.

       :param graph: undirected NetworkX graph
       :param use_gpu: defaults to False; set to True to use GPU (if available)
       :return: a float
       &quot;&quot;&quot;
       return spectral_scaling(graph, k=k, use_gpu=use_gpu, kwargs=kwargs)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L332">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-algebraic_connectivity"><summary><code>algebraic_connectivity(graph, **kwargs)</code></summary><div class="api-body"><p>Second-smallest combinatorial Laplacian eigenvalue. Zero for disconnected nonnegative-weight graphs. Do not interpret it as an unconditional strict bound on node connectivity.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-fiedler1973algebraic">Algebraic connectivity of graphs (1973)</a>; <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def algebraic_connectivity(graph, **kwargs):
       r&quot;&quot;&quot;
       The larger the algebraic connectivity, the more robust the graph.
       This is due to it's close connection to edge connectivity, where it serves as a lower bound:
       0 &lt; :math:`u_2` &lt; node connectivity &lt; edge connectivity. This means that a network with larger algebraic connectivity
       is harder to disconnect :cite:`fiedler1973algebraic,ellens2013graph`.

       :param graph: undirected NetworkX graph
       :return: a float
       &quot;&quot;&quot;
       if len(graph) &lt; 2:
           return 0

       lam = get_laplacian_spectrum(graph, k=2, use_gpu=kwargs.get('use_gpu', False))

       return round(lam[1], 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L349">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-num_spanning_trees"><summary><code>num_spanning_trees(graph, k=np.inf, **kwargs)</code></summary><div class="api-body"><p>Product of nonzero Laplacian eigenvalues divided by graph order. Full spectrum gives a tree count for unit weights, or a weighted tree-product sum. A partial product is not a generally reliable count estimate.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>Nonnegative intervention budget when diffusion is requested; default None. Default: np.inf.</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-baras2009efficient">Efficient and robust communication topologies for distributed decision making in networked systems (2009)</a>; <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def num_spanning_trees(graph, k=np.inf, **kwargs):
       &quot;&quot;&quot;
       The number of spanning trees *T* is the number of unique spanning trees that can be found in a graph.
       The larger the number of spanning trees, the more robust the graph.
       This can be viewed from the perspective of network connectivity, where a larger set of
       spanning trees means more alternative pathways in the network :cite:`baras2009efficient,ellens2013graph`.

       :param graph: undirected NetworkX graph
       :return: a float
       &quot;&quot;&quot;
       if len(graph) == 0:
           return 0
       if len(graph) == 1:
           return 1
       if not nx.is_connected(graph):
           return 0

       lam = get_laplacian_spectrum(graph, k=k, use_gpu=kwargs.get('use_gpu', False))
       num_trees = np.prod(lam[1:]) / len(graph)

       return round(float(num_trees), 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L367">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-measures-effective_resistance"><summary><code>effective_resistance(graph, k=np.inf, **kwargs)</code></summary><div class="api-body"><p>Sum of effective resistances over unordered node pairs. Edge weights are conductances. Infinity on disconnected graphs; zero for zero or one node.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>Nonnegative intervention budget when diffusion is requested; default None. Default: np.inf.</td></tr></tbody></table></div><p><strong>Returns</strong> a float</p><p>References: <a href="references.html#ref-ellens2013graph">Graph measures and network robustness (2013)</a>; <a href="references.html#ref-ellens2011effective">Effective graph resistance (2011)</a>; <a href="references.html#ref-ghosh2008minimizing">Minimizing effective resistance of a graph (2008)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def effective_resistance(graph, k=np.inf, **kwargs):
       &quot;&quot;&quot;
       This measure views a graph as an electrical circuit where an edge :math:`(i, j)`
       corresponds to a resister of :math:`r_{ij} = 1` Ohm and a node *i* corresponds to a junction.
       We say the *effective graph resistance* *R* is the sum of resistances for all distinct pairs of vertices
       The smaller the effective resistance, the more robust the graph :cite:`ellens2013graph,ellens2011effective,ghosh2008minimizing`.

       :param graph: undirected NetworkX graph
       :return: a float
       &quot;&quot;&quot;
       if len(graph) &lt;= 1:
           return 0
       if not nx.is_connected(graph):
           return np.inf

       lam = get_laplacian_spectrum(graph, k=k, use_gpu=kwargs.get('use_gpu', False))
       resistance = len(graph) * np.sum(1.0 / lam[1:])

       return round(float(resistance), 2)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/measures.py#L390">View this source on GitHub</a>.</p></div></details></div>
