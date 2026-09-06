graph_tiger.defenses
====================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.6.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-defenses"><details class="method-keys"><summary>Accepted method names (17)</summary><div class="table-scroll"><table><thead><tr><th>Method key</th><th>Function</th></tr></thead><tbody><tr><td><code>ns_node</code></td><td><a href="#api-defenses-get_node_ns">get_node_ns</a></td></tr><tr><td><code>pr_node</code></td><td><a href="#api-defenses-get_node_pr">get_node_pr</a></td></tr><tr><td><code>eig_node</code></td><td><a href="#api-defenses-get_node_eig">get_node_eig</a></td></tr><tr><td><code>id_node</code></td><td><a href="#api-defenses-get_node_id">get_node_id</a></td></tr><tr><td><code>rd_node</code></td><td><a href="#api-defenses-get_node_rd">get_node_rd</a></td></tr><tr><td><code>ib_node</code></td><td><a href="#api-defenses-get_node_ib">get_node_ib</a></td></tr><tr><td><code>rb_node</code></td><td><a href="#api-defenses-get_node_rb">get_node_rb</a></td></tr><tr><td><code>rnd_node</code></td><td><a href="#api-defenses-get_node_rnd">get_node_rnd</a></td></tr><tr><td><code>add_edge_pr</code></td><td><a href="#api-defenses-add_edge_pr">add_edge_pr</a></td></tr><tr><td><code>add_edge_eig</code></td><td><a href="#api-defenses-add_edge_eig">add_edge_eig</a></td></tr><tr><td><code>add_edge_deg</code></td><td><a href="#api-defenses-add_edge_degree">add_edge_degree</a></td></tr><tr><td><code>add_edge_random</code></td><td><a href="#api-defenses-add_edge_rnd">add_edge_rnd</a></td></tr><tr><td><code>add_edge_preferential</code></td><td><a href="#api-defenses-add_edge_pref">add_edge_pref</a></td></tr><tr><td><code>rewire_edge_random</code></td><td><a href="#api-defenses-rewire_edge_rnd">rewire_edge_rnd</a></td></tr><tr><td><code>rewire_edge_random_neighbor</code></td><td><a href="#api-defenses-rewire_edge_rnd_neighbor">rewire_edge_rnd_neighbor</a></td></tr><tr><td><code>rewire_edge_preferential</code></td><td><a href="#api-defenses-rewire_edge_pref">rewire_edge_pref</a></td></tr><tr><td><code>rewire_edge_preferential_random</code></td><td><a href="#api-defenses-rewire_edge_pref_rnd">rewire_edge_pref_rnd</a></td></tr></tbody></table></div></details><details class="api-entry" id="api-defenses-run_defense_method"><summary><code>run_defense_method(graph, method, k=3, seed=None)</code></summary><div class="api-body"><p>Runs a specified defense on an undirected graph.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>method</code></td><td>a string representing one of the defense methods</td></tr><tr><td><code>k</code></td><td>number of nodes or edges to defend</td></tr><tr><td><code>seed</code></td><td>sets the seed in order to obtain reproducible defense runs</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes or a dictionary of edge changes</p><details class="source-code"><summary>View source code</summary><pre><code>def run_defense_method(graph, method, k=3, seed=None):
       &quot;&quot;&quot;
       Runs a specified defense on an undirected graph.

       :param graph: an undirected NetworkX graph
       :param method: a string representing one of the defense methods
       :param k: number of nodes or edges to defend
       :param seed: sets the seed in order to obtain reproducible defense runs
       :return: a list of nodes or a dictionary of edge changes
       &quot;&quot;&quot;

       if method not in methods:
           raise ValueError(&quot;defense method '{}' is not implemented&quot;.format(method))
       if not isinstance(k, (int, np.integer)) or k &lt; 0:
           raise ValueError('k must be a nonnegative integer')

       category = get_defense_category(method)
       if k == 0:
           return [] if category == 'node' else defaultdict(list)

       if category == 'node' and k &gt; len(graph):
           raise ValueError('k exceeds the number of available nodes')

       if method.startswith('add_edge') and k &gt; len(list(nx.non_edges(graph))):
           raise ValueError('k exceeds the number of available nonedges')

       if method.startswith('rewire_edge') and k &gt; len(graph.edges):
           raise ValueError('k exceeds the number of available edges')

       rng = np.random.RandomState(seed)
       if method in ['rnd_node', 'add_edge_random', 'rewire_edge_random',
                     'rewire_edge_random_neighbor', 'rewire_edge_preferential',
                     'rewire_edge_preferential_random']:
           return methods[method](graph, k, rng=rng)

       return methods[method](graph, k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L22">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_defense_methods"><summary><code>get_defense_methods()</code></summary><div class="api-body"><p>Gets a list of available defense methods as a list of functions.</p><p><strong>Returns</strong> a list of all defense functions</p><details class="source-code"><summary>View source code</summary><pre><code>def get_defense_methods():
       &quot;&quot;&quot;
       Gets a list of available defense methods as a list of functions.

       :return: a list of all defense functions
       &quot;&quot;&quot;

       return methods.keys()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L59">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_defense_category"><summary><code>get_defense_category(method)</code></summary><div class="api-body"><p>Gets the defense category e.g., 'node', 'edge' defense.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>method</code></td><td>a string representing the defense method</td></tr></tbody></table></div><p><strong>Returns</strong> a string representing the defense type ('node' or 'edge')</p><details class="source-code"><summary>View source code</summary><pre><code>def get_defense_category(method):
       &quot;&quot;&quot;
       Gets the defense category e.g., 'node', 'edge' defense.

       :param method: a string representing the defense method
       :return: a string representing the defense type ('node' or 'edge')
       &quot;&quot;&quot;

       category = None

       if method in categories:
           category = categories[method]

       return category</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L69">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_ns"><summary><code>get_node_ns(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to defend based on the Netshield algorithm .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><p>References: <a href="references.html#ref-tong2010vulnerability">On the vulnerability of large graphs (2010)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_ns(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to defend based on the Netshield algorithm :cite:`tong2010vulnerability`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend

       :return: a list of nodes to defend
       &quot;&quot;&quot;

       return get_node_ns_attack(graph, k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L85">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_pr"><summary><code>get_node_pr(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to defend based on top PageRank entries .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><p>References: <a href="references.html#ref-page1999pagerank">The pagerank citation ranking: Bringing order to the web. (1999)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_pr(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to defend based on top PageRank entries :cite:`page1999pagerank`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend

       :return: a list of nodes to defend
       &quot;&quot;&quot;

       return get_node_pr_attack(graph, k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L98">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_eig"><summary><code>get_node_eig(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to defend based on top eigenvector centrality entries</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_eig(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to defend based on top eigenvector centrality entries

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend
       :return: a list of nodes to defend
       &quot;&quot;&quot;

       return get_node_eig_attack(graph, k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L111">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_ib"><summary><code>get_node_ib(graph, k=3, approx=np.inf)</code></summary><div class="api-body"><p>Get k nodes to defend based on Initial Betweenness (IB) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr><tr><td><code>approx</code></td><td>number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
       is the number of nodes in the graph</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_ib(graph, k=3, approx=np.inf):
       &quot;&quot;&quot;
       Get k nodes to defend based on Initial Betweenness (IB) Removal :cite:`holme2002attack`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend
       :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
           is the number of nodes in the graph

       :return: a list of nodes to defend
       &quot;&quot;&quot;

       return get_node_ib_attack(graph, k, approx)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L123">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_rb"><summary><code>get_node_rb(graph, k=3, approx=np.inf)</code></summary><div class="api-body"><p>Get k nodes to defend based on Recalculated Betweenness (RB) Removal</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr><tr><td><code>approx</code></td><td>number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
       is the number of nodes in the graph</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_rb(graph, k=3, approx=np.inf):
       &quot;&quot;&quot;
       Get k nodes to defend based on Recalculated Betweenness (RB) Removal :cite:`holme2002attack`

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend
       :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
           is the number of nodes in the graph

       :return: a list of nodes to defend
       &quot;&quot;&quot;

       return get_node_rb_attack(graph, k, approx)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L138">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_id"><summary><code>get_node_id(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to defend based on Initial Degree (ID) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_id(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to defend based on Initial Degree (ID) Removal :cite:`holme2002attack`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend

       :return: a list of nodes to defend
       &quot;&quot;&quot;

       return get_node_id_attack(graph, k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L153">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_rd"><summary><code>get_node_rd(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to defend based on Recalculated Degree (RD) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_rd(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to defend based on Recalculated Degree (RD) Removal :cite:`holme2002attack`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend

       :return: a list of nodes to defend
       &quot;&quot;&quot;

       return get_node_rd_attack(graph, k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L166">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_node_rnd"><summary><code>get_node_rnd(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Randomly select k distinct nodes to defend</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to defend</td></tr><tr><td><code>rng</code></td><td>Argument used by the implementation shown below. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to defend</p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_rnd(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Randomly select k distinct nodes to defend

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to defend

       :return: a list of nodes to defend
       &quot;&quot;&quot;
       rng = np.random if rng is None else rng
       return rng.choice(list(graph.nodes), k, replace=False).tolist()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L179">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_central_edges"><summary><code>get_central_edges(graph, k, method='eig')</code></summary><div class="api-body"><p>Internal function to compute edge PageRank, eigenvector centrality and degree centrality.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>int number of edges to add</td></tr><tr><td><code>method</code></td><td>string representing defense method</td></tr></tbody></table></div><p><strong>Returns</strong> list of edges to add</p><details class="source-code"><summary>View source code</summary><pre><code>def get_central_edges(graph, k, method='eig'):
       &quot;&quot;&quot;
       Internal function to compute edge PageRank, eigenvector centrality and degree centrality.

       :param graph: undirected NetworkX graph
       :param k: int number of edges to add
       :param method: string representing defense method
       :return: list of edges to add
       &quot;&quot;&quot;

       if method == 'pr':
           centrality = nx.pagerank(graph)
       elif method == 'eig':
           centrality = nx.eigenvector_centrality(graph)
       elif method == 'deg':
           centrality = dict(graph.degree)
       else:
           raise ValueError(&quot;central edge method '{}' is not implemented&quot;.format(method))

       score = {(u, v): centrality[u] * centrality[v] for u, v in nx.non_edges(graph)}

       return heapq.nlargest(k, score, key=score.get)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L192">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-add_edge_pr"><summary><code>add_edge_pr(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to defend based on top edge PageRank entries .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to add</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'added'</p><p>References: <a href="references.html#ref-tong2012gelling">Gelling, and melting, large graphs by edge manipulation (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def add_edge_pr(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to defend based on top edge PageRank entries :cite:`tong2012gelling`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to add
       :return: a dictionary of the edges to be 'added'
       &quot;&quot;&quot;

       info = defaultdict(list)
       info['added'] = get_central_edges(graph, k, method='pr')

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L216">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-add_edge_eig"><summary><code>add_edge_eig(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to defend based on top edge eigenvector centrality entries .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to add</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'added'</p><p>References: <a href="references.html#ref-tong2012gelling">Gelling, and melting, large graphs by edge manipulation (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def add_edge_eig(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to defend based on top edge eigenvector centrality entries :cite:`tong2012gelling`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to add
       :return: a dictionary of the edges to be 'added'
       &quot;&quot;&quot;

       info = defaultdict(list)
       info['added'] = get_central_edges(graph, k, method='eig')

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L231">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-add_edge_degree"><summary><code>add_edge_degree(graph, k=3)</code></summary><div class="api-body"><p>Add k edges to defend based on top edge degree centrality entries .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to add</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'added'</p><p>References: <a href="references.html#ref-tong2012gelling">Gelling, and melting, large graphs by edge manipulation (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def add_edge_degree(graph, k=3):
       &quot;&quot;&quot;
       Add k edges to defend based on top edge degree centrality entries :cite:`tong2012gelling`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to add
       :return: a dictionary of the edges to be 'added'
       &quot;&quot;&quot;

       info = defaultdict(list)
       info['added'] = get_central_edges(graph, k, method='deg')

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L246">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-add_edge_rnd"><summary><code>add_edge_rnd(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Add k random nonedges to the graph.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to add</td></tr><tr><td><code>rng</code></td><td>optional NumPy random generator</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'added'</p><details class="source-code"><summary>View source code</summary><pre><code>def add_edge_rnd(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Add k random nonedges to the graph.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to add
       :param rng: optional NumPy random generator
       :return: a dictionary of the edges to be 'added'
       &quot;&quot;&quot;

       rng = np.random if rng is None else rng
       info = defaultdict(list)
       available = list(nx.non_edges(graph))

       if k &gt; len(available):
           raise ValueError('k exceeds the number of available nonedges')

       idx = rng.choice(len(available), k, replace=False)
       info['added'] = [available[i] for i in np.atleast_1d(idx)]

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L261">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-add_edge_pref"><summary><code>add_edge_pref(graph, k=3)</code></summary><div class="api-body"><p>Adds edges between the lowest-degree feasible pairs .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to add</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'added'</p><p>References: <a href="references.html#ref-beygelzimer2005improving">Improving network robustness by edge modification (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def add_edge_pref(graph, k=3):
       &quot;&quot;&quot;
       Adds edges between the lowest-degree feasible pairs :cite:`beygelzimer2005improving`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to add
       :return: a dictionary of the edges to be 'added'
       &quot;&quot;&quot;

       graph_ = graph.copy()
       info = defaultdict(list)
       order = {n: idx for idx, n in enumerate(graph_.nodes)}

       for _ in range(k):
           available = list(nx.non_edges(graph_))
           if len(available) == 0:
               raise ValueError('k exceeds the number of available nonedges')

           degree = dict(graph_.degree)
           edge = min(available, key=lambda e: (degree[e[0]] + degree[e[1]],
                                                max(degree[e[0]], degree[e[1]]),
                                                order[e[0]], order[e[1]]))
           graph_.add_edge(*edge)
           info['added'].append(edge)

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L284">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-get_random_nonedges"><summary><code>get_random_nonedges(graph, k, rng, excluded=None)</code></summary><div class="api-body"><p>Return k random nonedges, excluding specified unordered node pairs.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>Undirected simple NetworkX graph. See Network inputs for node ordering, conversions, and weight conventions. Default: required.</td></tr><tr><td><code>k</code></td><td>Nonnegative intervention budget when diffusion is requested; default None. Default: required.</td></tr><tr><td><code>rng</code></td><td>Argument used by the implementation shown below. Default: required.</td></tr><tr><td><code>excluded</code></td><td>Argument used by the implementation shown below. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def get_random_nonedges(graph, k, rng, excluded=None):
       &quot;&quot;&quot;
       Return k random nonedges, excluding specified unordered node pairs.
       &quot;&quot;&quot;

       excluded = set() if excluded is None else excluded
       available = [edge for edge in nx.non_edges(graph) if frozenset(edge) not in excluded]

       if k &gt; len(available):
           raise ValueError('not enough nonedges are available for rewiring')

       idx = rng.choice(len(available), k, replace=False)
       return [available[i] for i in np.atleast_1d(idx)]</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L312">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-rewire_edge_rnd"><summary><code>rewire_edge_rnd(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Removes k random edges and adds k different random nonedges .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to rewire</td></tr><tr><td><code>rng</code></td><td>optional NumPy random generator</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'removed' and edges to be 'added'</p><p>References: <a href="references.html#ref-beygelzimer2005improving">Improving network robustness by edge modification (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def rewire_edge_rnd(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Removes k random edges and adds k different random nonedges :cite:`beygelzimer2005improving`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to rewire
       :param rng: optional NumPy random generator
       :return: a dictionary of the edges to be 'removed' and edges to be 'added'
       &quot;&quot;&quot;

       rng = np.random if rng is None else rng
       graph_ = graph.copy()
       info = defaultdict(list)
       edges = list(graph_.edges)

       if k &gt; len(edges):
           raise ValueError('k exceeds the number of available edges')

       idx = rng.choice(len(edges), k, replace=False)
       info['removed'] = [edges[i] for i in np.atleast_1d(idx)]
       graph_.remove_edges_from(info['removed'])

       excluded = {frozenset(edge) for edge in info['removed']}
       info['added'] = get_random_nonedges(graph_, k, rng, excluded=excluded)

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L327">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-rewire_edge_rnd_neighbor"><summary><code>rewire_edge_rnd_neighbor(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Randomly removes a neighbor edge and adds a different random edge .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to rewire</td></tr><tr><td><code>rng</code></td><td>optional NumPy random generator</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'removed' and edges to be 'added'</p><p>References: <a href="references.html#ref-beygelzimer2005improving">Improving network robustness by edge modification (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def rewire_edge_rnd_neighbor(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Randomly removes a neighbor edge and adds a different random edge :cite:`beygelzimer2005improving`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to rewire
       :param rng: optional NumPy random generator
       :return: a dictionary of the edges to be 'removed' and edges to be 'added'
       &quot;&quot;&quot;

       rng = np.random if rng is None else rng
       graph_ = graph.copy()
       info = defaultdict(list)
       removed_seen = set()

       for _ in range(k):
           candidates = [(u, v) for u in graph_.nodes for v in graph_.neighbors(u)
                         if frozenset((u, v)) not in removed_seen]

           if len(candidates) == 0:
               raise ValueError('not enough distinct neighbor edges are available')

           removed = candidates[int(rng.choice(len(candidates)))]
           graph_.remove_edge(*removed)
           removed_seen.add(frozenset(removed))

           added = get_random_nonedges(graph_, 1, rng, excluded=removed_seen)[0]
           graph_.add_edge(*added)

           info['removed'].append(removed)
           info['added'].append(added)

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L355">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-rewire_edge_pref"><summary><code>rewire_edge_pref(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Detaches a neighbor of a highest-degree node and reconnects that neighbor elsewhere.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to rewire</td></tr><tr><td><code>rng</code></td><td>optional NumPy random generator</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'removed' and edges to be 'added'</p><details class="source-code"><summary>View source code</summary><pre><code>def rewire_edge_pref(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Detaches a neighbor of a highest-degree node and reconnects that neighbor elsewhere.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to rewire
       :param rng: optional NumPy random generator
       :return: a dictionary of the edges to be 'removed' and edges to be 'added'
       &quot;&quot;&quot;

       rng = np.random if rng is None else rng
       graph_ = graph.copy()
       info = defaultdict(list)

       for _ in range(k):
           nodes = [n for n in graph_.nodes if graph_.degree(n) &gt; 0]
           if len(nodes) == 0:
               raise ValueError('not enough edges are available for rewiring')

           u = max(nodes, key=dict(graph_.degree).get)
           nbrs = list(graph_.neighbors(u))
           nbr = nbrs[int(rng.choice(len(nbrs)))]
           removed = (u, nbr)

           graph_.remove_edge(*removed)
           excluded = {frozenset(removed)}
           candidates = [(nbr, v) for v in graph_.nodes
                         if nbr != v and not graph_.has_edge(nbr, v)
                         and frozenset((nbr, v)) not in excluded]

           if len(candidates) == 0:
               raise ValueError('not enough nonedges are available for rewiring')

           added = candidates[int(rng.choice(len(candidates)))]
           graph_.add_edge(*added)

           info['removed'].append(removed)
           info['added'].append(added)

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L389">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-rewire_edge_pref_rnd"><summary><code>rewire_edge_pref_rnd(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Disconnects the higher-degree endpoint and reconnects the other endpoint randomly.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to rewire</td></tr><tr><td><code>rng</code></td><td>optional NumPy random generator</td></tr></tbody></table></div><p><strong>Returns</strong> a dictionary of the edges to be 'removed' and edges to be 'added'</p><details class="source-code"><summary>View source code</summary><pre><code>def rewire_edge_pref_rnd(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Disconnects the higher-degree endpoint and reconnects the other endpoint randomly.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to rewire
       :param rng: optional NumPy random generator
       :return: a dictionary of the edges to be 'removed' and edges to be 'added'
       &quot;&quot;&quot;

       rng = np.random if rng is None else rng
       graph_ = graph.copy()
       info = defaultdict(list)
       edges = list(graph_.edges)

       if k &gt; len(edges):
           raise ValueError('k exceeds the number of available edges')

       idx = rng.choice(len(edges), k, replace=False)
       selected = [edges[i] for i in np.atleast_1d(idx)]

       for u, v in selected:
           anchor = v if graph_.degree(u) &gt; graph_.degree(v) else u
           removed = (u, v)
           graph_.remove_edge(*removed)

           excluded = {frozenset(removed)}
           candidates = [(anchor, n) for n in graph_.nodes
                         if anchor != n and not graph_.has_edge(anchor, n)
                         and frozenset((anchor, n)) not in excluded]

           if len(candidates) == 0:
               raise ValueError('not enough nonedges are available for rewiring')

           added = candidates[int(rng.choice(len(candidates)))]
           graph_.add_edge(*added)

           info['removed'].append(removed)
           info['added'].append(added)

       return info</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L431">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-Defense"><summary><code>Defense(graph, runs=10, steps=50, attack='id_node', defense=None, k_d=0, **kwargs)</code></summary><div class="api-body"><p>This class simulates a variety of defense techniques on an undirected NetworkX graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>Undirected simple NetworkX graph. See Network inputs for node ordering, conversions, and weight conventions.</td></tr><tr><td><code>runs</code></td><td>Positive integer; number of realizations for run_simulation, default 10 in simulation subclasses.</td></tr><tr><td><code>steps</code></td><td>Nonnegative transition limit. Returned trajectories have steps+1 entries, including index 0.</td></tr><tr><td><code>attack</code></td><td>Attack-selection key or None; simulation subclasses default to id_node. Budgets determine whether an initial attack is performed.</td></tr><tr><td><code>defense</code></td><td>None or a supported defense key. Cascade node defense doubles selected capacities, but does not cancel initial failures.</td></tr><tr><td><code>k_d</code></td><td>Nonnegative node/edge defense budget, default 0. Attack: number selected for protection/intervention. Defense: number of proposed edge changes. Cascading: defense budget.</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization.</td></tr><tr><td><code>plot_transition</code></td><td>Boolean, default False. Save selected first-run snapshots.</td></tr><tr><td><code>gif_animation</code></td><td>Boolean, default False. Write an MP4 with FFmpeg on supported non-Windows platforms.</td></tr><tr><td><code>gif_snaps</code></td><td>Boolean, default False. Save animation frames when that workflow runs.</td></tr><tr><td><code>node_style</code></td><td>None (dataset coordinates or spectral layout) or force_atlas (optional dependency).</td></tr><tr><td><code>edge_style</code></td><td>None (straight edges) or bundled (optional dependency).</td></tr><tr><td><code>fa_iter</code></td><td>Positive integer, default 200; ForceAtlas2 iterations.</td></tr><tr><td><code>robust_measure</code></td><td>Measure key; default largest_connected_component for attack, defense and non-Crucitti cascades. Crucitti overrides it with weighted efficiency.</td></tr><tr><td><code>attack_approx</code></td><td>Sampled source-node count for betweenness attack selection; default None means exact. Not a fraction or edge count.</td></tr><tr><td><code>k_a</code></td><td>Nonnegative initial attack budget. Default 10 for Cascading and 0 for Defense; must not exceed available components.</td></tr></tbody></table></div><p><strong>Returns</strong> A configured Defense instance. Construction initializes the graph, state and random generators.</p><p><a href="reproducibility.html">Output shapes, state lifecycle, stopping and random-seed conventions</a></p><details class="source-code"><summary>View source code</summary><pre><code>class Defense(Simulation):
       &quot;&quot;&quot;
       This class simulates a variety of defense techniques on an undirected NetworkX graph

       :param graph: an undirected NetworkX graph
       :param runs: an integer number of times to run the simulation
       :param steps: an integer number of steps to run a single simulation
       :param attack: a string representing the attack strategy to run
       :param defense: a string representing the defense strategy to run
       :param k_d: an integer number of nodes to defend
       :param kwargs: see parent class Simulation for additional options
       &quot;&quot;&quot;

       def __init__(self, graph, runs=10, steps=50, attack='id_node', defense=None, k_d=0, **kwargs):
           super().__init__(graph, runs, steps, **kwargs)
           self.graph = self.graph_og.copy()

           self.prm.update({
               'attack': attack,
               'attack_approx': None,
               'k_a': 0,

               'k_d': k_d,
               'defense': defense,

               'robust_measure': 'largest_connected_component',
           })

           self.prm.update(kwargs)

           if self.prm['plot_transition'] or self.prm['gif_animation']:
               self.node_pos, self.edge_pos = self.get_graph_coordinates()

           self.save_dir = os.path.join(os.getcwd(), 'plots', self.get_plot_title(steps))
           os.makedirs(self.save_dir, exist_ok=True)

           self.attacked = []
           self.protected = defaultdict(list)
           self.connectivity = []

           self.reset_simulation()

       def reset_simulation(self):
           &quot;&quot;&quot;
           Resets the simulation between each run
           &quot;&quot;&quot;

           self.begin_reset()

           self.graph_ = self.graph.copy()
           self.attacked = []
           self.protected = []
           self.connectivity = []

           # attacked nodes or edges
           if self.prm['attack'] is not None and self.prm['k_a'] &gt; 0:
               self.attacked = run_attack_method(self.graph_, self.prm['attack'], self.prm['k_a'], approx=self.prm['attack_approx'], seed=self.get_random_seed())

               if get_attack_category(self.prm['attack']) == 'edge':
                   self.graph_.remove_edges_from(self.attacked)

           elif self.prm['attack'] is not None:
               print(self.prm['attack'], &quot;not available or k &lt;= 0&quot;)

           # defended nodes or edges
           if self.prm['defense'] is not None and self.prm['k_d'] &gt; 0:
               self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.get_random_seed())

           elif self.prm['defense'] is not None:
               print(self.prm['defense'], &quot;not available or k &lt;= 0&quot;)

           # remove attacked nodes after checking that they are not defended
           if get_attack_category(self.prm['attack']) == 'node':
               if get_defense_category(self.prm['defense']) == 'node':
                   diff = set(self.attacked) - set(self.protected)
                   self.graph_.remove_nodes_from(diff)
               else:
                   self.graph_.remove_nodes_from(self.attacked)

           self.track_simulation(step=0)

       def track_simulation(self, step):
           &quot;&quot;&quot;
            Keeps track of important simulation information at each step of the simulation

            :param step: current simulation iteration
            &quot;&quot;&quot;

           measure = run_measure(self.graph_, self.prm['robust_measure'])

           ccs = list(nx.connected_components(self.graph_))
           ccs.sort(key=len, reverse=True)
           m = interp1d([0, len(ccs)], [0.15, 1]) if len(ccs) &gt; 0 else None

           status = {}
           for n in self.graph:
               status[n] = 0
               for idx, cc in enumerate(ccs):
                   if n in self.attacked and n not in self.protected:
                       status[n] = 1
                       break
                   elif n in cc:
                       status[n] = float(m(idx))
                       break

           lcc = len(ccs[0]) if len(ccs) &gt; 0 else 0
           self.sim_info[step] = {
               'status':  list(status.values()),
               'failed': len(self.graph) - lcc,
               'measure': measure,
               'protected': self.protected,
               'edges_added': self.protected['added'][0:step] if 'added' in self.protected else [],
               'edges_removed': self.protected['removed'][0:step] if 'removed' in self.protected else []
           }

       def run_single_sim(self):
           &quot;&quot;&quot;
           Run the defense simulation
           &quot;&quot;&quot;

           for step in range(self.prm['steps']):
               if get_defense_category(self.prm['defense']) == 'edge' and step &lt; len(self.protected['added']):
                   if 'removed' in self.protected and step &lt; len(self.protected['removed']):
                       u, v = self.protected['removed'][step]
                       self.graph_.remove_edge(u, v)

                   u, v = self.protected['added'][step]
                   self.graph_.add_edge(u, v)

               else:
                   print(&quot;Ending defense simulation early, not an 'edge' defense or out of {}s&quot;.format(get_defense_category(self.prm['defense'])))

               self.track_simulation(step + 1)

           results = [self.sim_info[step]['measure'] if self.sim_info[step]['measure'] is not None else 0
                      for step in range(self.prm['steps'] + 1)]
           return results</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L518">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-Defense-reset_simulation"><summary><code>Defense.reset_simulation()</code></summary><div class="api-body"><p>Resets the simulation between each run</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def reset_simulation(self):
           &quot;&quot;&quot;
           Resets the simulation between each run
           &quot;&quot;&quot;

           self.begin_reset()

           self.graph_ = self.graph.copy()
           self.attacked = []
           self.protected = []
           self.connectivity = []

           # attacked nodes or edges
           if self.prm['attack'] is not None and self.prm['k_a'] &gt; 0:
               self.attacked = run_attack_method(self.graph_, self.prm['attack'], self.prm['k_a'], approx=self.prm['attack_approx'], seed=self.get_random_seed())

               if get_attack_category(self.prm['attack']) == 'edge':
                   self.graph_.remove_edges_from(self.attacked)

           elif self.prm['attack'] is not None:
               print(self.prm['attack'], &quot;not available or k &lt;= 0&quot;)

           # defended nodes or edges
           if self.prm['defense'] is not None and self.prm['k_d'] &gt; 0:
               self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.get_random_seed())

           elif self.prm['defense'] is not None:
               print(self.prm['defense'], &quot;not available or k &lt;= 0&quot;)

           # remove attacked nodes after checking that they are not defended
           if get_attack_category(self.prm['attack']) == 'node':
               if get_defense_category(self.prm['defense']) == 'node':
                   diff = set(self.attacked) - set(self.protected)
                   self.graph_.remove_nodes_from(diff)
               else:
                   self.graph_.remove_nodes_from(self.attacked)

           self.track_simulation(step=0)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L560">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-Defense-track_simulation"><summary><code>Defense.track_simulation(step)</code></summary><div class="api-body"><p>Keeps track of important simulation information at each step of the simulation</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>current simulation iteration</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def track_simulation(self, step):
           &quot;&quot;&quot;
            Keeps track of important simulation information at each step of the simulation

            :param step: current simulation iteration
            &quot;&quot;&quot;

           measure = run_measure(self.graph_, self.prm['robust_measure'])

           ccs = list(nx.connected_components(self.graph_))
           ccs.sort(key=len, reverse=True)
           m = interp1d([0, len(ccs)], [0.15, 1]) if len(ccs) &gt; 0 else None

           status = {}
           for n in self.graph:
               status[n] = 0
               for idx, cc in enumerate(ccs):
                   if n in self.attacked and n not in self.protected:
                       status[n] = 1
                       break
                   elif n in cc:
                       status[n] = float(m(idx))
                       break

           lcc = len(ccs[0]) if len(ccs) &gt; 0 else 0
           self.sim_info[step] = {
               'status':  list(status.values()),
               'failed': len(self.graph) - lcc,
               'measure': measure,
               'protected': self.protected,
               'edges_added': self.protected['added'][0:step] if 'added' in self.protected else [],
               'edges_removed': self.protected['removed'][0:step] if 'removed' in self.protected else []
           }</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L599">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-defenses-Defense-run_single_sim"><summary><code>Defense.run_single_sim()</code></summary><div class="api-body"><p>Run the defense simulation</p><p><strong>Returns</strong> List of steps+1 values. Attack/Defense and non-Crucitti cascades: selected measure. Crucitti: weighted efficiency. SIS: infected count. SIR: recovered count. run_simulation averages these across runs.</p><details class="source-code"><summary>View source code</summary><pre><code>def run_single_sim(self):
           &quot;&quot;&quot;
           Run the defense simulation
           &quot;&quot;&quot;

           for step in range(self.prm['steps']):
               if get_defense_category(self.prm['defense']) == 'edge' and step &lt; len(self.protected['added']):
                   if 'removed' in self.protected and step &lt; len(self.protected['removed']):
                       u, v = self.protected['removed'][step]
                       self.graph_.remove_edge(u, v)

                   u, v = self.protected['added'][step]
                   self.graph_.add_edge(u, v)

               else:
                   print(&quot;Ending defense simulation early, not an 'edge' defense or out of {}s&quot;.format(get_defense_category(self.prm['defense'])))

               self.track_simulation(step + 1)

           results = [self.sim_info[step]['measure'] if self.sim_info[step]['measure'] is not None else 0
                      for step in range(self.prm['steps'] + 1)]
           return results</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/defenses.py#L633">View this source on GitHub</a>.</p></div></details></div>
