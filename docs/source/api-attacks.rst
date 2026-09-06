graph_tiger.attacks
===================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.6.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-attacks"><details class="method-keys"><summary>Accepted method names (17)</summary><div class="table-scroll"><table><thead><tr><th>Method key</th><th>Function</th></tr></thead><tbody><tr><td><code>ns_node</code></td><td><a href="#api-attacks-get_node_ns">get_node_ns</a></td></tr><tr><td><code>pr_node</code></td><td><a href="#api-attacks-get_node_pr">get_node_pr</a></td></tr><tr><td><code>eig_node</code></td><td><a href="#api-attacks-get_node_eig">get_node_eig</a></td></tr><tr><td><code>id_node</code></td><td><a href="#api-attacks-get_node_id">get_node_id</a></td></tr><tr><td><code>rd_node</code></td><td><a href="#api-attacks-get_node_rd">get_node_rd</a></td></tr><tr><td><code>ib_node</code></td><td><a href="#api-attacks-get_node_ib">get_node_ib</a></td></tr><tr><td><code>rb_node</code></td><td><a href="#api-attacks-get_node_rb">get_node_rb</a></td></tr><tr><td><code>rnd_node</code></td><td><a href="#api-attacks-get_node_rnd">get_node_rnd</a></td></tr><tr><td><code>ns_line_edge</code></td><td><a href="#api-attacks-get_edge_line_ns">get_edge_line_ns</a></td></tr><tr><td><code>pr_line_edge</code></td><td><a href="#api-attacks-get_edge_line_pr">get_edge_line_pr</a></td></tr><tr><td><code>eig_line_edge</code></td><td><a href="#api-attacks-get_edge_line_eig">get_edge_line_eig</a></td></tr><tr><td><code>deg_line_edge</code></td><td><a href="#api-attacks-get_edge_line_deg">get_edge_line_deg</a></td></tr><tr><td><code>id_edge</code></td><td><a href="#api-attacks-get_edge_id">get_edge_id</a></td></tr><tr><td><code>rd_edge</code></td><td><a href="#api-attacks-get_edge_rd">get_edge_rd</a></td></tr><tr><td><code>ib_edge</code></td><td><a href="#api-attacks-get_edge_ib">get_edge_ib</a></td></tr><tr><td><code>rb_edge</code></td><td><a href="#api-attacks-get_edge_rb">get_edge_rb</a></td></tr><tr><td><code>rnd_edge</code></td><td><a href="#api-attacks-get_edge_rnd">get_edge_rnd</a></td></tr></tbody></table></div></details><details class="api-entry" id="api-attacks-run_attack_method"><summary><code>run_attack_method(graph, method, k=3, approx=None, seed=None)</code></summary><div class="api-body"><p>Runs a specified attack on an undirected graph, returning a list of nodes or edges.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>method</code></td><td>a string representing one of the attack methods</td></tr><tr><td><code>k</code></td><td>number of nodes or edges to attack</td></tr><tr><td><code>approx</code></td><td>attack approximation parameter (not available for every measure)</td></tr><tr><td><code>seed</code></td><td>sets the seed in order to obtain reproducible attacks</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes or edges selected for attack</p><details class="source-code"><summary>View source code</summary><pre><code>def run_attack_method(graph, method, k=3, approx=None, seed=None):
       &quot;&quot;&quot;
       Runs a specified attack on an undirected graph, returning a list of nodes or edges.

       :param graph: an undirected NetworkX graph
       :param method: a string representing one of the attack methods
       :param k: number of nodes or edges to attack
       :param approx: attack approximation parameter (not available for every measure)
       :param seed: sets the seed in order to obtain reproducible attacks
       :return: a list of nodes or edges selected for attack
       &quot;&quot;&quot;

       if method not in methods:
           raise ValueError(&quot;attack method '{}' is not implemented&quot;.format(method))
       if not isinstance(k, (int, np.integer)) or k &lt; 0:
           raise ValueError('k must be a nonnegative integer')
       if k == 0:
           return []

       category = get_attack_category(method)
       feasible = len(graph) if category == 'node' else len(graph.edges)
       if k &gt; feasible:
           raise ValueError('k exceeds the number of available {}s'.format(category))

       rng = np.random.RandomState(seed)

       if method in ['rnd_node', 'rnd_edge']:
           return methods[method](graph, k, rng=rng)
       if method in ['ib_node', 'rb_node', 'ib_edge', 'rb_edge']:
           approx = np.inf if approx is None else approx
           return methods[method](graph, k, approx=approx, seed=seed)

       return methods[method](graph, k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L13">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_attack_methods"><summary><code>get_attack_methods()</code></summary><div class="api-body"><p>Gets a list of available attach methods as a list of functions</p><p><strong>Returns</strong> a list of all attack functions</p><details class="source-code"><summary>View source code</summary><pre><code>def get_attack_methods():
       &quot;&quot;&quot;
       Gets a list of available attach methods as a list of functions

       :return: a list of all attack functions
       &quot;&quot;&quot;

       return methods.keys()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L47">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_attack_category"><summary><code>get_attack_category(method)</code></summary><div class="api-body"><p>Gets the attack category e.g., 'node', 'edge' attack</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>method</code></td><td>a string representing the attack method</td></tr></tbody></table></div><p><strong>Returns</strong> a string representing the attack type ('node' or 'edge')</p><details class="source-code"><summary>View source code</summary><pre><code>def get_attack_category(method):
       &quot;&quot;&quot;
       Gets the attack category e.g., 'node', 'edge' attack

       :param method: a string representing the attack method

       :return: a string representing the attack type ('node' or 'edge')
       &quot;&quot;&quot;

       category = None

       if method in categories:
           category = categories[method]

       return category</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L57">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_ns"><summary><code>get_node_ns(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to attack based on the NetShield algorithm .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of node labels to attack</p><p>References: <a href="references.html#ref-tong2010vulnerability">On the vulnerability of large graphs (2010)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_ns(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to attack based on the NetShield algorithm :cite:`tong2010vulnerability`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack
       :return: a list of node labels to attack
       &quot;&quot;&quot;

       if scipy.sparse.issparse(graph):
           sparse_graph = graph
           nodes_graph = list(range(graph.shape[0]))
       else:
           sparse_graph = get_sparse_graph(graph)
           nodes_graph = list(graph.nodes)

       if k == 0:
           return []
       if k &lt; 0 or k &gt; len(nodes_graph):
           raise ValueError('k must satisfy 0 &lt;= k &lt;= number of nodes')
       if len(nodes_graph) == 1:
           return nodes_graph.copy()

       lam, u = eigsh(sparse_graph, k=1, which='LA')
       lam = lam[0]

       u = np.abs(np.real(u).flatten())
       v = (2 * lam * np.ones(len(u))) * np.power(u, 2)

       nodes = []
       for _ in range(k):
           score = v.copy()
           if len(nodes) &gt; 0:
               B = sparse_graph[:, nodes]
               b = np.asarray(B.dot(u[nodes])).flatten()
               score = score - 2 * b * u

           score[nodes] = -np.inf
           nodes.append(int(np.argmax(score)))

       return [nodes_graph[idx] for idx in nodes]</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L74">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_pr"><summary><code>get_node_pr(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to attack based on top PageRank entries :citepage1999pagerank.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to attack</p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_pr(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to attack based on top PageRank entries :cite`page1999pagerank`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack

       :return: a list of nodes to attack
       &quot;&quot;&quot;

       centrality = nx.pagerank(graph, alpha=0.85)
       nodes = heapq.nlargest(k, centrality, key=centrality.get)

       return nodes</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L116">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_eig"><summary><code>get_node_eig(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to attack based on top eigenvector centrality entries</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to attack</p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_eig(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to attack based on top eigenvector centrality entries

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack
       :return: a list of nodes to attack
       &quot;&quot;&quot;

       centrality = nx.eigenvector_centrality(graph, tol=1E-3, max_iter=500)
       nodes = heapq.nlargest(k, centrality, key=centrality.get)

       return nodes</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L132">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_id"><summary><code>get_node_id(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to attack based on Initial Degree (ID) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to attack</p><p>References: <a href="references.html#ref-beygelzimer2005improving">Improving network robustness by edge modification (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_id(graph,  k=3):
       &quot;&quot;&quot;
       Get k nodes to attack based on Initial Degree (ID) Removal :cite:`beygelzimer2005improving`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack

       :return: a list of nodes to attack
       &quot;&quot;&quot;

       centrality = dict(graph.degree())
       nodes = heapq.nlargest(k, centrality, key=centrality.get)

       return nodes</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L147">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_rd"><summary><code>get_node_rd(graph, k=3)</code></summary><div class="api-body"><p>Get k nodes to attack based on Recalculated Degree (RD) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to attack</p><p>References: <a href="references.html#ref-beygelzimer2005improving">Improving network robustness by edge modification (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_rd(graph, k=3):
       &quot;&quot;&quot;
       Get k nodes to attack based on Recalculated Degree (RD) Removal :cite:`beygelzimer2005improving`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack

       :return: a list of nodes to attack
       &quot;&quot;&quot;
       graph_ = graph.copy()

       nodes = []
       for _ in range(k):
           u = get_node_id(graph_, k=1)[0]

           nodes.append(u)
           graph_.remove_node(u)

       return nodes</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L163">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_ib"><summary><code>get_node_ib(graph, k=3, approx=np.inf, seed=None)</code></summary><div class="api-body"><p>Get k nodes to attack based on Initial Betweenness (IB) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr><tr><td><code>approx</code></td><td>number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
       is the number of nodes in the graph</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to attack</p><p>References: <a href="references.html#ref-beygelzimer2005improving">Improving network robustness by edge modification (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_ib(graph, k=3, approx=np.inf, seed=None):
       &quot;&quot;&quot;
       Get k nodes to attack based on Initial Betweenness (IB) Removal :cite:`beygelzimer2005improving`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack
       :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
           is the number of nodes in the graph

       :return: a list of nodes to attack
       &quot;&quot;&quot;

       samples = None if np.isinf(approx) or approx &gt;= len(graph) else int(approx)
       centrality = nx.betweenness_centrality(graph, k=samples, seed=seed)
       nodes = heapq.nlargest(k, centrality, key=centrality.get)

       return nodes</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L184">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_rb"><summary><code>get_node_rb(graph, k=3, approx=np.inf, seed=None)</code></summary><div class="api-body"><p>Get k nodes to attack based on Recalculated Betweenness (RB) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr><tr><td><code>approx</code></td><td>number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
       is the number of nodes in the graph</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to attack</p><p>References: <a href="references.html#ref-beygelzimer2005improving">Improving network robustness by edge modification (2005)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_rb(graph, k=3, approx=np.inf, seed=None):
       &quot;&quot;&quot;
       Get k nodes to attack based on Recalculated Betweenness (RB) Removal :cite:`beygelzimer2005improving`.

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack
       :param approx: number of nodes to approximate the betweenness centrality, k=0.1n is a good approximation, where n
           is the number of nodes in the graph

       :return: a list of nodes to attack
       &quot;&quot;&quot;
       graph_ = graph.copy()

       nodes = []
       for _ in range(k):
           u = get_node_ib(graph_, k=1, approx=approx, seed=seed)[0]

           nodes.append(u)
           graph_.remove_node(u)

       return nodes</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L203">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_node_rnd"><summary><code>get_node_rnd(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Randomly select k distinct nodes to attack</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of nodes to attack</td></tr><tr><td><code>rng</code></td><td>Argument used by the implementation shown below. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> a list of nodes to attack</p><details class="source-code"><summary>View source code</summary><pre><code>def get_node_rnd(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Randomly select k distinct nodes to attack

       :param graph: an undirected NetworkX graph
       :param k: number of nodes to attack

       :return: a list of nodes to attack
       &quot;&quot;&quot;
       rng = np.random if rng is None else rng
       return rng.choice(list(graph.nodes), k, replace=False).tolist()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L226">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_line_ns"><summary><code>get_edge_line_ns(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to attack using Netshield by transforming the graph into a line graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-tong2010vulnerability">On the vulnerability of large graphs (2010)</a>; <a href="references.html#ref-tong2012gelling">Gelling, and melting, large graphs by edge manipulation (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_line_ns(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to attack using Netshield by transforming the graph into a line graph :cite:`tong2010vulnerability,tong2012gelling`

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;
       line_graph = nx.line_graph(graph)

       return get_node_ns(line_graph, k=k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L239">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_line_pr"><summary><code>get_edge_line_pr(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to attack using PageRank by transforming the graph into a line graph .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-tong2012gelling">Gelling, and melting, large graphs by edge manipulation (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_line_pr(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to attack using PageRank by transforming the graph into a line graph :cite:`tong2012gelling`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;
       line_graph = nx.line_graph(graph)

       return get_node_pr(line_graph, k=k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L253">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_line_eig"><summary><code>get_edge_line_eig(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to attack using eigenvector centrality by transforming the graph into a line graph .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-tong2012gelling">Gelling, and melting, large graphs by edge manipulation (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_line_eig(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to attack using eigenvector centrality by transforming the graph into a line graph :cite:`tong2012gelling`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;
       line_graph = nx.line_graph(graph)

       return get_node_eig(line_graph, k=k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L267">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_line_deg"><summary><code>get_edge_line_deg(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to attack using degree centrality by transforming the graph into a line graph .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-tong2012gelling">Gelling, and melting, large graphs by edge manipulation (2012)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_line_deg(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to attack using degree centrality by transforming the graph into a line graph :cite:`tong2012gelling`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;
       line_graph = nx.line_graph(graph)

       return get_node_id(line_graph, k=k)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L281">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_id"><summary><code>get_edge_id(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to attack based on Initial Degree (ID) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_id(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to attack based on Initial Degree (ID) Removal :cite:`holme2002attack`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;

       centrality = {(u, v): graph.degree(u) * graph.degree(v) for u, v in graph.edges}
       edges = heapq.nlargest(k, centrality, key=centrality.get)

       return edges</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L295">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_rd"><summary><code>get_edge_rd(graph, k=3)</code></summary><div class="api-body"><p>Get k edges to attack based on Recalculated Degree (RD) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_rd(graph, k=3):
       &quot;&quot;&quot;
       Get k edges to attack based on Recalculated Degree (RD) Removal :cite:`holme2002attack`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;
       graph_ = graph.copy()

       edges = []
       for _ in range(k):
           u, v = get_edge_id(graph_, k=1)[0]

           edges.append((u, v))
           graph_.remove_edge(u, v)

       return edges</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L311">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_ib"><summary><code>get_edge_ib(graph, k=3, approx=np.inf, seed=None)</code></summary><div class="api-body"><p>Get k edges to attack based on Initial Betweenness (IB) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr><tr><td><code>approx</code></td><td>number of edges to approximate the betweenness centrality</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_ib(graph, k=3, approx=np.inf, seed=None):
       &quot;&quot;&quot;
       Get k edges to attack based on Initial Betweenness (IB) Removal :cite:`holme2002attack`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack
       :param approx: number of edges to approximate the betweenness centrality

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;

       samples = None if np.isinf(approx) or approx &gt;= len(graph) else int(approx)
       centrality = nx.edge_betweenness_centrality(graph, k=samples, seed=seed)
       edges = heapq.nlargest(k, centrality, key=centrality.get)

       return edges</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L332">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_rb"><summary><code>get_edge_rb(graph, k=3, approx=np.inf, seed=None)</code></summary><div class="api-body"><p>Get k edges to attack based on Recalculated Betweenness (RB) Removal .</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr><tr><td><code>approx</code></td><td>number of edges to approximate the betweenness centrality</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><p>References: <a href="references.html#ref-holme2002attack">Attack vulnerability of complex networks (2002)</a></p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_rb(graph, k=3, approx=np.inf, seed=None):
       &quot;&quot;&quot;
       Get k edges to attack based on Recalculated Betweenness (RB) Removal :cite:`holme2002attack`.

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack
       :param approx: number of edges to approximate the betweenness centrality

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;

       top_edges = []
       graph_ = graph.copy()

       for _ in range(k):
           u, v = get_edge_ib(graph_, k=1, approx=approx, seed=seed)[0]

           top_edges.append((u, v))
           graph_.remove_edge(u, v)

       return top_edges</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L350">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-get_edge_rnd"><summary><code>get_edge_rnd(graph, k=3, rng=None)</code></summary><div class="api-body"><p>Randomly select k edges to attack</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>an undirected NetworkX graph</td></tr><tr><td><code>k</code></td><td>number of edges to attack</td></tr><tr><td><code>rng</code></td><td>Argument used by the implementation shown below. Default: None.</td></tr></tbody></table></div><p><strong>Returns</strong> a list of edge tuples to attack</p><details class="source-code"><summary>View source code</summary><pre><code>def get_edge_rnd(graph, k=3, rng=None):
       &quot;&quot;&quot;
       Randomly select k edges to attack

       :param graph: an undirected NetworkX graph
       :param k: number of edges to attack

       :return: a list of edge tuples to attack
       &quot;&quot;&quot;
       edges = list(graph.edges)
       rng = np.random if rng is None else rng
       idx = rng.choice(len(edges), k, replace=False)
       rnd_edges = [edges[i] for i in idx]

       return rnd_edges</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L373">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-Attack"><summary><code>Attack(graph, runs=10, steps=50, attack='id_node', defense=None, k_d=0, **kwargs)</code></summary><div class="api-body"><p>This class simulates a variety of attack strategies on an undirected NetworkX graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph</code></td><td>Undirected simple NetworkX graph. See Network inputs for node ordering, conversions, and weight conventions.</td></tr><tr><td><code>runs</code></td><td>Positive integer; number of realizations for run_simulation, default 10 in simulation subclasses.</td></tr><tr><td><code>steps</code></td><td>Nonnegative transition limit. Returned trajectories have steps+1 entries, including index 0.</td></tr><tr><td><code>attack</code></td><td>Attack-selection key or None; simulation subclasses default to id_node. Budgets determine whether an initial attack is performed.</td></tr><tr><td><code>defense</code></td><td>None or a supported defense key. Cascade node defense doubles selected capacities, but does not cancel initial failures.</td></tr><tr><td><code>k_d</code></td><td>Nonnegative node/edge defense budget, default 0. Attack: number selected for protection/intervention. Defense: number of proposed edge changes. Cascading: defense budget.</td></tr><tr><td><code>seed</code></td><td>Integer or None (default 1 for simulations). Initializes per-instance random generators; resets advance to a new reproducible realization.</td></tr><tr><td><code>plot_transition</code></td><td>Boolean, default False. Save selected first-run snapshots.</td></tr><tr><td><code>gif_animation</code></td><td>Boolean, default False. Write an MP4 with FFmpeg on supported non-Windows platforms.</td></tr><tr><td><code>gif_snaps</code></td><td>Boolean, default False. Save animation frames when that workflow runs.</td></tr><tr><td><code>node_style</code></td><td>None (dataset coordinates or spectral layout) or force_atlas (optional dependency).</td></tr><tr><td><code>edge_style</code></td><td>None (straight edges) or bundled (optional dependency).</td></tr><tr><td><code>fa_iter</code></td><td>Positive integer, default 200; ForceAtlas2 iterations.</td></tr><tr><td><code>robust_measure</code></td><td>Measure key; default largest_connected_component for attack, defense and non-Crucitti cascades. Crucitti overrides it with weighted efficiency.</td></tr><tr><td><code>attack_approx</code></td><td>Sampled source-node count for betweenness attack selection; default None means exact. Not a fraction or edge count.</td></tr></tbody></table></div><p><strong>Returns</strong> A configured Attack instance. Construction initializes the graph, state and random generators.</p><p><a href="reproducibility.html">Output shapes, state lifecycle, stopping and random-seed conventions</a></p><details class="source-code"><summary>View source code</summary><pre><code>class Attack(Simulation):
       &quot;&quot;&quot;
       This class simulates a variety of attack strategies on an undirected NetworkX graph

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
           self.protected = []
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
           if self.prm['attack'] is not None and self.prm['steps'] &gt; 0:
               self.attacked = run_attack_method(self.graph_, self.prm['attack'], self.prm['steps'], approx=self.prm['attack_approx'], seed=self.get_random_seed())

           elif self.prm['attack'] is not None:
               print(self.prm['attack'], &quot;not available or k &lt;= 0&quot;)

           # defended nodes or edges
           if self.prm['defense'] is not None and self.prm['k_d'] &gt; 0:
               from graph_tiger.defenses import get_defense_category, run_defense_method

               if get_defense_category(self.prm['defense']) == 'node':
                   self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.get_random_seed())

               elif get_defense_category(self.prm['defense']) == 'edge':
                   protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.get_random_seed())

                   self.graph_.add_edges_from(protected['added'])
                   if 'removed' in protected:
                       self.graph_.remove_edges_from(protected['removed'])

           elif self.prm['defense'] is not None:
               print(self.prm['defense'], &quot;not available or k &lt;= 0&quot;)

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
                   if n in self.attacked[0:step] and n not in self.protected:
                       status[n] = 1
                       break
                   elif n in cc:
                       status[n] = float(m(idx))
                       break

           failed = set(self.attacked[0:step]).difference(self.protected)
           self.sim_info[step] = {
               'status':  list(status.values()),
               'failed': len(failed),
               'measure': measure,
               'protected': self.protected
           }

       def run_single_sim(self):
           &quot;&quot;&quot;
           Run the attack simulation
           &quot;&quot;&quot;

           for step in range(self.prm['steps']):
               if step &lt; len(self.attacked) and len(self.attacked) &gt; 0:
                   v = self.attacked[step]

                   if get_attack_category(self.prm['attack']) == 'edge':
                       self.graph_.remove_edge(v[0], v[1])

                   elif get_attack_category(self.prm['attack']) == 'node' and v not in self.protected:
                       self.graph_.remove_node(v)

               else:
                   print(&quot;Ending attack simulation early, ran of {}s&quot;.format(get_attack_category(self.prm['attack'])))

               self.track_simulation(step + 1)

           results = [self.sim_info[step]['measure'] if self.sim_info[step]['measure'] is not None else 0
                      for step in range(self.prm['steps'] + 1)]
           return results</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L434">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-Attack-reset_simulation"><summary><code>Attack.reset_simulation()</code></summary><div class="api-body"><p>Resets the simulation between each run</p><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def reset_simulation(self):
           &quot;&quot;&quot;
           Resets the simulation between each run
           &quot;&quot;&quot;

           self.begin_reset()

           self.graph_ = self.graph.copy()
           self.attacked = []
           self.protected = []
           self.connectivity = []

           # attacked nodes or edges
           if self.prm['attack'] is not None and self.prm['steps'] &gt; 0:
               self.attacked = run_attack_method(self.graph_, self.prm['attack'], self.prm['steps'], approx=self.prm['attack_approx'], seed=self.get_random_seed())

           elif self.prm['attack'] is not None:
               print(self.prm['attack'], &quot;not available or k &lt;= 0&quot;)

           # defended nodes or edges
           if self.prm['defense'] is not None and self.prm['k_d'] &gt; 0:
               from graph_tiger.defenses import get_defense_category, run_defense_method

               if get_defense_category(self.prm['defense']) == 'node':
                   self.protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.get_random_seed())

               elif get_defense_category(self.prm['defense']) == 'edge':
                   protected = run_defense_method(self.graph_, self.prm['defense'], self.prm['k_d'], seed=self.get_random_seed())

                   self.graph_.add_edges_from(protected['added'])
                   if 'removed' in protected:
                       self.graph_.remove_edges_from(protected['removed'])

           elif self.prm['defense'] is not None:
               print(self.prm['defense'], &quot;not available or k &lt;= 0&quot;)

           self.track_simulation(step=0)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L475">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-Attack-track_simulation"><summary><code>Attack.track_simulation(step)</code></summary><div class="api-body"><p>Keeps track of important simulation information at each step of the simulation</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>step</code></td><td>current simulation iteration</td></tr></tbody></table></div><p><strong>Returns</strong> No result value is documented; this method updates instance state or writes plotting artifacts.</p><details class="source-code"><summary>View source code</summary><pre><code>def track_simulation(self, step):
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
                   if n in self.attacked[0:step] and n not in self.protected:
                       status[n] = 1
                       break
                   elif n in cc:
                       status[n] = float(m(idx))
                       break

           failed = set(self.attacked[0:step]).difference(self.protected)
           self.sim_info[step] = {
               'status':  list(status.values()),
               'failed': len(failed),
               'measure': measure,
               'protected': self.protected
           }</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L513">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-attacks-Attack-run_single_sim"><summary><code>Attack.run_single_sim()</code></summary><div class="api-body"><p>Run the attack simulation</p><p><strong>Returns</strong> List of steps+1 values. Attack/Defense and non-Crucitti cascades: selected measure. Crucitti: weighted efficiency. SIS: infected count. SIR: recovered count. run_simulation averages these across runs.</p><details class="source-code"><summary>View source code</summary><pre><code>def run_single_sim(self):
           &quot;&quot;&quot;
           Run the attack simulation
           &quot;&quot;&quot;

           for step in range(self.prm['steps']):
               if step &lt; len(self.attacked) and len(self.attacked) &gt; 0:
                   v = self.attacked[step]

                   if get_attack_category(self.prm['attack']) == 'edge':
                       self.graph_.remove_edge(v[0], v[1])

                   elif get_attack_category(self.prm['attack']) == 'node' and v not in self.protected:
                       self.graph_.remove_node(v)

               else:
                   print(&quot;Ending attack simulation early, ran of {}s&quot;.format(get_attack_category(self.prm['attack'])))

               self.track_simulation(step + 1)

           results = [self.sim_info[step]['measure'] if self.sim_info[step]['measure'] is not None else 0
                      for step in range(self.prm['steps'] + 1)]
           return results</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.6.0/graph_tiger/attacks.py#L545">View this source on GitHub</a>.</p></div></details></div>
