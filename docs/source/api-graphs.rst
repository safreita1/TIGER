graph_tiger.graphs
==================

.. raw:: html

   <p>Callable signatures, parameters, return conventions, and source for TIGER 0.7.0. <a href="api.html">All modules</a>.</p><label for="api-filter">Filter functions and methods</label><input id="api-filter" type="search" placeholder="Name, parameter, or description"><p id="api-count" aria-live="polite"></p><div class="api-module" id="module-graphs"><details class="api-entry" id="api-graphs-graph_loader"><summary><code>graph_loader(graph_type, **kwargs)</code></summary><div class="api-body"><p>Loads any of the available graph models, supported user-downloaded datasets and toy graphs.
   In order to get a list of available graph options run 'get_graph_options()'.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>graph_type</code></td><td>a string representing the graph you want to load. For example, 'ER', 'WS', 'BA',
          'oregon_1' (must first download), 'electrical' (must first download)</td></tr></tbody></table></div><p><strong>Returns</strong> an undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def graph_loader(graph_type, **kwargs):
       &quot;&quot;&quot;
       Loads any of the available graph models, supported user-downloaded datasets and toy graphs.
       In order to get a list of available graph options run 'get_graph_options()'.

       :param graph_type: a string representing the graph you want to load. For example, 'ER', 'WS', 'BA',
              'oregon_1' (must first download), 'electrical' (must first download)
       :param kwargs: allows user to specify specific graph model properties
       :return: an undirected NetworkX graph
       &quot;&quot;&quot;
       if graph_type in models.keys():
           graph = models[graph_type](**kwargs)

       elif graph_type in datasets:
           if graph_type in graph_urls:
               download_dataset(graph_type)
           graph = datasets[graph_type]()

       elif graph_type in custom.keys():
           graph = custom[graph_type]()

       else:
           raise ValueError(&quot;Graph not supported. Select from one of the following graphs: {}&quot;.format(get_graph_options()))

       return graph</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L11">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-download_dataset"><summary><code>download_dataset(dataset)</code></summary><div class="api-body"><p>Reading the dataset from the web.</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>dataset</code></td><td>a string representing the dataset to download</td></tr></tbody></table></div><p><strong>Returns</strong> See the implementation below for the exact return contract.</p><details class="source-code"><summary>View source code</summary><pre><code>def download_dataset(dataset):
       &quot;&quot;&quot;
       Reading the dataset from the web.

       :param dataset: a string representing the dataset to download
       &quot;&quot;&quot;
       if dataset not in graph_urls:
           raise ValueError(&quot;dataset '{}' is bundled and does not require downloading&quot;.format(dataset))

       url_path = graph_urls[dataset][0]
       local_path = graph_dir + url_path.split('datasets/')[1]

       if not os.path.exists(local_path):
           os.makedirs(graph_dir, exist_ok=True)
           urllib.request.urlretrieve(url_path, local_path)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L38">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-get_graph_urls"><summary><code>get_graph_urls()</code></summary><div class="api-body"><p>Returns a dictionary of the datasets used in TIGER and the original link to download them</p><p><strong>Returns</strong> dictionary containing links to each dataset</p><details class="source-code"><summary>View source code</summary><pre><code>def get_graph_urls():
       &quot;&quot;&quot;
       Returns a dictionary of the datasets used in TIGER and the original link to download them

       :return: dictionary containing links to each dataset
       &quot;&quot;&quot;
       return graph_urls</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L55">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-get_graph_options"><summary><code>get_graph_options()</code></summary><div class="api-body"><p>Returns a formatted string containing all of the generators, datasets and custom graphs implemented in TIGER</p><p><strong>Returns</strong> formatted string</p><details class="source-code"><summary>View source code</summary><pre><code>def get_graph_options():
       &quot;&quot;&quot;
       Returns a formatted string containing all of the generators, datasets and custom graphs implemented in TIGER

       :return: formatted string
       &quot;&quot;&quot;

       graph_options = {
           'models': list(models.keys()),
           'datasets': list(datasets.keys()),
           'custom': list(custom.keys())
       }

       return json.dumps(graph_options, indent=1)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L64">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-erdos_reyni"><summary><code>erdos_reyni(n, p=None, seed=None)</code></summary><div class="api-body"><p>Returns a Erdos Reyni NetworkX graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>n</code></td><td>number of nodes</td></tr><tr><td><code>p</code></td><td>probability for edge creation</td></tr><tr><td><code>seed</code></td><td>fixes the graph generation process</td></tr></tbody></table></div><p><strong>Returns</strong> a NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def erdos_reyni(n, p=None, seed=None):
       &quot;&quot;&quot;
       Returns a Erdos Reyni NetworkX graph

       :param n: number of nodes
       :param p: probability for edge creation
       :param seed: fixes the graph generation process
       :return: a NetworkX graph
       &quot;&quot;&quot;

       if p is None: p = (1.0 / n + 0.1)
       return nx.generators.erdos_renyi_graph(n=n, p=p, seed=seed)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L85">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-watts_strogatz"><summary><code>watts_strogatz(n, m=4, p=0.05, seed=None)</code></summary><div class="api-body"><p>Returns an ordinary Watts Strogatz NetworkX graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>n</code></td><td>number of nodes</td></tr><tr><td><code>m</code></td><td>each node is joined with its k nearest neighbors in a ring topology</td></tr><tr><td><code>p</code></td><td>probability of rewiring each edge</td></tr><tr><td><code>seed</code></td><td>fixes the graph generation process</td></tr></tbody></table></div><p><strong>Returns</strong> a NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def watts_strogatz(n, m=4, p=0.05, seed=None):
       &quot;&quot;&quot;
       Returns an ordinary Watts Strogatz NetworkX graph

       :param n: number of nodes
       :param m: each node is joined with its k nearest neighbors in a ring topology
       :param p: probability of rewiring each edge
       :param seed: fixes the graph generation process
       :return: a NetworkX graph
       &quot;&quot;&quot;

       return nx.generators.watts_strogatz_graph(n=n, k=m, p=p, seed=seed)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L99">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-barabasi_albert"><summary><code>barabasi_albert(n, m=3, seed=None)</code></summary><div class="api-body"><p>Returns a Barabasi Albert NetworkX graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>n</code></td><td>number of nodes</td></tr><tr><td><code>m</code></td><td>number of edges to attach from a new node to existing nodes</td></tr><tr><td><code>seed</code></td><td>fixes the graph generation process</td></tr></tbody></table></div><p><strong>Returns</strong> a NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def barabasi_albert(n, m=3, seed=None):
       &quot;&quot;&quot;
       Returns a Barabasi Albert NetworkX graph

       :param n: number of nodes
       :param m: number of edges to attach from a new node to existing nodes
       :param seed: fixes the graph generation process
       :return: a NetworkX graph
       &quot;&quot;&quot;

       return nx.generators.barabasi_albert_graph(n=n, m=m, seed=seed)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L113">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-clustered_scale_free"><summary><code>clustered_scale_free(n, m=3, p=0.3, seed=None)</code></summary><div class="api-body"><p>Returns a Clustered Scale-Free NetworkX graph</p><div class="table-scroll"><table><thead><tr><th>Parameter</th><th>Meaning / default</th></tr></thead><tbody><tr><td><code>n</code></td><td>number of nodes</td></tr><tr><td><code>m</code></td><td>the number of random edges to add for each new node</td></tr><tr><td><code>p</code></td><td>probability of adding a triangle after adding a random edge</td></tr><tr><td><code>seed</code></td><td>fixes the graph generation process</td></tr></tbody></table></div><p><strong>Returns</strong> a NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def clustered_scale_free(n, m=3, p=0.3, seed=None):
       &quot;&quot;&quot;
       Returns a Clustered Scale-Free NetworkX graph

       :param n: number of nodes
       :param m: the number of random edges to add for each new node
       :param p:  probability of adding a triangle after adding a random edge
       :param seed: fixes the graph generation process
       :return: a NetworkX graph
       &quot;&quot;&quot;

       return nx.powerlaw_cluster_graph(n=n, m=m, p=p, seed=seed)</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L126">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-wdn_ky2"><summary><code>wdn_ky2()</code></summary><div class="api-body"><p>Returns the graph from: https://uknowledge.uky.edu/wdst/4/,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def wdn_ky2():
       &quot;&quot;&quot;
       Returns the graph from: https://uknowledge.uky.edu/wdst/4/,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.Graph()

       with open(graph_dir + 'ky2.txt') as f:
           lines = f.readlines()
           for line in lines:
               if len(line.split('\t')) == 9:
                   u, v = line.strip().split('\t')[1:3]
                   u = u.strip()
                   v = v.strip()
                   if 'J' in u and 'J' in v:
                       graph.add_edge(u, v)
               else:
                   name, x_pos, y_pos = line.strip().split('\t')
                   name = name.strip()
                   x_pos = float(x_pos.strip())
                   y_pos = float(y_pos.strip())
                   graph.nodes[name]['pos'] = [x_pos, y_pos]

       graph = nx.convert_node_labels_to_integers(graph)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L145">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-as_733"><summary><code>as_733()</code></summary><div class="api-body"><p>Returns the 'as19971108' graph from: http://snap.stanford.edu/data/as-733.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def as_733():
       &quot;&quot;&quot;
       Returns the 'as19971108' graph from: http://snap.stanford.edu/data/as-733.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;as19971108.txt&quot;)
       graph = nx.convert_node_labels_to_integers(graph)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L175">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-p2p_gnuetella08"><summary><code>p2p_gnuetella08()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/p2p-Gnutella08.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def p2p_gnuetella08():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/p2p-Gnutella08.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;p2p-Gnutella08.txt&quot;, create_using=nx.DiGraph()).to_undirected()
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L188">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-ca_grqc"><summary><code>ca_grqc()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/ca-GrQc.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def ca_grqc():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/ca-GrQc.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;ca-GrQc.txt&quot;)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L200">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-cit_hep_th"><summary><code>cit_hep_th()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/cit-HepTh.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def cit_hep_th():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/cit-HepTh.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;cit-HepTh.txt&quot;, create_using=nx.DiGraph()).to_undirected()
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L212">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-wiki_vote"><summary><code>wiki_vote()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/wiki-Vote.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def wiki_vote():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/wiki-Vote.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;wiki-Vote.txt&quot;, create_using=nx.DiGraph()).to_undirected()
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L224">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-email_eu_all"><summary><code>email_eu_all()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/email-EuAll.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def email_eu_all():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/email-EuAll.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;email-EuAll.txt&quot;, create_using=nx.DiGraph()).to_undirected()
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L236">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-dblp"><summary><code>dblp()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/com-DBLP.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def dblp():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/com-DBLP.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;dblp.txt&quot;)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L248">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-ca_astro_ph"><summary><code>ca_astro_ph()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/ca-AstroPh.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def ca_astro_ph():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/ca-AstroPh.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;ca-AstroPh.txt&quot;)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L272">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-ca_hep_th"><summary><code>ca_hep_th()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/cit-HepTh.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def ca_hep_th():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/cit-HepTh.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;ca-HepTh.txt&quot;)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L284">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-enron_email"><summary><code>enron_email()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/email-Enron.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def enron_email():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/email-Enron.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;email-enron.txt&quot;)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L296">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-karate"><summary><code>karate()</code></summary><div class="api-body"><p>Returns the graph from: https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html,</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def karate():
       &quot;&quot;&quot;
       Returns the graph from: https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html,

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       return nx.karate_club_graph()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L308">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-oregeon_1"><summary><code>oregeon_1()</code></summary><div class="api-body"><p>Returns the graph from: https://snap.stanford.edu/data/oregon1_010331.html,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def oregeon_1():
       &quot;&quot;&quot;
       Returns the graph from: https://snap.stanford.edu/data/oregon1_010331.html,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_edgelist(graph_dir + &quot;as-oregon1.txt&quot;)
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L318">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-electrical"><summary><code>electrical()</code></summary><div class="api-body"><p>Returns the graph from: http://konect.cc/networks/opsahl-powergrid/,
   where we preprocess it to only keep the largest connected component</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def electrical():
       &quot;&quot;&quot;
       Returns the graph from: http://konect.cc/networks/opsahl-powergrid/,
       where we preprocess it to only keep the largest connected component

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       graph = nx.read_gml(graph_dir + &quot;power.gml&quot;, label='id')
       return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L330">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-o4_graph"><summary><code>o4_graph()</code></summary><div class="api-body"><p>Returns a 4 node disconnected graph</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def o4_graph():
       &quot;&quot;&quot;
       Returns a 4 node disconnected graph

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_nodes_from([0, 1, 2, 3])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L359">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-p4_graph"><summary><code>p4_graph()</code></summary><div class="api-body"><p>Returns a 4 node path graph</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def p4_graph():
       &quot;&quot;&quot;
       Returns a 4 node path graph

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_edges_from([(0, 1), (1, 2), (2, 3)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L371">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-s4_graph"><summary><code>s4_graph()</code></summary><div class="api-body"><p>Returns a 4 node star graph</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def s4_graph():
       &quot;&quot;&quot;
       Returns a 4 node star graph

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_edges_from([(0, 1), (1, 2), (1, 3)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L383">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-c4_graph"><summary><code>c4_graph()</code></summary><div class="api-body"><p>Returns a 4 node cycle graph</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def c4_graph():
       &quot;&quot;&quot;
       Returns a 4 node cycle graph

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L395">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-k4_1_graph"><summary><code>k4_1_graph()</code></summary><div class="api-body"><p>Returns a 4 node diamond graph (1 diagonal edge)</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def k4_1_graph():
       &quot;&quot;&quot;
       Returns a 4 node diamond graph (1 diagonal edge)

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L407">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-k4_2_graph"><summary><code>k4_2_graph()</code></summary><div class="api-body"><p>Returns a 4 node diamond graph (2 diagonal edges), a.k.a. complete graph</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def k4_2_graph():
       &quot;&quot;&quot;
       Returns a 4 node diamond graph (2 diagonal edges), a.k.a. complete graph

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (2, 3)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L419">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-two_c4_0_bridge"><summary><code>two_c4_0_bridge()</code></summary><div class="api-body"><p>Returns two disconnected 4 node cycle graphs</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def two_c4_0_bridge():
       &quot;&quot;&quot;
       Returns two disconnected 4 node cycle graphs

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                         (4, 5), (5, 6), (6, 7), (7, 4)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L431">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-two_c4_1_bridge"><summary><code>two_c4_1_bridge()</code></summary><div class="api-body"><p>Returns two 4 node cycle graphs connected by 1 edge</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def two_c4_1_bridge():
       &quot;&quot;&quot;
       Returns two 4 node cycle graphs connected by 1 edge

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.Graph()
       G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                         (4, 5), (5, 6), (6, 7), (7, 4),
                         (2, 4)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L444">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-two_c4_2_bridge"><summary><code>two_c4_2_bridge()</code></summary><div class="api-body"><p>Returns two 4 node cycle graphs connected by 2 edges</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def two_c4_2_bridge():
       &quot;&quot;&quot;
       Returns two 4 node cycle graphs connected by 2 edges

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.MultiGraph()
       G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                         (4, 5), (5, 6), (6, 7), (7, 4),
                         (2, 4), (2, 4)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L458">View this source on GitHub</a>.</p></div></details><details class="api-entry" id="api-graphs-two_c4_3_bridge"><summary><code>two_c4_3_bridge()</code></summary><div class="api-body"><p>Returns two 4 node cycle graphs connected by 3 edges</p><p><strong>Returns</strong> undirected NetworkX graph</p><details class="source-code"><summary>View source code</summary><pre><code>def two_c4_3_bridge():
       &quot;&quot;&quot;
       Returns two 4 node cycle graphs connected by 3 edges

       :return: undirected NetworkX graph
       &quot;&quot;&quot;

       G = nx.MultiGraph()
       G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3),
                         (4, 5), (5, 6), (6, 7), (7, 4),
                         (2, 4), (2, 4), (2, 4)])
       return G</code></pre></details><p class="source-link"><a href="https://github.com/safreita1/TIGER/blob/0.7.0/graph_tiger/graphs.py#L472">View this source on GitHub</a>.</p></div></details></div>
