Loading graphs
==============

.. raw:: html

   <p>TIGER works with NetworkX graph objects. You can generate a graph, load a named dataset or toy example through <code>graph_loader</code>, or read your own file with NetworkX and pass the resulting graph directly to TIGER. Your own graphs do not need to be registered with the loader.</p>

.. _network-inputs-built-in:

.. raw:: html

   <span id="built-in"></span>

What is supported out of the box?
---------------------------------

.. raw:: html

   <p>TIGER 0.6.0 provides four generators, fourteen named datasets, and ten toy graphs. Names are case-sensitive. Print the installed version’s catalog rather than guessing a loader key:</p><pre><code>from graph_tiger.graphs import graph_loader, get_graph_options

   print(get_graph_options())
   G = graph_loader(&quot;karate&quot;)
   print(G.number_of_nodes(), G.number_of_edges())</code></pre>

.. _network-inputs-generators:

.. raw:: html

   <span id="generators"></span>

Generate a synthetic network
----------------------------

.. raw:: html

   <p>Generator arguments are passed through to the selected model. Fix the graph seed to hold topology constant while comparing simulation settings. A simulation seed is separate: it controls events on that graph.</p><div class="table-scroll"><table><thead><tr><th>Key</th><th>Model</th><th>Parameters</th><th>Example arguments</th></tr></thead><tbody><tr><td>ER</td><td>Erdős–Rényi independent-edge graph</td><td>n: nodes; p: edge probability</td><td>n=100, p=0.05, seed=17</td></tr><tr><td>WS</td><td>Watts–Strogatz small-world graph</td><td>n: nodes; m: ring neighbors per node; p: rewiring probability</td><td>n=100, m=4, p=0.1, seed=17</td></tr><tr><td>BA</td><td>Barabási–Albert preferential attachment</td><td>n: nodes; m: edges per new node</td><td>n=100, m=3, seed=17</td></tr><tr><td>CSF</td><td>Clustered scale-free graph</td><td>n: nodes; m: edges per new node; p: triangle-formation probability</td><td>n=100, m=3, p=0.3, seed=17</td></tr></tbody></table></div><pre><code>ER = graph_loader(&quot;ER&quot;, n=100, p=0.05, seed=17)
   WS = graph_loader(&quot;WS&quot;, n=100, m=4, p=0.1, seed=17)
   BA = graph_loader(&quot;BA&quot;, n=100, m=3, seed=17)
   CSF = graph_loader(&quot;CSF&quot;, n=100, m=3, p=0.3, seed=17)</code></pre><p>Use valid NetworkX generator parameters: for example, BA and CSF require 1 ≤ m &lt; n. In WS, <code>m</code> is the number of ring neighbors, not the number of edges added by a growing node; choose an even value for the intended symmetric ring degree. These generators need no dataset downloads.</p>

.. _network-inputs-datasets:

.. raw:: html

   <span id="datasets"></span>

Load a named dataset
--------------------

.. raw:: html

   <p>The same loader reads each dataset below. Except for karate, it fetches the required file when it is absent from the selected dataset directory; subsequent loads reuse the file. Built-in does not mean every dataset is already bundled or that the first load works offline.</p><pre><code>G = graph_loader(&quot;ky2&quot;)
   power = graph_loader(&quot;electrical&quot;)
   internet = graph_loader(&quot;as_733&quot;)</code></pre><div class="table-scroll"><table><thead><tr><th>Loader key</th><th>Network</th><th>Preprocessing / notes</th></tr></thead><tbody><tr><td>karate</td><td>Zachary’s 34-node karate-club social network</td><td>Available through NetworkX; no download. Preserves its supplied edge weights.</td></tr><tr><td>ky2</td><td>Kentucky water-distribution network</td><td>Junction-to-junction pipe links, node positions, integer relabeling, largest connected component. Not a road network or a hydraulic simulator.</td></tr><tr><td>electrical</td><td>Western-US power-grid topology</td><td>Reads power.gml and retains its largest component; 4,941 nodes in the documented study.</td></tr><tr><td>as_733</td><td>Internet autonomous systems, AS-733 snapshot as19971108</td><td>Largest component; integer relabeling.</td></tr><tr><td>oregon_1</td><td>Internet autonomous systems, Oregon-1</td><td>Largest component; a different dataset from AS-733.</td></tr><tr><td>p2p_gnuetella08</td><td>Gnutella peer-to-peer network</td><td>Converts to undirected; largest component. Use this exact loader spelling.</td></tr><tr><td>wiki_vote</td><td>Wikipedia voting network</td><td>Converts to undirected; largest component. Vote direction is discarded.</td></tr><tr><td>email_eu_all</td><td>EU email network</td><td>Converts to undirected; largest component.</td></tr><tr><td>enron_email</td><td>Enron email network</td><td>Undirected edge list; largest component.</td></tr><tr><td>dblp</td><td>DBLP coauthorship network</td><td>Undirected edge list; largest component.</td></tr><tr><td>ca_grqc</td><td>General relativity / quantum cosmology collaborations</td><td>Undirected edge list; largest component.</td></tr><tr><td>ca_hep_th</td><td>High-energy theory collaborations</td><td>Undirected edge list; largest component.</td></tr><tr><td>ca_astro_ph</td><td>Astrophysics collaborations</td><td>Undirected edge list; largest component.</td></tr><tr><td>cit_hep_th</td><td>High-energy theory citations</td><td>Converts to undirected; largest component. Citation direction is discarded.</td></tr></tbody></table></div><p>Most empirical loaders retain only the largest connected component. That changes the population under study; their returned graphs are not necessarily the complete raw datasets. KY-2 contains 809 nodes in the documented experiments. <a href="guide-results/manifest.json">Recorded dataset fingerprints and preprocessing</a> identify the exact study inputs.</p>

.. _network-inputs-dataset-files:

.. raw:: html

   <span id="dataset-files"></span>

Dataset files and cache location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>TIGER uses its bundled datasets directory if that directory exists; otherwise it uses <code>~/.graph_tiger/datasets</code> under your user directory. The chosen directory must be writable for a missing-file download. Inspect the current location and source URLs, or select a writable directory before loading:</p><pre><code>from pathlib import Path
   from graph_tiger import graphs

   print(graphs.graph_dir)
   print(graphs.get_graph_urls()[&quot;ky2&quot;])

   cache = Path(&quot;data/tiger&quot;).resolve()
   cache.mkdir(parents=True, exist_ok=True)
   graphs.graph_dir = str(cache) + &quot;/&quot;
   G = graphs.graph_loader(&quot;ky2&quot;)</code></pre><p><code>get_graph_urls()</code> maps downloadable keys to a file URL and its original source URL. For offline use, put the expected files in the selected directory first; for example, <code>ky2.txt</code> and <code>power.gml</code>. A missing file still requires network access. Dataset loaders do not use generator parameters such as <code>n</code> to select a smaller sample.</p>

.. _network-inputs-toy-graphs:

.. raw:: html

   <span id="toy-graphs"></span>

Load a small toy graph
----------------------

.. raw:: html

   <p>These examples require no downloads and are useful for checking a measure or following individual simulation steps.</p><div class="table-scroll"><table><thead><tr><th>Key</th><th>Structure</th></tr></thead><tbody><tr><td>o4</td><td>Four isolated nodes</td></tr><tr><td>p4</td><td>Four-node path</td></tr><tr><td>s4</td><td>Four-node star</td></tr><tr><td>c4</td><td>Four-node cycle</td></tr><tr><td>k4-1</td><td>Four-node complete graph minus one edge</td></tr><tr><td>k4-2</td><td>Four-node complete graph</td></tr><tr><td>c4_no_bridge</td><td>Two disconnected four-node cycles</td></tr><tr><td>c4_1_bridge</td><td>Two cycles joined by one link</td></tr><tr><td>c4_2_bridge</td><td>Two cycles joined by two links</td></tr><tr><td>c4_3_bridge</td><td>Two cycles joined by three parallel links; returns a MultiGraph</td></tr></tbody></table></div><pre><code>G = graph_loader(&quot;p4&quot;)
   print(list(G.edges()))</code></pre><p>The parallel-link example <code>c4_3_bridge</code> is an exception to the simple-graph convention. Its availability does not imply that every measure, attack, or simulation supports parallel edges. Use a simple undirected graph for the main workflows in these guides.</p>

.. _network-inputs-your-own:

.. raw:: html

   <span id="your-own"></span>

Load your own network
---------------------

.. _network-inputs-edge-list:

.. raw:: html

   <span id="edge-list"></span>

Read a two-column edge list
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>For a whitespace-separated file with one pair of node labels per line, use <code>nx.read_edgelist</code>. This example treats links as unweighted and keeps every component represented in the file. An edge list cannot represent isolated nodes unless you load those node IDs separately and add them with <code>G.add_nodes_from(node_ids)</code>.</p><pre><code>import networkx as nx
   from graph_tiger.measures import run_measure
   from graph_tiger.attacks import Attack

   G = nx.read_edgelist(&quot;network.edgelist&quot;, create_using=nx.Graph(),
                        nodetype=str, data=False)
   print(G.number_of_nodes(), G.number_of_edges())
   print(&quot;Self-loops:&quot;, nx.number_of_selfloops(G))
   print(&quot;Largest component:&quot;, run_measure(G, &quot;largest_connected_component&quot;))

   sim = Attack(G, attack=&quot;rnd_node&quot;, steps=min(5, len(G)), runs=1, seed=17)
   trajectory = sim.run_single_sim()</code></pre><p>The simulation example assumes a nonempty simple graph without self-loops. Inspect your input before selecting a process; do not silently drop components or change edge meaning to make a method run. Labels remain strings here, so the original identifiers are retained.</p>

.. _network-inputs-other-formats:

.. raw:: html

   <span id="other-formats"></span>

Read CSV, weighted edges, GraphML, or GML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Choose the reader that matches the file’s actual format. For a CSV with headers <code>source,target</code>, preserve identifier text explicitly:</p><pre><code>import pandas as pd
   import networkx as nx

   edges = pd.read_csv(&quot;edges.csv&quot;, dtype={&quot;source&quot;: str, &quot;target&quot;: str})
   G = nx.from_pandas_edgelist(edges, source=&quot;source&quot;, target=&quot;target&quot;,
                              create_using=nx.Graph())</code></pre><p>The following are alternative readers, not consecutive preprocessing steps. A weighted edge list contains three columns: source, target, and numeric weight. GraphML and GML can retain node and edge attributes, and may return directed or multigraph objects depending on the file.</p><pre><code>weighted = nx.read_weighted_edgelist(&quot;weighted.edgelist&quot;, nodetype=str)
   graphml = nx.read_graphml(&quot;network.graphml&quot;)
   gml = nx.read_gml(&quot;network.gml&quot;)</code></pre><p>If you already have a NetworkX graph—from another generator, a database, or your own code—use that object directly. Preserve a node-order list such as <code>node_order = list(G.nodes())</code> when interpreting saved state arrays. For spatial figures, store physical coordinates as node attributes, for example <code>G.nodes["A"]["pos"] = (x, y)</code>, and state their units.</p>

.. _network-inputs-representation:

.. raw:: html

   <span id="representation"></span>

Check representation before analysis
------------------------------------

.. raw:: html

   <p>Directed-to-undirected conversion discards direction; collapsing parallel edges discards multiplicity and needs an explicit rule for combining weights. Retaining only the largest component discards other nodes. Make these choices because they fit the study, not because the loader happens to do them. Disconnected graphs are valid inputs for some measures, but not for all path-based quantities.</p>

.. _network-inputs-section-1:

.. raw:: html

   <span id="section-1"></span>

Which operations read weights?
------------------------------

.. raw:: html

   <div class="table-scroll"><table><thead><tr><th>Operation</th><th>TIGER 0.6.0 behavior</th></tr></thead><tbody><tr><td>Spectral measures and NetShield</td><td>Read the NetworkX weight attribute; missing weights mean 1.</td></tr><tr><td>Path measures, ID/IB/RD/RB attacks</td><td>Use unweighted degree or hop-based paths. Edge-degree attack score is d(u)d(v).</td></tr><tr><td>SIS and SIR</td><td>Use one transmission probability b per infected–susceptible edge; ignore edge weights.</td></tr><tr><td>Motter–Lai</td><td>Unweighted shortest-path betweenness; input edge weights are not traffic distances.</td></tr><tr><td>Local load sharing</td><td>Defaults use intact unweighted degree for initial load and recipient weights; supplied loads, capacities, failures, and headroom-based allocation policies are also supported.</td></tr><tr><td>Crucitti</td><td>Initial incident-edge efficiency is 1; simulated efficiencies determine routing distances.</td></tr></tbody></table></div><p>A spectral diagnostic computed from weighted contacts need not describe an unweighted epidemic on those same links. For an unweighted study, set every weight to 1 before both selection and simulation. Some karate-club examples retain NetworkX’s supplied weights; the preventive-defense and SIR-ensemble studies explicitly use unit weights.</p>
    <p><a href="api-graphs.html">Graph loader API and source</a> · <a href="measures.html">Robustness measures</a></p>
