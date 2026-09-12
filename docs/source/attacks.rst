Attack types
============

.. raw:: html

   <div class="tag">Guide</div><p class="lead">State the threatened component, removal budget, attacker information, whether scores are recalculated, and the service outcome. Compare targeted policies with random removal under the same budget.</p>
   <div class="grid"><div class="card"><h3 id="section-1">Node removal</h3><p><span class="pill">id_node</span><span class="pill">rd_node</span><span class="pill">ib_node</span><span class="pill">rb_node</span><span class="pill">rnd_node</span></p></div><div class="card"><h3 id="section-2">Edge removal</h3><p><span class="pill">id_edge</span><span class="pill">rd_edge</span><span class="pill">ib_edge</span><span class="pill">rb_edge</span><span class="pill">rnd_edge</span></p></div><div class="card"><h3 id="section-3">Report</h3><p>Topology, attack order, retained outcome curve, random-reference uncertainty, and exact budget.</p></div></div>
   <pre><span class="kw">from</span> graph_tiger.attacks <span class="kw">import</span> Attack
   <span class="kw">from</span> graph_tiger.graphs <span class="kw">import</span> karate

   G = karate()

   targeted = Attack(G, runs=1, steps=8, attack=<span class="st">"id_node"</span>,
                     robust_measure=<span class="st">"largest_connected_component"</span>, seed=17)
   random = Attack(G, runs=50, steps=8, attack=<span class="st">"rnd_node"</span>,
                   robust_measure=<span class="st">"largest_connected_component"</span>, seed=17)
   targeted_curve = targeted.run_simulation()
   random_curve = random.run_simulation()</pre>

.. _attacks-policies:

.. raw:: html

   <span id="policies"></span>

Initial and recalculated policies
---------------------------------

.. raw:: html

   <p>Initial degree (ID) ranks nodes once by incident-edge count. Initial betweenness (IB) ranks them by shortest-path brokerage. Recalculated degree (RD) and betweenness (RB) update their scores after every removal. Recalculation can adapt to changed structure, at added computational cost; it does not guarantee a stronger attack on every graph.</p>

.. _attacks-road-snapshots:

.. raw:: html

   <span id="road-snapshots"></span>

Follow damage on a water network
--------------------------------

.. raw:: html

   <div class="study-introduction" data-study="network-results"><p>The KY-2 water-distribution graph contains 809 junctions connected by pipe links. We remove 0, 10, and 30 junctions using recalculated degree: after each removal, the next target is chosen from the remaining graph. These snapshots ask where damage occurs and which surviving junctions can still reach one another; they do not model hydraulic flow, pressure, or delivered water demand.</p><p>The code reads the dataset’s node positions once, computes the attack order once, and removes progressively longer prefixes from fresh graph copies. It then identifies the largest connected component in each copy. Teal marks that component, amber marks other surviving components, and red crosses mark removed junctions. Pale original links provide context; darker links belong to the surviving graph. Keeping positions fixed lets you follow the same junction across all three views.</p></div><details class="study-code" data-for-figure="network-results"><summary>Draw the water-network snapshots — code</summary><pre><code>import networkx as nx
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import graph_loader
   from graph_tiger.attacks import run_attack_method

   G = graph_loader("ky2")
   pos = nx.get_node_attributes(G, "pos")
   removed = run_attack_method(G, "rd_node", k=30, seed=17)
   for k in [0, 10, 30]:
       H = G.copy()
       H.remove_nodes_from(removed[:k])
       largest = max(nx.connected_components(H), key=len)
       plt.figure(figsize=(8, 8))
       nx.draw_networkx_edges(G, pos, edge_color="#e4e8eb", width=0.6)
       nx.draw_networkx_edges(H, pos, edge_color="#8b9aa2", width=0.9)
       nx.draw_networkx_nodes(H, pos, node_size=15,
           node_color=["#157a79" if n in largest else "#e4a24b" for n in H])
       nx.draw_networkx_nodes(G, pos, nodelist=removed[:k], node_shape="x",
                             node_color="#b83e58", node_size=38)
       plt.title(f"{k} nodes removed; largest component {len(largest)}/{len(G)}")
       plt.axis("equal")
       plt.axis("off")
       plt.savefig(f"ky2-{k}-removed.png", dpi=180)
   plt.show()</code></pre></details><figure class="study-figure" id="network-results"><a href="guide-results/network-0.svg" target="_blank"><img src="guide-results/network-0.svg" alt="Intact KY-2 network in dataset coordinates" loading="lazy"></a><a href="guide-results/network-10.svg" target="_blank"><img src="guide-results/network-10.svg" alt="KY-2 after ten recalculated-degree removals" loading="lazy"></a><a href="guide-results/network-30.svg" target="_blank"><img src="guide-results/network-30.svg" alt="KY-2 after thirty recalculated-degree removals" loading="lazy"></a><figcaption>The same network coordinates are used in all three states. Teal denotes the largest surviving component, amber other components, and red crosses removed nodes. Faint edges retain the original network as context. These snapshots use RD, not the RB attack used in the repair study. <a href="guide-results/visualization.csv">Run data (CSV)</a> · <a href="guide-results/visualization.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">The largest component contains 809, 797, 772 junctions after 0, 10, 30 removals, respectively. Read those counts together with the maps, not just the amount of colored ink. A removed junction can separate nearby branches even when most of the drawing still looks intact. These maps use recalculated degree, whereas the repair experiment uses recalculated betweenness; they illustrate different attack scenarios.</p><p><a href="visualization.html#plotting-options">Shared plotting and export options</a>.</p>

.. _attacks-ky2:

.. raw:: html

   <span id="ky2"></span>

Compare node and edge attacks on KY-2
-------------------------------------

.. raw:: html

   <div class="study-introduction" data-study="attack-results"><p>Both panels start from the same 809-node KY-2 water-distribution graph. Each trial removes up to 30 nodes or 30 edges using random, initial-degree, recalculated-degree, initial-betweenness, or recalculated-betweenness selection. The outcome is the largest surviving component divided by 809: the share of original junctions still connected within one component.</p><p>The code loops over policies and ten seeds, creates a fresh simulation for each trial, and normalizes its component-size history before computing the mean and 10th–90th percentiles. Betweenness is estimated using 80 sampled sources. Read node attacks and edge attacks separately: equal removal counts do not represent equal damage or cost.</p></div><details class="study-code" data-for-figure="attack-results"><summary>Run the attack comparison — code</summary><pre><code>import numpy as np
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import graph_loader
   from graph_tiger.attacks import Attack

   G = graph_loader("ky2")
   methods = {"node": ["rnd_node", "id_node", "rd_node", "ib_node", "rb_node"],
              "edge": ["rnd_edge", "id_edge", "rd_edge", "ib_edge", "rb_edge"]}
   fig, axes = plt.subplots(1, 2, figsize=(12, 4))
   for ax, (kind, names) in zip(axes, methods.items()):
       for name in names:
           trials = []
           for seed in range(10):
               sim = Attack(G, attack=name, steps=30, runs=1, seed=seed,
                            attack_approx=min(80, len(G)),
                            robust_measure="largest_connected_component")
               trials.append(np.asarray(sim.run_simulation()) / len(G))
           values = np.asarray(trials)
           x = np.arange(values.shape[1])
           ax.plot(x, values.mean(axis=0), label=name)
           ax.fill_between(x, *np.quantile(values, [0.1, 0.9], axis=0), alpha=0.12)
       ax.set(xlabel=f"{kind.title()}s removed", ylabel="LCC / original nodes", ylim=(0, 1))
       ax.legend()
   fig.tight_layout()
   fig.savefig("ky2-attacks.png", dpi=180)</code></pre></details><figure class="study-figure" id="attack-results"><a href="guide-results/attacks-node.svg" target="_blank"><img src="guide-results/attacks-node.svg" alt="Node removal strategies and retained connectivity on KY-2" loading="lazy"></a><a href="guide-results/attacks-edge.svg" target="_blank"><img src="guide-results/attacks-edge.svg" alt="Edge removal strategies and retained connectivity on KY-2" loading="lazy"></a><figcaption>KY-2, 30 removals, seeds 0–9. Lines are means; shading is the 10th–90th percentile across runs, not a confidence interval. Betweenness uses 80 sampled sources. Compare strategies within a panel: one node removal is not the same intervention as one edge removal. <a href="guide-results/attacks.csv">Run data (CSV)</a> · <a href="guide-results/attacks.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">A steep drop means a removal separates a large part of the network, rather than merely deleting one more component. Comparing initial and recalculated policies tests whether updating the ranking as damage accumulates matters for this graph.</p><p>Recalculated betweenness ends at 31.4% mean retained connectivity for node attacks and 55.6% for edge attacks. These are separate removal budgets, not equal-cost damage.</p><p>The shaded region is the 10th–90th percentile range across runs, not a confidence interval. Exact deterministic policies may produce identical runs; approximation and random removal introduce variation.</p><p><a href="visualization.html">Draw the network and save attack snapshots →</a></p>

.. _attacks-section-4:

.. raw:: html

   <span id="section-4"></span>

Other selection policies
------------------------

.. raw:: html

   <p>For edge ID and RD, TIGER scores an edge {u,v} by <code>degree(u) * degree(v)</code>. RD updates this score after each planned deletion. IB/RB rank shortest-path betweenness, with <code>approx</code> counting sampled source nodes even for edge attacks. <a href="references.html#ref-holme2002attack">Reference</a></p><div class="table-scroll"><table><thead><tr><th>Family</th><th>Selection idea</th><th>What it does not establish</th></tr></thead><tbody><tr><td>NetShield (ns_node)</td><td>Greedily selects a set using a leading-eigenpair approximation to spectral-radius reduction. <a href="references.html#ref-tong2010vulnerability">Reference</a></td><td>Not an optimal LCC attack or defense for every process.</td></tr><tr><td>PageRank / eigenvector</td><td>Ranks recursively central nodes; emphasizes well-connected neighborhoods. <a href="references.html#ref-page1999pagerank">Reference</a></td><td>Not a direct measure of geographic exposure or operational demand.</td></tr><tr><td>Line-graph edge policies</td><td>Treats each original edge as a node and ranks the resulting line graph.</td><td>Can be expensive around hubs; differs from endpoint-degree-product scoring.</td></tr></tbody></table></div><p><a href="api-attacks.html">All supported keys and selection functions</a>. In Attack, recalculated attack orders are generated on a working copy before node protection is applied. For an attacker that retargets a defended network, write that decision loop explicitly and state the changed threat model.</p>
