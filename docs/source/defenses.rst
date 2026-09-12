Defense measures
================

.. raw:: html

   <div class="tag">Guide</div><p class="lead">Hold the attack and outcome fixed when comparing node hardening, edge addition, or rewiring. Match both action count and meaningful cost whenever possible.</p>
   <div class="table-scroll"><table><thead><tr><th>Intervention</th><th>Question</th><th>Required baseline</th></tr></thead><tbody><tr><td>Protect nodes</td><td>Which components receive hardening or immunity?</td><td>Random protection with the same count.</td></tr><tr><td>Add edges</td><td>Where can feasible new routes create alternatives?</td><td>Random feasible additions with matched cost.</td></tr><tr><td>Rewire edges</td><td>Can placement improve with fixed edge count?</td><td>Random rewiring under the same feasibility rules.</td></tr></tbody></table></div>

.. _defenses-policies:

.. raw:: html

   <span id="policies"></span>

Edge policies
-------------

.. raw:: html

   <p>Random addition samples nonedges. Preferential addition joins low-degree endpoints. Random-edge rewiring replaces a sampled edge; random-neighbor rewiring samples uniformly from oriented vertex–neighbor pairs for deletion (not a uniform vertex followed by a uniform neighbor). Preferential rewiring detaches a neighbor of a highest-degree node, while preferential-random rewiring detaches the higher-degree endpoint of a sampled edge. These are topological rules; the implementation does not enforce geographic cost or construction constraints.</p><p>These policy families follow the edge-modification literature; operational feasibility remains study-specific. <a href="references.html#ref-beygelzimer2005improving">Reference</a> <a href="references.html#ref-tong2012gelling">Reference</a></p>

.. _defenses-ky2:

.. raw:: html

   <span id="ky2"></span>

Reconnect KY-2 after a fixed attack
-----------------------------------

.. raw:: html

   <div class="study-introduction" data-study="edge-defense-results"><p>First remove 30 KY-2 junctions using one recalculated-betweenness attack order (seed 17). Every repair trial starts from that same damaged graph. The experiment asks whether 30 edge additions or replacements can reconnect the surviving junctions, without restoring the removed ones.</p><p>The code separates the common attack from the defense loop, runs ten seeds per repair policy, and divides each largest-component count by the original 809 nodes. Additions increase edge count; rewiring replaces one edge at a time. The dashed no-repair baseline makes the gain over the damaged state visible.</p></div><details class="study-code" data-for-figure="edge-defense-results"><summary>Run the post-attack repair comparison — code</summary><pre><code>import numpy as np
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import graph_loader
   from graph_tiger.attacks import run_attack_method
   from graph_tiger.defenses import Defense

   G = graph_loader("ky2")
   removed = run_attack_method(G, "rb_node", k=30, approx=80, seed=17)
   damaged = G.copy()
   damaged.remove_nodes_from(removed)
   policies = ["rewire_edge_random", "rewire_edge_random_neighbor",
               "rewire_edge_preferential_random", "add_edge_random",
               "add_edge_preferential"]
   for policy in policies:
       trials = []
       for seed in range(10):
           sim = Defense(damaged, attack=None, defense=policy, k_d=30,
                         runs=1, steps=30, seed=seed,
                         robust_measure="largest_connected_component")
           trials.append(np.asarray(sim.run_single_sim()) / len(G))
       values = np.asarray(trials)
       x = np.arange(values.shape[1])
       plt.plot(x, values.mean(axis=0), label=policy)
       plt.fill_between(x, *np.quantile(values, [0.1, 0.9], axis=0), alpha=0.12)
   plt.axhline(values[0, 0], color="black", linestyle="--", label="No repair")
   plt.xlabel("Edge interventions")
   plt.ylabel("LCC / original nodes")
   plt.ylim(0, 1)
   plt.legend(fontsize=8)
   plt.tight_layout()
   plt.show()</code></pre></details><figure class="study-figure" id="edge-defense-results"><a href="guide-results/defenses-edge.svg" target="_blank"><img src="guide-results/defenses-edge.svg" alt="Five edge repair policies compared against the unchanged damaged KY-2 network" loading="lazy"></a><figcaption>Every policy starts from the same KY-2 graph after 30 RB node removals (attack seed 17). Ten defense seeds, 0–9; mean and 10th–90th percentile range. The dashed line is no repair. One addition increases edge count; one rewiring replaces an edge. These are action budgets, not equal construction costs. <a href="guide-results/defenses-edge.csv">Run data (CSV)</a> · <a href="guide-results/defenses-edge.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">An upward jump means a new connection has joined previously separate components. A flat segment may still change paths within a component, but that change is invisible to the selected connectivity measure. The plot evaluates topological reconnection, not geographic feasibility or engineering cost.</p><p>The common damaged baseline retains 26.7% of original nodes. Low-degree addition reaches 96.3% after 30 repairs. Edge repair cannot restore the 30 removed nodes; geographic feasibility is not enforced.</p><p>Edge count and engineering cost are different budgets. Added edges increase available infrastructure; rewiring keeps edge count fixed. Report both achieved connectivity and intervention cost before recommending a policy.</p>

.. _defenses-section-1:

.. raw:: html

   <span id="section-1"></span>

Node protection
---------------

.. raw:: html

   <p>In an Attack simulation, protected nodes survive selected removals. In a cascade, node defense doubles selected capacities; it does not immunize an initially attacked node. In epidemic simulations, node selection can confer immunity. Use the interpretation corresponding to the process.</p><div class="study-introduction" data-study="node-defense-results"><p>On the 34-node karate graph, compare no protection with protection of three randomly selected or NetShield-selected nodes. Each run then makes eight recalculated-degree attack attempts. Protection here means a selected node survives an attempted removal; it is not extra cascade capacity or epidemic recovery.</p><p>The code uses an Attack simulation because the intervention prevents node deletion during an attack. TIGER computes the order before applying protection, so a protected target is skipped rather than replaced with another target. Twenty seeds provide the run distribution; component sizes are normalized by the original 34 nodes.</p></div><details class="study-code" data-for-figure="node-defense-results"><summary>Run the node-protection comparison — code</summary><pre><code>import numpy as np
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import karate
   from graph_tiger.attacks import Attack

   G = karate()
   for policy in [None, "rnd_node", "ns_node"]:
       trials = []
       for seed in range(20):
           sim = Attack(G, attack="rd_node", steps=8, defense=policy,
                        k_d=0 if policy is None else 3, runs=1, seed=seed)
           trials.append(np.asarray(sim.run_single_sim()) / len(G))
       plt.plot(np.mean(trials, axis=0), label=policy or "No protection")
   plt.xlabel("Attack attempts")
   plt.ylabel("LCC / original nodes")
   plt.legend()
   plt.show()</code></pre></details><figure class="study-figure" id="node-defense-results"><a href="guide-results/defenses-node.svg" target="_blank"><img src="guide-results/defenses-node.svg" alt="Retained connectivity under eight attack attempts with no, random, or NetShield node protection" loading="lazy"></a><figcaption>Karate club; three protected nodes; seeds 0–19. A protected target survives an attempt, so the horizontal axis counts attempts, not successful removals. TIGER generates the RD attack order before applying protection; it does not retarget the defended graph. <a href="guide-results/defenses-node.csv">Run data (CSV)</a> · <a href="guide-results/defenses-node.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">A defended curve can retain more nodes both by avoiding fragmentation and by blocking removals. Its horizontal axis therefore counts attempts, not successful deletions. That distinction is essential when comparing this example with an attacker allowed to retarget protected nodes.</p><p>After eight attempts, mean retained connectivity is 14.7% without protection, 27.9% with random protection, and 85.3% with NetShield. These karate-club runs use the supplied weights for NetShield selection; the attack uses unweighted structure.</p>

.. _defenses-preventive-defense:

.. raw:: html

   <span id="preventive-defense"></span>

Improve topology before damage
------------------------------

.. raw:: html

   <div class="study-introduction" data-study="preventive-results"><p>This experiment changes the karate graph before damage occurs. Each policy receives five additions or rewiring actions, then faces the same eight-node recalculated-betweenness order computed on the intact graph. Unit weights keep the comparison focused on topology.</p><p>The code copies the original graph for every trial, applies additions or sequential removal/addition pairs, and only then deletes the attacked nodes. Sequential rewiring preserves the intended edge budget even when a later edit touches an edge created earlier. The attacker does not observe the intervention or recompute its order.</p></div><details class="study-code" data-for-figure="preventive-results"><summary>Run the preventive topology experiment — code</summary><pre><code>import numpy as np
   import networkx as nx
   from graph_tiger.attacks import run_attack_method
   from graph_tiger.defenses import run_defense_method
   from graph_tiger.measures import run_measure

   G = nx.karate_club_graph()
   nx.set_edge_attributes(G, 1.0, &quot;weight&quot;)
   order = run_attack_method(G, &quot;rb_node&quot;, k=8, seed=17)
   policies = [None, &quot;add_edge_random&quot;, &quot;add_edge_preferential&quot;, &quot;rewire_edge_random&quot;]
   results = {}
   for policy in policies:
       trials = []
       for seed in range(20):
           H = G.copy()
           if policy:
               edits = run_defense_method(H, policy, k=5, seed=seed)
               if policy.startswith(&quot;rewire_&quot;):
                   for removed, added in zip(edits[&quot;removed&quot;], edits[&quot;added&quot;]):
                       H.remove_edge(*removed)
                       H.add_edge(*added)
               else:
                   H.add_edges_from(edits[&quot;added&quot;])
           curve = [run_measure(H, &quot;largest_connected_component&quot;) / len(G)]
           for node in order:
               H.remove_node(node)
               curve.append(run_measure(H, &quot;largest_connected_component&quot;) / len(G))
           trials.append(curve)
       results[policy or &quot;none&quot;] = trials</code></pre></details><figure class="study-figure" id="preventive-results"><a href="guide-results/preventive-defense.svg" target="_blank"><img src="guide-results/preventive-defense.svg" alt="Connectivity under a fixed node attack after preventive edge interventions" loading="lazy"></a><figcaption>Twenty seeds per policy; lines are means and bands show the 10th–90th percentiles. No defense is the baseline. <a href="guide-results/preventive-defense.csv">Data (CSV)</a> · <a href="guide-results/preventive-defense.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure><p class="study-interpretation">Compare each curve with the no-defense baseline at the same number of removals. A gain means that these pre-positioned connections help against this specified attack order; it does not establish protection against an adaptive attacker. The before-and-after network views below show one of these five-edge interventions directly.</p><p>The complete runner plots the stored <code>results</code>. A realistic design study should also filter infeasible links and evaluate several plausible attack realizations.</p>

.. _defenses-defense-snapshots:

.. raw:: html

   <span id="defense-snapshots"></span>

See the defense itself
----------------------

.. raw:: html

   <div class="study-introduction" data-study="defense-network-results"><p>The preventive topology experiment uses the 34-node karate-club social graph. We use unit edge weights and add five connections between low-degree endpoints. The before-and-after views isolate the intervention itself: no attack has occurred yet.</p><p>The code computes one spring layout on the original graph, saves the returned additions, and draws the modified graph at the same positions. Orange dashed lines are new connections; gray lines already existed. The spring layout arranges nodes for readability, so line length is not a construction distance. The preventive experiment above tests whether these changes retain connectivity under its fixed attack.</p></div><details class="study-code" data-for-figure="defense-network-results"><summary>Draw the added connections — code</summary><pre><code>import networkx as nx
   import matplotlib.pyplot as plt
   from matplotlib.lines import Line2D
   from graph_tiger.defenses import run_defense_method

   G = nx.karate_club_graph()
   nx.set_edge_attributes(G, 1.0, "weight")
   pos = nx.spring_layout(G, seed=17, weight=None)
   added = run_defense_method(G, "add_edge_preferential", k=5, seed=17)["added"]
   H = G.copy()
   H.add_edges_from(added)
   for graph, title in [(G, "Before defense"), (H, "Five edges added")]:
       fig, ax = plt.subplots(figsize=(8, 6.5))
       nx.draw(graph, pos, ax=ax, node_color="#157a79", node_size=140,
               edge_color="#b6c3c8", width=1.2, with_labels=False)
       if graph is H:
           nx.draw_networkx_edges(H, pos, ax=ax, edgelist=added,
                                  edge_color="#cf6636", style="dashed", width=3)
           ax.legend(handles=[Line2D([], [], color="#cf6636", linestyle="--",
                                     label="Added edge")])
       ax.set_title(title)
       ax.set_aspect("equal")
       plt.show()</code></pre></details><figure class="study-figure" id="defense-network-results"><a href="guide-results/defense-before.svg" target="_blank"><img src="guide-results/defense-before.svg" alt="Karate graph before preventive edge addition" loading="lazy"></a><a href="guide-results/defense-after.svg" target="_blank"><img src="guide-results/defense-after.svg" alt="Karate graph with five added edges highlighted" loading="lazy"></a><figcaption>The intervention is the one used in the preventive-defense guide. <a href="guide-results/preventive-defense.csv">Data (CSV)</a> · <a href="guide-results/preventive-defense.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure><p class="study-interpretation">The picture shows which nodes receive alternatives, not whether those alternatives are effective under every threat. An added connection helps an attack outcome only if its endpoints survive and it connects useful parts of the remaining network.</p><p><a href="visualization.html#plotting-options">Shared plotting and export options</a>.</p>
