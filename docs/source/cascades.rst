Cascading failures
==================

.. raw:: html

   <div class="tag">Guide</div><p class="lead">A cascade begins when an initial failure changes the work carried by the remaining network. Choose the model by how that work moves and what happens when capacity is exceeded. The examples below use an illustrative load-sharing chain and an empirical power-network topology—not a social network. These are simplified load and routing models, not electrical power-flow simulations.</p>
   <div class="table-scroll"><table><thead><tr><th>Model</th><th>Load</th><th>Overload</th><th>Outcome</th></tr></thead><tbody>
   <tr><td><a href="references.html#ref-motter">Motter–Lai (2002)</a></td><td>Global shortest-path betweenness</td><td>Node fails</td><td>Retained connectivity</td></tr>
   <tr><td><a href="references.html#ref-crucitti">Crucitti–Latora–Marchiori (2004)</a></td><td>Most-efficient weighted paths</td><td>Incident edge efficiency degrades</td><td>Weighted network efficiency</td></tr>
   <tr><td><a href="references.html#ref-wei">Local load sharing: Wei et al. (2012)</a></td><td>Failed-node load moves to functioning neighbors</td><td>Over-capacity recipient fails next round</td><td>Connectivity, failed nodes, and lost service</td></tr></tbody></table></div>
   <div class="call"><strong>Beta intuition.</strong> A functioning neighbor receives weight proportional to <code>degree**beta</code>. At <code>beta=0</code>, neighbors receive equal shares. Increasing beta directs more load toward high-degree neighbors, which may absorb load or become concentrated overload points.</div>

.. _cascades-local-rule:

.. raw:: html

   <span id="local-rule"></span>

Local redistribution
--------------------

.. raw:: html

   <p>By default, L<sub>i</sub>(0) = k<sub>i</sub>, where k<sub>i</sub> is the intact degree, and capacity C<sub>i</sub> = (1+r)L<sub>i</sub>(0). When i fails, its current load is shared among functioning neighbors N<sub>i</sub><sup>+</sup>:</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Local load redistribution"><mrow><msub><mi>ΔL</mi><mrow><mi>i</mi><mo>→</mo><mi>j</mi></mrow></msub><mo>=</mo><msub><mi>L</mi><mi>i</mi></msub><mfrac><msup><msub><mi>k</mi><mi>j</mi></msub><mi>β</mi></msup><mrow><munder><mo>∑</mo><mrow><mi>u</mi><mo>∈</mo><msup><msub><mi>N</mi><mi>i</mi></msub><mo>+</mo></msup></mrow></munder><msup><msub><mi>k</mi><mi>u</mi></msub><mi>β</mi></msup></mrow></mfrac></mrow></math></div><p>Degrees in the weights come from the intact graph. Transfers from the same round are accumulated before checking new overloads. If two recipients have degrees 1 and 3 and the failed load is 8, β=0 gives shares 4 and 4; β=1 gives 2 and 6; β=2 gives 0.8 and 7.2.</p>

.. _cascades-local-allocation-policies:

.. raw:: html

   <span id="local-allocation-policies"></span>

Choose a local allocation policy
--------------------------------

.. raw:: html

   <p>All four policies move the full workload of each newly failed node to functioning neighbors. They differ in which neighbors receive it, not whether excess work can be discarded. Overloads are checked only after all transfers in the round have been accumulated. Work becomes <code>lost_load</code> only when its source has no eligible recipient.</p><div class="table-scroll"><table><thead><tr><th>allocation</th><th>Where displaced work goes</th></tr></thead><tbody><tr><td><code>degree</code></td><td>Default. Intact-degree weights raised to beta; beta=0 gives equal sharing.</td></tr><tr><td><code>greedy</code></td><td>Fill the largest spare capacities first, independently for each failed source; split any remainder equally among its recipients.</td></tr><tr><td><code>proportional</code></td><td>Divide the full workload in proportion to spare capacity; use equal shares when all spare capacities are zero.</td></tr><tr><td><code>max_flow</code></td><td>Coordinate all failed sources to maximize the work placed within shared spare capacities, then split each source’s remainder equally among its recipients.</td></tr></tbody></table></div>

.. _cascades-section-1:

.. raw:: html

   <span id="section-1"></span>

What spare capacity means
~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>For a recipient j, C<sub>j</sub> is its fixed capacity and L<sub>j</sub> its load before this round. Its headroom is s<sub>j</sub>=max(0, C<sub>j</sub>−L<sub>j</sub>). For one failed source, d is its displaced load, q its number of eligible neighbors, and S the sum of their headrooms.</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>s</mi><mi>j</mi></msub><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><msub><mi>C</mi><mi>j</mi></msub><mo>−</mo><msub><mi>L</mi><mi>j</mi></msub><mo>)</mo><mo>,</mo><mspace width="1em"/><mi>S</mi><mo>=</mo><munder><mo>∑</mo><mi>j</mi></munder><msub><mi>s</mi><mi>j</mi></msub></math></div><p>Proportional allocation assigns x<sub>j</sub>=d s<sub>j</sub>/S when S&gt;0, and d/q otherwise. Greedy allocation visits neighbors in decreasing headroom, assigns the smaller of remaining work and headroom, then adds an equal share of any remainder to every eligible neighbor. Each source uses the same pre-round loads; greedy does not reserve capacity for other sources. Ties follow the original graph’s node insertion order.</p>

.. _cascades-section-2:

.. raw:: html

   <span id="section-2"></span>

Coordinating a whole round with maximum flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>The auxiliary flow network has source-to-failed-node arcs of capacity d<sub>i</sub>, failed-node-to-eligible-recipient arcs of capacity d<sub>i</sub>, and one recipient-to-sink arc of capacity s<sub>j</sub> per recipient. That single shared sink arc prevents different failures from spending the same headroom twice. TIGER uses NetworkX’s Edmonds–Karp solver with nodes and transfer arcs inserted in original graph node order.</p><p>If f<sub>ij</sub> is the first-pass flow, the remaining work from source i is u<sub>i</sub>=d<sub>i</sub>−∑<sub>j</sub>f<sub>ij</sub>. Its complete transfer is f<sub>ij</sub>+u<sub>i</sub>/q<sub>i</sub>. Overflow is still assigned even when it causes failure. This maximizes the capacity-fitting first pass, not long-term survival; tied maximum flows can lead to different later cascades. Record graph insertion order and NetworkX version when reproducing ties.</p>

.. _cascades-section-3:

.. raw:: html

   <span id="section-3"></span>

Supply application loads and capacities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p><code>initial_load</code> and <code>capacities</code> are complete node-to-value dictionaries of finite nonnegative numbers. Omit loads to use intact unweighted degrees; omit capacities to use (1+r) times the supplied or default initial loads. Explicit capacities are used directly, without multiplying by (1+r) again. Inputs are copied and restored on reset. Existing node-defense options can still increase those capacities.</p><p><code>initial_failures</code> supplies an exact initiating set instead of attack selection; an empty list means no initiating failures. This option and the supplied load/capacity mappings apply to <code>local_load_sharing</code> only. The field <code>last_transfers[(i, j)]</code> reports the completed transfers of the latest redistribution round. <code>lost_load</code> is cumulative unserved work; <code>shed_load</code> remains a backward-compatible alias, including in older CSVs and figures. Neither is a deliberate shedding control.</p>

.. _cascades-allocation-single-example:

.. raw:: html

   <span id="allocation-single-example"></span>

One failed source: headroom and overflow
----------------------------------------

.. raw:: html

   <p>A failed node F hands work to recipients a, b, and c, initially carrying zero work and having capacities 3, 5, and 7. Run the same four policies with 12 and then 18 units displaced. Equal sharing uses beta=0. The dashed marks show available headroom; a bar above its mark means the recipient overloads in this round.</p><details class="study-code"><summary>Run the single-source comparison — code</summary><pre><code>import networkx as nx
   from graph_tiger.cascading import Cascading

   G = nx.Graph([(&quot;F&quot;, &quot;a&quot;), (&quot;F&quot;, &quot;b&quot;), (&quot;F&quot;, &quot;c&quot;)])
   policies = [&quot;degree&quot;, &quot;greedy&quot;, &quot;proportional&quot;, &quot;max_flow&quot;]
   allocations = {}
   for demand in [12, 18]:
       for policy in policies:
           sim = Cascading(G.copy(), model=&quot;local_load_sharing&quot;, allocation=policy,
                           initial_load={&quot;F&quot;: demand, &quot;a&quot;: 0, &quot;b&quot;: 0, &quot;c&quot;: 0},
                           capacities={&quot;F&quot;: demand, &quot;a&quot;: 3, &quot;b&quot;: 5, &quot;c&quot;: 7},
                           initial_failures=[&quot;F&quot;], beta=0, runs=1, steps=1)
           sim.run_single_sim()
           allocations[demand, policy] = [sim.last_transfers[&quot;F&quot;, n] for n in &quot;abc&quot;]</code></pre></details><div class="vis-grid"><figure><a href="guide-results/local-allocation-single-12.svg"><img src="guide-results/local-allocation-single-12.svg" alt="12 units: enough total headroom, but equal sharing overloads a" loading="lazy"></a><figcaption>12 units: enough total headroom, but equal sharing overloads a · <a href="guide-results/local-allocation-single-12.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/local-allocation-single-18.svg"><img src="guide-results/local-allocation-single-18.svg" alt="18 units: all policies assign the full load, despite insufficient headroom" loading="lazy"></a><figcaption>18 units: all policies assign the full load, despite insufficient headroom · <a href="guide-results/local-allocation-single-18.pdf">PDF</a></figcaption></figure></div><p>With 12 units, greedy gives (0,5,7) and proportional gives (2.4,4,5.6); both fit. Equal sharing gives (4,4,4), overloading a despite sufficient total headroom. The fixed-order maximum-flow solution gives (3,5,4); it is another feasible allocation, not the uniquely best one. With 18 units, greedy and maximum flow give (4,6,8), while proportional gives (3.6,6,8.4). All recipients overload for those three policies. Equal sharing gives (6,6,6), initially overloading only a and b—showing that maximizing the first-pass fit does not necessarily minimize failures.</p>

.. _cascades-allocation-shared-example:

.. raw:: html

   <span id="allocation-shared-example"></span>

Two failed sources: why coordination matters
--------------------------------------------

.. raw:: html

   <p>A has 5 units and can use U or V. B has 7 units and can use only U. The recipients start empty, with capacities 7 and 5. Arrows show permitted handoffs; their labels are the work actually transferred in one round. Gray nodes failed initially, orange indicates a new overload, and teal means functioning. All panels use the same positions, capacities, and initial failures.</p><details class="study-code"><summary>Run the shared-recipient comparison — code</summary><pre><code>import networkx as nx
   from graph_tiger.cascading import Cascading

   G = nx.Graph([(&quot;A&quot;, &quot;U&quot;), (&quot;A&quot;, &quot;V&quot;), (&quot;B&quot;, &quot;U&quot;)])
   policies = [&quot;degree&quot;, &quot;greedy&quot;, &quot;proportional&quot;, &quot;max_flow&quot;]
   trials = {}
   for policy in policies:
       sim = Cascading(G.copy(), model=&quot;local_load_sharing&quot;, allocation=policy,
                       initial_load={&quot;A&quot;: 5, &quot;B&quot;: 7, &quot;U&quot;: 0, &quot;V&quot;: 0},
                       capacities={&quot;A&quot;: 5, &quot;B&quot;: 7, &quot;U&quot;: 7, &quot;V&quot;: 5},
                       initial_failures=[&quot;A&quot;, &quot;B&quot;], beta=0, runs=1, steps=1)
       sim.run_single_sim()
       trials[policy] = {&quot;transfers&quot;: sim.last_transfers.copy(),
                         &quot;failed&quot;: sim.failed.copy(), &quot;load&quot;: sim.load.copy(),
                         &quot;lost_load&quot;: sim.sim_info[1][&quot;lost_load&quot;]}</code></pre></details><div class="vis-grid"><figure><a href="guide-results/local-allocation-shared-degree.svg"><img src="guide-results/local-allocation-shared-degree.svg" alt="Equal sharing" loading="lazy"></a><figcaption>Equal sharing · <a href="guide-results/local-allocation-shared-degree.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/local-allocation-shared-greedy.svg"><img src="guide-results/local-allocation-shared-greedy.svg" alt="Greedy headroom allocation" loading="lazy"></a><figcaption>Greedy headroom allocation · <a href="guide-results/local-allocation-shared-greedy.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/local-allocation-shared-proportional.svg"><img src="guide-results/local-allocation-shared-proportional.svg" alt="Proportional headroom allocation" loading="lazy"></a><figcaption>Proportional headroom allocation · <a href="guide-results/local-allocation-shared-proportional.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/local-allocation-shared-max_flow.svg"><img src="guide-results/local-allocation-shared-max_flow.svg" alt="Coordinated maximum flow" loading="lazy"></a><figcaption>Coordinated maximum flow · <a href="guide-results/local-allocation-shared-max_flow.pdf">PDF</a></figcaption></figure></div><p>Greedy sends A’s 5 units to U, then B also sends its 7 units to U: U receives 12 against capacity 7 and fails. Proportional and equal sharing also overload U here. Maximum flow instead sends A’s 5 units to V and B’s 7 units to U, so neither recipient fails. No policy loses or discards work during this round. If B instead carries 9 units, even maximum flow must send U two units beyond its headroom; U then fails.</p><p>Download the <a href="guide-results/run-local-allocation-study.py">complete runner</a>, <a href="guide-results/local-allocation.csv">transfer data</a>, and <a href="guide-results/local-allocation.json">settings and figure hashes</a>. These small examples isolate allocation mechanics; they do not establish which policy is best for every application.</p>

.. _cascades-cascade-snapshots:

.. raw:: html

   <span id="cascade-snapshots"></span>

Watch failures propagate
------------------------

.. raw:: html

   <div class="study-introduction" data-study="cascade-state-results"><p>This four-node chain isolates a single mechanism: a failed node hands its entire workload to its functioning neighbor. It is an invented example with arbitrary work units, not measured grid demand. F fails initially. A and B cannot absorb the incoming work; C has enough capacity to stop the cascade.</p><p>The initial loads are (6, 2, 2, 1) and capacities are (6, 5, 8, 12) for F, A, B, C. Each snapshot follows one redistribution round. Pink nodes stay failed; their remaining workload is transferred in the next round. Lines show the original connections, not functioning service.</p></div><details class="study-code" data-for-figure="cascade-state-results"><summary>Run the chain example — code</summary><pre><code>import numpy as np
   import networkx as nx
   from graph_tiger import graphs
   from graph_tiger.cascading import Cascading

   G = nx.path_graph([&quot;F&quot;, &quot;A&quot;, &quot;B&quot;, &quot;C&quot;])
   loads = dict(F=6, A=2, B=2, C=1)
   capacities = dict(F=6, A=5, B=8, C=12)
   sim = Cascading(G, model=&quot;local_load_sharing&quot;, allocation=&quot;degree&quot;, beta=0,
                   initial_load=loads, capacities=capacities,
                   initial_failures=[&quot;F&quot;], runs=1, steps=3)
   states = []
   for t in range(4):
       states.append(dict(round=t, failed=sorted(sim.failed),
                          load=sim.load.copy(), lost_load=sim.lost_load))
       if t &lt; 3:
           sim.run_local_load_sharing_step()</code></pre></details><figure id="cascade-state-results"><img loading="lazy" src="guide-results/chain-cascade-0.svg" alt="Four-node chain after round 0 with loads, capacities, and failed nodes" style="width:100%;height:auto"><img loading="lazy" src="guide-results/chain-cascade-1.svg" alt="Four-node chain after round 1 with loads, capacities, and failed nodes" style="width:100%;height:auto"><img loading="lazy" src="guide-results/chain-cascade-2.svg" alt="Four-node chain after round 2 with loads, capacities, and failed nodes" style="width:100%;height:auto"><img loading="lazy" src="guide-results/chain-cascade-3.svg" alt="Four-node chain after round 3 with loads, capacities, and failed nodes" style="width:100%;height:auto"><figcaption>Read downward: F sends 6 to A; A sends 8 to B; B sends 10 to C. C finishes with 11, below capacity 12. All 11 initial units are accounted for, and no workload is lost.</figcaption></figure><p class="study-interpretation">The failed-node counts are 1, 2, 3, and 3. A stable count alone is not a stopping rule: B still has work to transfer after round 2. The cascade is complete after round 3, when every failed workload has been processed and no new overload remains. <a href="guide-results/chain-cascade.json">Download the states.</a></p>

.. _cascades-beta-study:

.. raw:: html

   <span id="beta-study"></span>

Compare beta on a power network
-------------------------------

.. raw:: html

   <div class="study-introduction" data-study="local-cascade-results"><p>The electrical-network dataset contains 4,941 nodes and 6,594 edges. We treat it as a topology: initial load is degree, not measured power demand, and capacity is 1.2 times initial load. The highest-degree node fails first. Only the redistribution preference β changes between the three runs.</p><p>The code follows 20 redistribution rounds and reads cumulative failed nodes and lost load separately. β=0 shares equally among functioning neighbors; β=1 or 2 increasingly favors neighbors with higher intact degree. Lost load is work with no functioning recipient, not another count of failed nodes.</p></div><details class="study-code" data-for-figure="local-cascade-results"><summary>Run the local-sharing comparison — code</summary><pre><code>import matplotlib.pyplot as plt
   from graph_tiger.graphs import graph_loader
   from graph_tiger.cascading import Cascading

   G = graph_loader("electrical")
   fig, axes = plt.subplots(2, 1, figsize=(8, 8))
   for beta in [0, 1, 2]:
       sim = Cascading(G, model="local_load_sharing", beta=beta, r=0.2,
                       attack="id_node", k_a=1, runs=1, steps=20, seed=7)
       sim.run_single_sim()
       for ax, field in zip(axes, ["failed", "shed_load"]):
           ax.plot([sim.sim_info[t][field] for t in range(21)], label=f"beta={beta}")
           ax.set_ylabel(field)
           ax.set_xlabel("Redistribution rounds")
           ax.legend()
   fig.tight_layout()
   plt.show()</code></pre></details><figure class="study-figure" id="local-cascade-results"><a href="guide-results/cascades-local-failed.svg" target="_blank"><img src="guide-results/cascades-local-failed.svg" alt="Cumulative failed power-network nodes under equal and degree-weighted local sharing" loading="lazy"></a><a href="guide-results/cascades-local-shed.svg" target="_blank"><img src="guide-results/cascades-local-shed.svg" alt="Load that cannot be redistributed to functioning neighbors" loading="lazy"></a><figcaption>Electrical-network dataset; one initial high-degree failure; r=0.2, β=0,1,2, seed 7. Initial load equals intact degree, not measured electrical demand. Failed nodes and shed load are distinct outcomes. Shifting load toward hubs need not reduce either outcome. <a href="guide-results/cascades-local.csv">Run data (CSV)</a> · <a href="guide-results/cascades-local.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">A flatter failure curve over this horizon indicates slower damage growth, not necessarily a smaller completed cascade. The longer-run section below checks the final outcome; choosing β from the first 20 rounds alone can be misleading.</p><p>By round 20, β=0, 1 and 2 have caused 3,516, 3,714, 3,459 failures. These curves are still rising; see the completed-cascade comparison below.</p><p>Local-sharing overloads satisfy L<sub>i</sub>&gt;C<sub>i</sub>. Failed sources are processed once; a source with no functioning recipient contributes its current load to cumulative lost load. Fixed intact degrees are used even after damage. Continue until no unprocessed failure remains and no new overload appears. <a href="references.html#ref-wei2012analysis">Reference</a></p>

.. _cascades-section-4:

.. raw:: html

   <span id="section-4"></span>

State and timing
----------------

.. raw:: html

   <p>Index 0 records the attacked state; indices 1 through <code>steps</code> record transitions. <code>failed</code> counts cumulative failures, <code>status</code> stores node loads in original node order, and <code>lost_load</code> records work that could not be passed to a functioning neighbor. Crucitti also records <code>edge_efficiency</code> and <code>overloaded</code>.</p><p>Use <code>run_single_sim()</code> to inspect that realization’s history immediately. <code>run_simulation()</code> averages output and resets state after every run, including the last.</p>

.. _cascades-motter-lai-equations:

.. raw:: html

   <span id="motter-lai-equations"></span>

Motter–Lai: reroute, then remove overloads
------------------------------------------

.. raw:: html

   <p>Let F(t) be the cumulative failed-node set and G(t) the graph induced by functioning nodes. Set initial load to unnormalized node betweenness B<sub>i</sub>(G), excluding endpoints; split a shortest-path contribution among ties. Fix capacity before the attack:</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>L</mi><mi>i</mi></msub><mo>(0)=</mo><msub><mi>B</mi><mi>i</mi></msub><mo>(G),</mo><msub><mi>C</mi><mi>i</mi></msub><mo>=(1+r)</mo><msub><mi>L</mi><mi>i</mi></msub><mo>(0)</mo></math></div><p>For each functioning node, B<sub>i</sub>(G(t)) = ∑<sub>s&lt;u; s,u≠i</sub> σ<sub>su</sub>(i)/σ<sub>su</sub>, where σ<sub>su</sub> counts shortest paths and σ<sub>su</sub>(i) counts those through i. Count each unordered pair once; unreachable pairs contribute zero. After the initial attack, recompute all loads from the surviving topology. Every node whose new load exceeds its fixed capacity fails in the same round:</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>L</mi><mi>i</mi></msub><mo>(t)=</mo><msub><mi>B</mi><mi>i</mi></msub><mo>(G(t))</mo></math></div><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>F</mi><mo>(t+1)=F(t)∪{i∉F(t):</mo><msub><mi>L</mi><mi>i</mi></msub><mo>(t)&gt;</mo><msub><mi>C</mi><mi>i</mi></msub><mo>}</mo></math></div><p>Stop when a round adds no failures. Disconnected source–target pairs carry no path contribution. A node with zero initial betweenness has zero base capacity in this implementation. <a href="references.html#ref-motter2002cascade">Reference</a></p>

.. _cascades-crucitti-equations:

.. raw:: html

   <span id="crucitti-equations"></span>

Crucitti: reroute through changing edge efficiencies
----------------------------------------------------

.. raw:: html

   <p>Use the same initial betweenness loads and fixed capacities, but keep overloaded nodes functioning. Each initial edge has efficiency e<sub>ij</sub>(0)=1 and routing distance 1/e<sub>ij</sub>. At each round, calculate an endpoint factor:</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>a</mi><mi>i</mi></msub><mo>(t)=</mo><mrow><mo>{</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mtext>if </mtext><msub><mi>L</mi><mi>i</mi></msub><mo>(t)≤</mo><msub><mi>C</mi><mi>i</mi></msub></mtd></mtr><mtr><mtd><mfrac><msub><mi>C</mi><mi>i</mi></msub><mrow><msub><mi>L</mi><mi>i</mi></msub><mo>(t)</mo></mrow></mfrac></mtd><mtd><mtext>otherwise</mtext></mtd></mtr></mtable></mrow></math></div><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>e</mi><mi>ij</mi></msub><mo>(t+1)=min(</mo><msub><mi>a</mi><mi>i</mi></msub><mo>(t),</mo><msub><mi>a</mi><mi>j</mi></msub><mo>(t))</mo></math></div><p>Recompute betweenness on shortest routes under the new reciprocal-efficiency distances. This is an assignment from the current capacity-to-load factors, not multiplication by last round’s efficiency. An edge can therefore recover as overload recedes. If both endpoints overload, the lower factor controls their undirected edge.</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>E</mi><mo>(t)=</mo><mfrac><mn>1</mn><mrow><mi>n</mi><mo>(t)(n(t)−1)</mo></mrow></mfrac><munder><mo>∑</mo><mrow><mi>i</mi><mo>≠</mo><mi>j</mi></mrow></munder><mfrac><mn>1</mn><msub><mi>d</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub></mfrac></math></div><p>Here d<sub>ij</sub> is the shortest sum of reciprocal edge efficiencies, n(t) counts functioning nodes, and unreachable pairs contribute zero. n(t)≤1 gives zero efficiency. TIGER stops when successive edge efficiencies are close under NumPy’s default <code>isclose</code> tolerance (rtol=10<sup>−5</sup>, atol=10<sup>−8</sup>), or when the step limit is reached. Report both; oscillation can persist. <a href="references.html#ref-crucitti2004model">Reference</a></p>

.. _cascades-power-routing-study:

.. raw:: html

   <span id="power-routing-study"></span>

Compare routing models on a power-grid subnetwork
-------------------------------------------------

.. raw:: html

   <p>Use the two-hop neighborhood of the highest-degree node in TIGER’s electrical-network dataset. This fixed selection gives 45 nodes and 71 connections. We use a small subnetwork so exact repeated routing calculations and its drawing remain easy to inspect. It is an induced topology: connections to the rest of the grid are excluded, so its results do not describe the full grid.</p><p>Nodes represent grid components and edges represent connections in the source topology. The data used here supplies neither operating demand nor line limits. Loads are shortest-path betweenness, all initial edge weights are one, and the spring layout is not a geographic map. The larger local-sharing studies on this page use the full 4,941-node network.</p><figure id="power-routing-network"><img loading="lazy" src="guide-results/power-routing-network.svg" alt="45-node empirical power-grid subnetwork; pink marks the initial attack" style="width:100%;height:auto"><figcaption>The same subnetwork and initial high-degree attack are used in all six runs. Node 2553 is attacked; the neighborhood was selected around node 2553.</figcaption></figure><div class="study-introduction" data-study="global-cascade-results"><p>Set capacity to (1+r) times intact betweenness and compare r=0.2, 0.5, and 1 within each model. Motter–Lai removes overloaded nodes; report the largest component divided by the original 45 nodes. Crucitti–Latora–Marchiori retains overloaded nodes but changes edge efficiencies; report weighted efficiency divided by its intact value.</p><p>The code applies the same initial attack and 200-round horizon to all six runs. It records retained connectivity for Motter–Lai and relative weighted efficiency for Crucitti. TIGER averages the latter over surviving nodes, so its denominator changes after the initial attack.</p></div><details class="study-code" data-for-figure="global-cascade-results"><summary>Run the routing-model comparison — code</summary><pre><code>import numpy as np
   import networkx as nx
   from graph_tiger import graphs
   from graph_tiger.cascading import Cascading

   power = graphs.graph_loader(&quot;electrical&quot;)
   center = max(power, key=power.degree)
   G = nx.ego_graph(power, center, radius=2)
   nx.set_edge_attributes(G, 1.0, &quot;weight&quot;)
   results = {}
   for model in [&quot;motter_lai&quot;, &quot;crucitti&quot;]:
       for r in [0.2, 0.5, 1.0]:
           sim = Cascading(G, model=model, r=r, attack=&quot;id_node&quot;, k_a=1,
                           runs=1, steps=200, seed=7)
           raw = np.asarray(sim.run_single_sim())
           baseline = sim.get_efficiency(G) if model == &quot;crucitti&quot; else len(G)
           results[model, r] = raw / baseline</code></pre></details><figure id="global-cascade-results"><img loading="lazy" src="guide-results/power-routing-motter_lai.svg" alt="Retained connectivity over 200 rounds for three Motter–Lai capacities" style="width:100%;height:auto"><img loading="lazy" src="guide-results/power-routing-crucitti.svg" alt="Relative weighted efficiency over 200 rounds for three Crucitti capacities" style="width:100%;height:auto"><figcaption>Exact routing calculations, one initial high-degree removal, 200 rounds, seed 7. These deterministic parameter comparisons have no uncertainty band. Step zero is already attacked.</figcaption></figure><p class="study-interpretation">At round 200, Motter–Lai retains 22.2%, 22.2%, 24.4% of original-node connectivity for r=0.2, 0.5, and 1. The corresponding Crucitti efficiency ratios are 24.0%, 25.0%, 26.5%. Compare capacities within a model, not percentages between these two different outcomes.</p><p>The last 20 Crucitti rounds span efficiency-ratio ranges of 0.0067, 0.0040, 0.0000. A nonzero range indicates continuing variation: a plotted endpoint is not automatically a steady state. More capacity does not by itself guarantee a better finite-time routing outcome.</p><p><a href="guide-results/power-routing.csv">Download all trajectories</a>, <a href="guide-results/power-routing.json">network selection and settings</a>, or the <a href="guide-results/run-cascade-guide-networks.py">complete figure runner</a>.</p>

.. _cascades-completed-local:

.. raw:: html

   <span id="completed-local"></span>

Continue to completed cascade outcomes
--------------------------------------

.. raw:: html

   <div class="study-introduction" data-study="completed-local-results"><p>Return to exactly the electrical graph, initial failure, capacities, and β values used in the 20-round local-sharing experiment. Extend the run limit to 200 rounds rather than changing the model or selecting a different attack.</p><p>The code retains both failed-node and lost-load histories and checks whether any failed node remains to be processed. Completion requires an empty pending-failure queue and stable totals. The plots show the active portion of the runs; the downloadable CSV also retains their flat tails.</p></div><details class="study-code" data-for-figure="completed-local-results"><summary>Run the cascades to completion — code</summary><pre><code>from graph_tiger import graphs
   from graph_tiger.cascading import Cascading

   G = graphs.graph_loader(&quot;electrical&quot;)
   results = {}
   for beta in [0, 1, 2]:
       sim = Cascading(G, model=&quot;local_load_sharing&quot;, beta=beta, r=0.2,
                       attack=&quot;id_node&quot;, k_a=1, runs=1, steps=200, seed=7)
       sim.run_single_sim()
       results[beta] = {
           &quot;failed&quot;: [sim.sim_info[t][&quot;failed&quot;] for t in range(201)],
           &quot;lost_load&quot;: [sim.sim_info[t][&quot;lost_load&quot;] for t in range(201)],
           &quot;pending&quot;: len(sim.failed - sim.processed),
       }</code></pre></details><figure class="study-figure" id="completed-local-results"><a href="guide-results/local-complete-failed.svg" target="_blank"><img src="guide-results/local-complete-failed.svg" alt="Cumulative failures until all three local cascade runs stabilize" loading="lazy"></a><a href="guide-results/local-complete-shed_load.svg" target="_blank"><img src="guide-results/local-complete-shed_load.svg" alt="Lost load through completed local cascades" loading="lazy"></a><figcaption>Same electrical graph, initial degree attack, r=0.2 and seed 7 as the short run. Curves are displayed through stabilization; flat tails are retained in the CSV. <a href="guide-results/local-complete.csv">Data (CSV)</a> · <a href="guide-results/local-complete.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure><p class="study-interpretation">Use the eventual totals to compare completed damage and the approach to those totals to compare timing. Two rules can produce different early trajectories yet fail the same number of nodes in the end.</p><p>Completed failures for β=0, 1 and 2 are 4,941, 4,941, 4,801, respectively. The last observed changes occur at rounds 34, 33, 36. Equal sharing and β=1 ultimately fail the entire network; β=2 leaves 140 nodes functioning. This conclusion differs from comparing only the first 20 rounds.</p>
