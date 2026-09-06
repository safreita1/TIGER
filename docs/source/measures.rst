Robustness measures
===================

.. raw:: html

   <div class="tag">Guide</div><p class="lead">The outcome should determine the measure—not the other way around. A small complementary set is often more defensible than one universal score.</p>
   <div class="table-scroll"><table><thead><tr><th>Question</th><th>Primary measure</th><th>Caution</th></tr></thead><tbody>
   <tr><td>How much remains mutually reachable?</td><td><code>largest_connected_component</code></td><td>Does not describe distances inside the component.</td></tr>
   <tr><td>Can destinations still be reached efficiently?</td><td><code>average_inverse_distance</code></td><td>Assumes shortest paths represent service.</td></tr>
   <tr><td>How many removals disconnect the graph?</td><td><code>node_connectivity</code> / <code>edge_connectivity</code></td><td>A minimum cut is not typical damage.</td></tr>
   <tr><td>Are there alternative routes?</td><td><code>natural_connectivity</code>, <code>number_spanning_trees</code></td><td>Structural redundancy may not equal operational capacity.</td></tr>
   <tr><td>Is routing concentrated?</td><td>vertex/edge betweenness</td><td>An average can conceal extreme components.</td></tr></tbody></table></div>
   <p>The reference also documents diameter, average distance, clustering, spectral radius and gap, spectral scaling, generalized robustness index, algebraic connectivity, and effective resistance.</p>

.. _measures-measure-reference:

.. raw:: html

   <span id="measure-reference"></span>

Measure conventions at a glance
-------------------------------

.. raw:: html

   <p>These conventions describe TIGER 0.6.0; they are not universal guarantees about network robustness. “Exact” distinguishes full calculations from sampling or truncation, not exact arithmetic. <a href="references.html#ref-ellens2013graph">Reference</a></p><div class="table-scroll"><table><thead><tr><th>Measure key</th><th>Output / normalization</th><th>Disconnected graphs</th><th>Weights / approximation</th></tr></thead><tbody><tr><td><code>largest_connected_component</code></td><td>Nodes; divide by original n for attack curves</td><td>Valid; largest remaining component</td><td>No weights; exact</td></tr><tr><td><code>node_connectivity / edge_connectivity</code></td><td>Minimum removal count</td><td>0 when already disconnected</td><td>No weights; exact</td></tr><tr><td><code>diameter</code></td><td>Maximum hop distance</td><td>Undefined; dispatcher can return None</td><td>No weights; exact</td></tr><tr><td><code>average_distance</code></td><td>Mean hop distance</td><td>Undefined; dispatcher can return None</td><td>No weights; exact; rounded to 2 decimals</td></tr><tr><td><code>average_inverse_distance</code></td><td>Mean reciprocal hop distance (global efficiency)</td><td>Unreachable pairs contribute 0</td><td>No weights; exact; 2 decimals</td></tr><tr><td><code>average_vertex_betweenness</code></td><td>Mean unnormalized brokerage over nodes</td><td>Unreachable pairs contribute 0</td><td>No weights; k sampled source nodes; 2 decimals</td></tr><tr><td><code>average_edge_betweenness</code></td><td>Mean unnormalized brokerage over edges</td><td>Unreachable pairs contribute 0; edgeless result 0</td><td>No weights; k sampled source nodes; 2 decimals</td></tr><tr><td><code>average_clustering_coefficient</code></td><td>Mean local clustering, in [0,1]</td><td>Valid; isolated nodes contribute 0</td><td>No weights; exact; 2 decimals</td></tr><tr><td><code>spectral_radius</code></td><td>Largest adjacency eigenvalue</td><td>Defined, but not a connectivity indicator alone</td><td>Uses weight attribute; 2 decimals</td></tr><tr><td><code>spectral_gap</code></td><td>Largest minus second-largest adjacency eigenvalue</td><td>Can remain positive despite disconnection</td><td>Uses weight attribute; 2 decimals</td></tr><tr><td><code>natural_connectivity</code></td><td>Log mean exponential adjacency eigenvalue</td><td>Defined; does not require connectedness</td><td>Uses weight attribute; k retained eigenvalues; 2 decimals</td></tr><tr><td><code>spectral_scaling</code></td><td>Log-scale residual; a structural diagnostic</td><td>May be undefined, especially with zero odd-walk centrality</td><td>Uses weight attribute; k retained eigenpairs</td></tr><tr><td><code>generalized_robustness_index</code></td><td>Truncated spectral-scaling residual</td><td>Same limitations as spectral scaling</td><td>Uses weight attribute; default k=30 for direct function</td></tr><tr><td><code>algebraic_connectivity</code></td><td>Second-smallest Laplacian eigenvalue</td><td>0 for disconnected nonnegative-weight graphs</td><td>Uses weights as conductances; 2 decimals</td></tr><tr><td><code>number_spanning_trees</code></td><td>Tree count, or weighted tree-product sum</td><td>0 when disconnected</td><td>Full spectrum for matrix-tree identity; partial product is not reliable</td></tr><tr><td><code>effective_resistance</code></td><td>Sum over unordered-pair resistances</td><td>Infinity when disconnected; singleton 0</td><td>Weights are conductances, not lengths; partial spectrum underestimates; 2 decimals</td></tr></tbody></table></div><p>The dispatcher supplies its default <code>k=∞</code> even to functions with a different direct-call default. Specify <code>k</code> explicitly when that distinction matters.</p>

.. _measures-definitions:

.. raw:: html

   <span id="definitions"></span>

Three worked measure definitions
--------------------------------

.. _measures-section-1:

.. raw:: html

   <span id="section-1"></span>

Average vertex betweenness
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>For an undirected graph with n = |V| nodes, let σ<sub>st</sub> count shortest paths between distinct endpoints s and t, and σ<sub>st</sub>(u) count those passing through u. Exclude u as an endpoint and count each unordered pair once.</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Node betweenness"><mrow><mrow><mi>b</mi><mo>(</mo><mi>u</mi><mo>)</mo></mrow><mo>=</mo><munder><mo>∑</mo><mtable><mtr><mtd><mrow><mi>s</mi><mo>&lt;</mo><mi>t</mi></mrow></mtd></mtr><mtr><mtd><mrow><mi>s</mi><mo>,</mo><mi>t</mi><mo>≠</mo><mi>u</mi></mrow></mtd></mtr></mtable></munder><mfrac><mrow><msub><mi>σ</mi><mrow><mi>s</mi><mi>t</mi></mrow></msub><mo>(</mo><mi>u</mi><mo>)</mo></mrow><msub><mi>σ</mi><mrow><mi>s</mi><mi>t</mi></mrow></msub></mfrac></mrow></math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Mean node betweenness"><mrow><mover><mi>b</mi><mo>¯</mo></mover><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><munder><mo>∑</mo><mrow><mi>u</mi><mo>∈</mo><mi>V</mi></mrow></munder><mrow><mi>b</mi><mo>(</mo><mi>u</mi><mo>)</mo></mrow></mrow></math></div><p>Disconnected pairs contribute zero. TIGER averages unnormalized betweenness over nodes. A smaller average can reflect shorter routes or lost reachability; it does not establish that routing load is more evenly distributed.</p><pre><code>from graph_tiger.graphs import graph_loader
   from graph_tiger.measures import run_measure

   G = graph_loader("BA", n=300, seed=17)
   exact = run_measure(G, "average_vertex_betweenness")
   approximate = run_measure(G, "average_vertex_betweenness", k=30)</code></pre>

.. _measures-section-2:

.. raw:: html

   <span id="section-2"></span>

Spectral scaling
~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Spectral scaling compares each node’s leading-eigenvector value with its participation in odd closed walks. For adjacency eigenpairs (λ<sub>j</sub>, u<sub>j</sub>), define odd subgraph centrality, a scaling constant C, and each node’s logarithmic residual eᵢ. Spectral scaling ξ is the root-mean-square residual:</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Odd subgraph centrality"><mrow><mrow><msub><mi>SC</mi><mtext>odd</mtext></msub><mo>(</mo><mi>i</mi><mo>)</mo></mrow><mo>=</mo><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><mrow><msub><mi>u</mi><mi>j</mi></msub><mo>(</mo><mi>i</mi><mo>)</mo></mrow><mn>2</mn></msup><mrow><mi>sinh</mi><mo>(</mo><msub><mi>λ</mi><mi>j</mi></msub><mo>)</mo></mrow></mrow></math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Scaling constant"><mrow><mi>C</mi><mo>=</mo><msup><mrow><mo>[</mo><mrow><mi>sinh</mi><mo>(</mo><msub><mi>λ</mi><mn>1</mn></msub><mo>)</mo></mrow><mo>]</mo></mrow><mrow><mo>−</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow></msup></mrow></math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Logarithmic scaling residual"><mrow><msub><mi>e</mi><mi>i</mi></msub><mo>=</mo><mrow><msub><mi>log</mi><mn>10</mn></msub><mrow><mo>(</mo><mfrac><mrow><mo>|</mo><mrow><msub><mi>u</mi><mn>1</mn></msub><mo>(</mo><mi>i</mi><mo>)</mo></mrow><mo>|</mo></mrow><mrow><mi>C</mi><msqrt><mrow><msub><mi>SC</mi><mtext>odd</mtext></msub><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msqrt></mrow></mfrac><mo>)</mo></mrow></mrow></mrow></math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Spectral scaling"><mrow><mi>ξ</mi><mo>=</mo><msqrt><mrow><mfrac><mn>1</mn><mi>n</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><msub><mi>e</mi><mi>i</mi></msub><mn>2</mn></msup></mrow></msqrt></mrow></math></div><p>This is a structural diagnostic of the proposed scaling relationship. Bipartite graphs have no odd closed walks, making the logarithm undefined. TIGER can return <code>None</code> for nonfinite results; do not treat this as zero vulnerability. Its implementation orders eigenpairs by absolute eigenvalue magnitude, which also merits care on bipartite graphs.</p>

.. _measures-section-3:

.. raw:: html

   <span id="section-3"></span>

Effective graph resistance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Treat every edge as a unit resistor. If the connected graph’s Laplacian eigenvalues are 0 = μ<sub>1</sub> &lt; μ<sub>2</sub> ≤ … ≤ μ<sub>n</sub>, the sum of effective resistances over unordered node pairs is:</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Effective graph resistance"><mrow><mi>R</mi><mo>=</mo><munder><mo>∑</mo><mrow><mi>i</mi><mo>&lt;</mo><mi>j</mi></mrow></munder><msub><mi>R</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mi>n</mi><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>2</mn></mrow><mi>n</mi></munderover><mfrac><mn>1</mn><msub><mi>μ</mi><mi>j</mi></msub></mfrac></mrow></math></div><p>Parallel routes reduce resistance. Adding a positive-conductance edge to a connected graph decreases total resistance. TIGER returns infinity for a disconnected graph. Compare like-sized networks, or state a size normalization.</p>

.. _measures-approximation:

.. raw:: html

   <span id="approximation"></span>

Approximation and runtime
-------------------------

.. raw:: html

   <p>The parameter <code>k</code> has a different meaning for different measures: sampled nodes for betweenness and retained eigenvalues for spectral measures. Use the original measure name with <code>k</code>; names ending in <code>_approx</code> are not accepted method keys.</p><div class="table-scroll measure-approximation-table" role="region" aria-label="How k controls each measure approximation" tabindex="0"><table><thead><tr><th scope="col">Measure</th><th scope="col">Role of <code>k</code></th><th scope="col">How to interpret the result</th></tr></thead><tbody><tr><th scope="row">Node and edge betweenness</th><td>Number of sampled source nodes</td><td>Each run estimates betweenness from a sample, so results can vary between samples.</td></tr><tr><th scope="row">Natural connectivity</th><td>Number of retained adjacency eigenvalues</td><td>The truncated exponential sum omits contributions from the remaining eigenvalues.</td></tr><tr><th scope="row">Effective resistance</th><td>Number of retained nonzero Laplacian eigenvalues</td><td>Omitting positive reciprocal terms underestimates the full sum.</td></tr><tr><th scope="row">Spanning-tree count</th><td>Number of retained Laplacian eigenvalues</td><td>A partial spectral product is not the matrix-tree identity and is not generally a reliable estimate.</td></tr></tbody></table></div><p>For graphs smaller than 100 nodes, the current eigensolver uses a full dense spectrum even when a smaller k is supplied. Benchmark larger graphs to observe spectral truncation. The example below evaluates five representative measures and compares each approximation with its exact calculation.</p><div class="study-introduction" data-study="approximation-figure"><p>We generate five clustered scale-free graphs with 300 nodes, m=3, triangle probability 0.3, and seeds 0–4. On each graph, a full calculation provides a separate reference for each measure. The comparison varies k over 10, 30, 60, 100, 200, and 300 without changing that graph.</p><p>The code times each estimate and computes its absolute difference from the full value on the same graph before averaging across graphs. For betweenness, k controls sampled sources; for spectral measures, it controls retained eigenvalues. Each figure pair shows error first and runtime second (side by side on wide screens). Their vertical scales differ between measures, so compare k values within a measure.</p></div><details class="study-code" data-for-figure="approximation-figure"><summary>Run the error and timing comparison — code</summary><pre><code>from time import perf_counter
   import random
   import numpy as np
   import networkx as nx
   import matplotlib.pyplot as plt
   from graph_tiger.measures import run_measure

   names = ["average_vertex_betweenness", "average_edge_betweenness",
            "natural_connectivity", "number_spanning_trees",
            "effective_resistance"]
   ks = [10, 30, 60, 100, 200, 300]
   errors = {name: [] for name in names}
   times = {name: [] for name in names}
   for seed in range(5):
       G = nx.powerlaw_cluster_graph(300, 3, 0.3, seed=seed)
       for name in names:
           exact = run_measure(G, name)
           trial_errors, trial_times = [], []
           for k in ks:
               random.seed(seed)
               np.random.seed(seed)
               start = perf_counter()
               estimate = run_measure(G, name, k=k)
               trial_times.append(perf_counter() - start)
               trial_errors.append(abs(estimate - exact))
           errors[name].append(trial_errors)
           times[name].append(trial_times)
   fig, axes = plt.subplots(2, len(names), figsize=(18, 7))
   for col, name in enumerate(names):
       axes[0, col].plot(ks, np.mean(errors[name], axis=0))
       axes[1, col].plot(ks, np.mean(times[name], axis=0))
       axes[0, col].set_title(name.replace("_", " "), fontsize=9)
       axes[0, col].set_ylabel("Mean absolute error")
       axes[1, col].set_ylabel("Seconds")
       axes[1, col].set_xlabel("k")
   fig.tight_layout()
   fig.savefig("approximation-study.png", dpi=180)</code></pre></details><figure id="approximation-figure" class="approximation-figure"><h3 id="section-4">Approximation error and computation time</h3><div class="approximation-pair"><img src="approximation-results/average_vertex_betweenness-absolute_error.svg" alt="Node betweenness: mean absolute error versus k"><img src="approximation-results/average_vertex_betweenness-seconds.svg" alt="Node betweenness: mean runtime in seconds versus k"></div><div class="approximation-pair"><img src="approximation-results/average_edge_betweenness-absolute_error.svg" alt="Edge betweenness: mean absolute error versus k"><img src="approximation-results/average_edge_betweenness-seconds.svg" alt="Edge betweenness: mean runtime in seconds versus k"></div><div class="approximation-pair"><img src="approximation-results/natural_connectivity-absolute_error.svg" alt="Natural connectivity: mean absolute error versus k"><img src="approximation-results/natural_connectivity-seconds.svg" alt="Natural connectivity: mean runtime in seconds versus k"></div><div class="approximation-pair"><img src="approximation-results/number_spanning_trees-absolute_error.svg" alt="Spanning trees: mean absolute error versus k"><img src="approximation-results/number_spanning_trees-seconds.svg" alt="Spanning trees: mean runtime in seconds versus k"></div><div class="approximation-pair"><img src="approximation-results/effective_resistance-absolute_error.svg" alt="Effective resistance: mean absolute error versus k"><img src="approximation-results/effective_resistance-seconds.svg" alt="Effective resistance: mean runtime in seconds versus k"></div><figcaption>Measured with TIGER 0.6.0 on five clustered scale-free graphs (300 nodes, m = 3, triangle probability 0.3; seeds 0–4). Each point averages five measurements. Errors compare each graph with its full calculation before averaging. Runtime is machine-dependent; k = 300 requests a full calculation.</figcaption><p><a href="approximation-results/approximation-comparison.pdf">Download figure (PDF)</a> · <a href="approximation-results/measurements.csv">Measurements (CSV)</a> · <a href="approximation-results/experiment.json">Experiment settings</a></p></figure><p class="study-interpretation">Seek a useful accuracy–runtime tradeoff rather than the smallest k in isolation. The spanning-tree partial product is not a valid general substitute for the full matrix-tree calculation, and a zero error at returned precision can result from rounding. Runtime describes this measured environment, not a platform-independent speed guarantee.</p><p><strong>Reading the results.</strong> Betweenness error generally falls as more source nodes are sampled, while runtime rises. Spectral runtime is not monotonic: at this graph size the full dense calculation can outperform partial eigensolvers. Spanning-tree error stays extremely large until the full spectrum is used. Natural-connectivity error reaches zero at TIGER’s returned precision; this does not establish equality before rounding.</p><p>Compute absolute error separately on each graph, then average the errors. Taking an absolute difference after averaging estimates can hide cancellation. TIGER rounds several returned measures, so inspect rounding before interpreting very small errors.</p>
