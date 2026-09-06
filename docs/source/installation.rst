Installation
============

.. raw:: html

   <p>Install TIGER in a Python environment, then start with a small graph before running a larger empirical study.</p><pre><code>python -m pip install graph-tiger==0.6.0</code></pre><p>Optional ForceAtlas2 layouts and edge bundling use the visualization extra.</p><pre><code>python -m pip install "graph-tiger[visualization]==0.6.0"</code></pre>

.. _installation-section-1:

.. raw:: html

   <span id="section-1"></span>

Load a graph and measure it
---------------------------

.. raw:: html

   <pre><code>from graph_tiger.graphs import graph_loader, get_graph_options
   from graph_tiger.measures import run_measure

   print(get_graph_options())
   G = graph_loader("BA", n=100, m=3, seed=17)
   print(run_measure(G, "spectral_radius"))
   print(run_measure(G, "effective_resistance"))</code></pre><p>Supported empirical loaders include <code>ky2</code>, <code>electrical</code>, and <code>as_733</code>. Dataset availability and preprocessing vary by loader; consult the <a href="api-graphs.html">graphs API</a>. TIGER returns undirected NetworkX graphs.</p>

.. _installation-section-2:

.. raw:: html

   <span id="section-2"></span>

Environment and optional dependencies
-------------------------------------

.. raw:: html

   <p>The package supports Python 3.8 and later. These documented results were run on Python 3.12 with versions recorded in the <a href="guide-results/requirements-lock.txt">environment lock</a>. Use an isolated environment and start with a small graph. Basic plotting does not need ForceAtlas2 or edge bundling.</p><p>The visualization extra can require a compiler for optional dependencies. If it fails, use NetworkX spring or spectral layouts. Native MP4 export is unavailable on Windows; static figures remain available. Laplacian GPU calculations are unavailable, while optional adjacency-spectrum calculations require a working CuPy/CUDA environment and a partial-spectrum request.</p>

.. _installation-section-3:

.. raw:: html

   <span id="section-3"></span>

Loaders and troubleshooting
---------------------------

.. raw:: html

   <p>See <a href="network-inputs.html">Loading graphs</a> for cache paths, graph conversions, and weights. A disconnected-graph measure error is different from a download failure; check the selected measure’s input conventions before changing the model.</p>
