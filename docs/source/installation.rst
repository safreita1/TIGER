Installation
============

.. raw:: html

   <p>Install TIGER in a Python environment, then start with a small graph before running a larger empirical study. The default installation is CPU-only.</p><pre><code>python -m pip install graph-tiger==0.8.0</code></pre><p>Optional ForceAtlas2 layouts and edge bundling use the visualization extra.</p><pre><code>python -m pip install "graph-tiger[visualization]==0.8.0"</code></pre>

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

   <p>The package supports Python 3.8 and later. Current GPU extras require Python 3.10 or later for CuPy; nx-cugraph currently requires Python 3.11 or later. These documented results were run on Python 3.14 with dependency versions recorded by each benchmark. Use an isolated environment and start with a small graph. Basic plotting does not need ForceAtlas2 or edge bundling.</p><p>The visualization extra can require a compiler for optional dependencies. If it fails, use NetworkX spring or spectral layouts. Native MP4 export is unavailable on Windows; static figures remain available.</p>

GPU installation
----------------

The base package never installs CUDA libraries or probes hardware during pip's
build step. Install TIGER first and run the readiness diagnostic:

.. code-block:: console

   tiger-gpu-status

Its hardware layer uses ``nvidia-smi`` without importing CuPy; its other layers
also report whether CuPy and nx-cugraph are already usable. If an NVIDIA device
and compatible driver are present, install exactly one CUDA extra. The NVIDIA
package index is required for nx-cugraph:

.. code-block:: console

   python -m pip install "graph-tiger[gpu-cu12]==0.8.0" \
       --extra-index-url https://pypi.nvidia.com
   # Or, for CUDA 13:
   python -m pip install "graph-tiger[gpu-cu13]==0.8.0" \
       --extra-index-url https://pypi.nvidia.com

Run ``tiger-gpu-status`` again after installation. TIGER checks a real CuPy
allocation and synchronization before selecting a GPU. It separately checks
that nx-cugraph is registered with NetworkX. ``backend="auto"`` falls back to
the CPU; ``backend="gpu"`` raises a clear error when its required runtime is
unavailable. RAPIDS centrality acceleration requires Linux or Windows through
WSL2. Native Windows can use TIGER's CuPy measure implementations, but not
nx-cugraph. See :doc:`gpu` for supported operations, selection policy, and
measured performance.

.. _installation-section-3:

.. raw:: html

   <span id="section-3"></span>

Loaders and troubleshooting
---------------------------

.. raw:: html

   <p>See <a href="network-inputs.html">Loading graphs</a> for cache paths, graph conversions, and weights. A disconnected-graph measure error is different from a download failure; check the selected measure’s input conventions before changing the model.</p>
