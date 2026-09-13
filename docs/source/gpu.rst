GPU acceleration
================

TIGER can use CuPy for spectral calculations, exact distance reductions,
connected components, NetShield, and synchronous state-transition models. It
can also use the ``nx-cugraph`` NetworkX backend for supported centrality and
clustering algorithms. GPU execution is optional and the CPU remains the
default. Small graphs commonly finish faster on the CPU because graph
conversion, device-array creation, and result transfer can cost more than the
GPU kernel saves.

Install GPU support
-------------------

The normal TIGER installation is CPU-only. This avoids installing large,
CUDA-specific packages on systems that cannot use them. First install TIGER
and run the readiness report:

.. code-block:: console

   python -m pip install graph-tiger
   tiger-gpu-status

The report's hardware check uses ``nvidia-smi`` without importing CuPy; the
same command also reports whether CuPy and nx-cugraph are already usable. If it
reports an NVIDIA device and compatible driver, install exactly one extra
matching the CUDA generation. The NVIDIA package index supplies nx-cugraph and
its RAPIDS dependencies:

.. code-block:: console

   python -m pip install "graph-tiger[gpu-cu12]" \
       --extra-index-url https://pypi.nvidia.com
   # Or, for CUDA 13:
   python -m pip install "graph-tiger[gpu-cu13]" \
       --extra-index-url https://pypi.nvidia.com

Do not install both CuPy variants in one environment. Current CuPy packages
require Python 3.10 or later. Current nx-cugraph packages require Python 3.11
or later and NetworkX 3.4 or later. NVIDIA supports RAPIDS on Windows through
WSL2 rather than native Windows Python. Native Windows can use TIGER's
CuPy-backed implementations, but not its nx-cugraph-dispatched algorithms.
Confirm current requirements in the `CuPy installation guide
<https://docs.cupy.dev/en/stable/install.html>`_,
`nx-cugraph installation guide <https://docs.nvidia.com/cugraph/latest/nx_cugraph/installation/>`_
and `RAPIDS requirements <https://docs.rapids.ai/install/>`_. TIGER declares
the optional packages but never installs them dynamically at import or runtime.

Check the device
----------------

``tiger-gpu-status`` reports three independent layers: NVIDIA hardware and
driver visibility through ``nvidia-smi``, CuPy execution readiness, and
nx-cugraph registration with NetworkX. ``gpu_status`` verifies more than
package installation: it reads available device memory, completes a small
allocation, synchronizes the default stream, and reports why any check failed.

.. code-block:: python

   from graph_tiger.utils import gpu_status, networkx_gpu_status, system_gpu_status

   print(system_gpu_status())
   print(gpu_status())
   print(networkx_gpu_status())

The ``cupy`` layer of a ready report has ``available=True`` and includes the
device name and memory. The ``networkx`` layer can still be unavailable when
CuPy works but nx-cugraph is not installed. Restart Python after changing a
CUDA or CuPy installation.

Choose a backend
----------------

``run_measure``, ``run_attack_method``, ``run_defense_method``, ``Attack``,
``Defense``, ``Cascading``, ``Diffusion``, and ``Influence`` accept
``backend``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Behavior
   * - ``"cpu"``
     - Always use the NumPy/SciPy/NetworkX implementation. This remains the
       default.
   * - ``"gpu"``
     - Require a working CUDA device and, for dispatched graph algorithms, an
       installed ``nx-cugraph`` backend. TIGER raises instead of silently
       substituting the CPU.
   * - ``"auto"``
     - Use a measured per-operation policy. Marginal or unsupported workloads
       stay on CPU. Spectral calls also require the estimated eigensolver
       working set to fit within half of currently free device memory.

.. code-block:: python

   from graph_tiger.graphs import graph_loader
   from graph_tiger.measures import run_measure

   graph = graph_loader("BA", n=5000, m=3, seed=17)

   cpu_value = run_measure(
       graph, "spectral_radius", backend="cpu"
   )
   gpu_value = run_measure(
       graph, "spectral_radius", backend="gpu"
   )
   automatic_value = run_measure(
       graph,
       "average_distance",
       backend="auto"
   )

The earlier ``use_gpu`` Boolean remains accepted for compatibility. New code
should use ``backend`` because it distinguishes required GPU execution from
automatic selection.

Inspect automatic selection
----------------------------

``select_backend`` exposes the decision before a calculation starts. Its result
includes the requested and selected backends, device readiness, graph size,
the estimated device-memory requirement, and a plain-language reason.

.. code-block:: python

   from graph_tiger.utils import select_backend

   decision = select_backend(
       graph,
       backend="auto",
       k=30,
       operation="effective_resistance"
   )
   print(decision["selected"], decision["reason"])

``min_gpu_nodes`` is the node-count crossover used by supported operations
only when ``backend="auto"`` is requested. TIGER compares ``len(graph)`` with
the threshold immediately before the operation runs. A graph at or above the
threshold is eligible for GPU execution; a smaller graph stays on the CPU.
Eligibility is not a guarantee: the CUDA runtime and required GPU backend must
also be ready, and spectral operations must pass the device-memory check.

The parameter accepts ``None`` or a nonnegative integer:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Value
     - Meaning with ``backend="auto"``
   * - ``None``
     - Use TIGER's measured threshold for the particular operation and its
       exact or approximate form. This is the recommended default.
   * - ``5000``
     - Make the GPU eligible when the current graph contains at least 5,000
       nodes, subject to runtime and memory checks.
   * - ``0``
     - Make every supported graph size GPU-eligible. This is useful for
       crossover experiments, but it is not equivalent to
       ``backend="gpu"`` because unavailable runtimes still fall back to CPU.

The threshold does not change ``k``, sampling, numerical tolerances, random
seeds, or the result definition. ``backend="cpu"`` and ``backend="gpu"`` do
not use it to choose a device: CPU is always CPU, while explicit GPU is strict
and raises if its runtime is unavailable or its estimated spectral working set
is too large.

For example:

.. code-block:: python

   # Use TIGER's measured average-distance crossover.
   run_measure(graph, "average_distance", backend="auto")

   # Override the crossover for this call only.
   run_measure(
       graph,
       "average_distance",
       backend="auto",
       min_gpu_nodes=5000
   )

   # Require GPU execution regardless of the automatic crossover.
   run_measure(graph, "average_distance", backend="gpu")

Without an override, TIGER uses this conservative policy for sparse graphs:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Workload
     - Automatic GPU threshold
   * - Diameter and average distance measures
     - 250 nodes
   * - Exact or sampled vertex betweenness
     - 500 nodes
   * - Exact or sampled edge betweenness
     - 5,000 nodes
   * - Average clustering
     - 5,000 nodes
   * - Exact natural connectivity, spanning trees, or effective resistance
     - 250 nodes, subject to memory
   * - Approximate spanning trees or effective resistance
     - 20,000 nodes
   * - PageRank or eigenvector attacks and defenses
     - 5,000 nodes
   * - Motter--Lai
     - 1,000 nodes
   * - Spectral radius, spectral gap, or algebraic connectivity
     - 20,000 nodes
   * - SIS, SIR, influence, NetShield, largest component, spectral scaling,
       GRI, and marginal partial-spectrum cases
     - CPU

These defaults come from an RTX 5090 study on sparse graphs with mean degree
eight. A threshold is not a hardware-independent guarantee. Explicit
``backend="gpu"`` remains available, and ``min_gpu_nodes`` supports local
calibration without changing TIGER's defaults.

``Attack``, ``Defense``, ``Cascading``, ``Diffusion``, and ``Influence`` store
the value in their parameter dictionaries and pass it to supported downstream
measure, centrality, and simulation operations. Operations on a changing graph
reconsider the current node count, so automatic execution may move from GPU to
CPU after enough nodes are removed. CPU-only operations still remain on CPU,
and largest-connected-component measurement deliberately ignores an automatic
override because its measured GPU implementation was slower at every tested
size. Use ``backend="gpu"`` when explicitly testing that implementation.

For NetworkX-dispatched algorithms, use ``networkx_gpu_status`` and
``select_networkx_backend``. The former distinguishes a working CuPy device
from a complete ``nx-cugraph`` installation.

Supported measures
------------------

GPU execution is available for all TIGER measures whose main calculation is an
adjacency or Laplacian eigensolution:

* spectral radius;
* spectral gap;
* natural connectivity;
* spectral scaling;
* generalized robustness index;
* algebraic connectivity;
* number of spanning trees; and
* effective resistance.

Average vertex and edge betweenness and average clustering can also use
``nx-cugraph``. TIGER applies the missing ``n/k`` rescaling to approximate edge
betweenness so its result matches NetworkX 3.6.1 rather than silently returning
a value smaller by the sampling fraction.

Diameter, average distance, and average inverse distance share an exact CuPy
implementation. It performs batched breadth-first traversals and reduces the
ordered distance sum, inverse-distance sum, diameter, and reachable-pair count
on the device instead of materializing all-pairs Python dictionaries. Largest
connected component also has an exact CuPy implementation. Its explicit GPU
path is useful for parity testing, but automatic selection keeps it on the CPU
because end-to-end GPU time was slower on every measured graph through 20,000
nodes.

For measures that expose it, a finite ``k`` has one of two meanings: retain
``k`` eigenpairs for a partial-spectrum measure, or sample ``k`` sources for a
betweenness measure. Clustering, diameter, average distance, average inverse
distance, and largest connected component are exact on both backends.

Node connectivity and edge connectivity remain CPU-only. ``nx-cugraph`` does
not currently implement the NetworkX connectivity calls TIGER uses for them.

Supported application areas
---------------------------

* PageRank, eigenvector, betweenness, and NetShield node or line-graph attacks;
* matching centrality defenses and centrality-based edge additions;
* unweighted Motter--Lai load recomputation;
* synchronous SIS and SIR epidemics;
* independent cascade, linear threshold, and competitive cascade.

Attack and Defense pass their ``backend`` and ``min_gpu_nodes`` settings into
``run_measure`` at every tracked step. Any GPU-capable measure listed above can
therefore be selected as ``robust_measure`` without a second integration path.
Automatic dispatch is evaluated again as the attacked graph shrinks, and the
largest-connected-component measure continues to select the CPU automatically.

The synchronous stochastic models use the same counter-based random values on
NumPy and CuPy. Equal seeds therefore produce identical trajectories, node
states, and recovery sets on CPU and GPU. This backend-independent stream can
produce different seeded trajectories than earlier TIGER releases, which used
iteration-dependent Python random draws. Their measured end-to-end speedups
were only 0.47x--1.06x through 20,000 nodes, so ``backend="auto"`` keeps SIS,
SIR, and influence models on CPU. Explicit GPU execution remains available for
experimentation. The asynchronous voter model, rewiring, local load allocation,
and maximum flow remain CPU-only. Weighted betweenness used by the Crucitti
path is not currently implemented by nx-cugraph and remains on CPU.

Benchmark CPU and GPU
---------------------

The benchmark supports all fifteen GPU-capable robustness measures on
Barabasi-Albert, Watts-Strogatz, and Erdos-Renyi graphs. The default command
retains the eight-measure spectral study at 1,000, 5,000, and 20,000 requested
nodes, three graph realizations per case, two warm-up runs, and seven timed
repetitions per backend. Select non-spectral measures explicitly because exact
all-pairs distance studies can be much longer than partial-spectrum runs.

The complete source is in the `TIGER experiments directory on GitHub
<https://github.com/safreita1/TIGER/tree/master/experiments>`_. The relevant
drivers are the `robustness-measure benchmark
<https://github.com/safreita1/TIGER/blob/master/experiments/robustness/gpu_benchmarks.py>`_,
the `benchmark report plotter
<https://github.com/safreita1/TIGER/blob/master/experiments/robustness/plot_gpu_benchmarks.py>`_,
and the `attack, defense, cascade, epidemic, and influence benchmark
<https://github.com/safreita1/TIGER/blob/master/experiments/gpu_application_benchmarks.py>`_.

.. code-block:: console

   python experiments/robustness/gpu_benchmarks.py \
       --output gpu-benchmark-results.csv

For example, benchmark the remaining exact and sampled measures with:

.. code-block:: console

   python experiments/robustness/gpu_benchmarks.py \
       --measures average_distance average_vertex_betweenness \
       average_edge_betweenness average_clustering_coefficient \
       largest_connected_component \
       --output gpu-remaining-measure-results.csv

The CSV records:

* graph family, realized node and edge counts, density, and random seed;
* the measure and its approximation parameter ``k`` (retained eigenpairs or
  sampled sources, depending on the measure);
* cold-start CPU and GPU time;
* median and 10th--90th percentile warm runtime;
* every individual timed observation;
* CPU-to-GPU speedup;
* absolute and relative result error;
* the numerical-parity decision; and
* the backend that automatic selection would choose.

The process exits unsuccessfully if any CPU--GPU comparison violates the
declared numerical tolerances. Every case compares the returned measure; a
spectral case also compares its unrounded eigenvalues. Correctness is therefore
a prerequisite for reporting performance. Run a separate dense-spectrum study
on smaller graphs when exact natural connectivity, spectral scaling,
spanning-tree, or resistance timing is needed:

.. code-block:: console

   python experiments/robustness/gpu_benchmarks.py --nodes 250 500 --exact \\
       --output gpu-exact-benchmark-results.csv

Exact-measure speedup
~~~~~~~~~~~~~~~~~~~~~

For each graph-family and seed case, the benchmark first discards the warm-up
runs and calculates:

``case speedup = median CPU wall time / median GPU wall time``

The times cover the complete public TIGER call, including graph/matrix
conversion, host-to-device transfer, computation, synchronization, and return
of the result to CPU memory. CPU and GPU calls are alternated to reduce ordering
bias. An ordinary table cell is the median of the nine case speedups produced by
three graph families and three seeds at that size; it is not a ratio computed
from all observations pooled together.

A value of ``4.00x`` means the CPU took four times as long, equivalently the GPU
used one quarter of the CPU time. ``1.00x`` means equal time. ``0.50x`` means
the GPU took twice as long, so values below ``1.00x`` favor the CPU.

Values marked :sup:`†` are single Barabasi--Albert probes rather than nine-case
medians. ``CPU-only`` means no GPU implementation exists. ``CPU-dominated
timeout`` means the benchmark could not complete a valid CPU/GPU pair within
900 seconds because the CPU leg dominated the limit; no speedup is inferred.
All 1,120 completed measure and application comparisons passed their numerical
or state-parity gates before a speedup was accepted.

Exact spectral measures
^^^^^^^^^^^^^^^^^^^^^^^

These measures compute eigenvalues or eigenvectors of the adjacency or
Laplacian matrix. ``Spectral radius``, ``spectral gap``, and ``algebraic
connectivity`` need only one or two extremal eigenpairs; the other rows use the
complete spectrum in this table.

.. csv-table:: Exact spectral-measure speedup
   :header: "Measure", "250 nodes", "500 nodes", "1,000 nodes", "5,000 nodes", "20,000 nodes"
   :widths: 34, 12, 12, 12, 14, 16

   "Spectral radius", "0.09x", "0.13x", "0.21x", "0.57x", "1.12x"
   "Spectral gap", "0.06x", "0.08x", "0.12x", "0.38x", "1.16x"
   "Algebraic connectivity", "0.12x", "0.13x", "0.23x", "0.40x", "1.32x"
   "Natural connectivity", "2.14x", "5.79x", "2.38x", "3.83x :sup:`†`", "7.93x :sup:`†`"
   "Spectral scaling", "1.19x", "1.20x", "1.11x", "1.16x :sup:`†`", "CPU-dominated timeout"
   "Generalized robustness index", "1.19x", "1.20x", "1.13x", "1.16x :sup:`†`", "CPU-dominated timeout"
   "Spanning trees", "1.96x", "5.27x", "1.94x", "3.08x :sup:`†`", "7.83x :sup:`†`"
   "Effective resistance", "1.95x", "5.15x", "1.95x", "3.06x :sup:`†`", "7.75x :sup:`†`"

Spectral radius, spectral gap, and algebraic connectivity do not favor the GPU
until 20,000 nodes. The complete-spectrum measures benefit earlier, but require
quadratic matrix storage and are subject to the memory and timeout limits
described above.

Exact path, centrality, and connectivity measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These rows use graph traversal, centrality, clustering, or connected-component
algorithms rather than eigensolvers. They therefore have different scaling,
conversion costs, and GPU crossover points from the spectral measures.

.. csv-table:: Exact path, centrality, and connectivity speedup
   :header: "Measure", "250 nodes", "500 nodes", "1,000 nodes", "5,000 nodes", "20,000 nodes"
   :widths: 34, 12, 12, 12, 14, 16

   "Vertex betweenness", "4.16x", "10.66x", "22.69x", "95.42x :sup:`†`", "CPU-dominated timeout"
   "Edge betweenness", "0.07x", "0.13x", "0.27x", "1.57x :sup:`†`", "CPU-dominated timeout"
   "Average clustering", "0.25x", "0.54x", "1.17x", "4.93x", "12.69x"
   "Diameter", "4.35x", "8.62x", "31.75x", "98.51x", "331.37x :sup:`†`"
   "Average distance", "4.40x", "8.32x", "32.37x", "97.88x", "323.15x :sup:`†`"
   "Average inverse distance", "4.29x", "8.14x", "31.88x", "97.58x", "357.33x :sup:`†`"
   "Largest connected component", "0.07x", "0.04x", "0.06x", "0.12x", "0.18x"
   "Node connectivity", "CPU-only", "CPU-only", "CPU-only", "CPU-only", "CPU-only"
   "Edge connectivity", "CPU-only", "CPU-only", "CPU-only", "CPU-only", "CPU-only"

Distance measures and vertex betweenness strongly favor the GPU. Exact edge
betweenness crosses over later, clustering crosses over near 1,000 nodes, and
largest connected component remains faster on CPU.

Representative 20,000-node complete-spectrum wall times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Measure", "CPU seconds", "GPU seconds", "Speedup"
   :widths: 46, 18, 18, 18

   "Natural connectivity", "115.58", "14.58", "7.93x"
   "Spanning trees", "116.11", "14.83", "7.83x"
   "Effective resistance", "115.99", "14.96", "7.75x"

Representative 20,000-node distance wall times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Measure", "CPU seconds", "GPU seconds", "Speedup"
   :widths: 46, 18, 18, 18

   "Diameter", "209.73", "0.633", "331.37x"
   "Average distance", "210.70", "0.652", "323.15x"
   "Average inverse distance", "209.81", "0.587", "357.33x"

Render the results
------------------

Generate a multipage PDF containing an overall speedup comparison and a
runtime, crossover, and numerical-agreement page for every measure:

.. code-block:: console

   python experiments/robustness/plot_gpu_benchmarks.py \
       gpu-benchmark-results.csv \
       --output gpu-benchmark-results.pdf

Interpret end-to-end time rather than eigensolver time alone. The public call
must construct a matrix, transfer it when needed, run the solver, and return a
NumPy result. Those costs determine whether GPU execution benefits an actual
TIGER study.

Benchmark applications
----------------------

The non-spectral application benchmark covers three graph families, 1,000,
5,000, and 20,000 nodes, three topology seeds, ten workloads, one warm-up, and
three alternating timed repetitions:

.. code-block:: console

   python experiments/gpu_application_benchmarks.py \
       --output gpu-application-benchmark-results.csv

The benchmark exits unsuccessfully if any CPU/GPU selection, trajectory,
final state, failed-node set, or load comparison violates its parity contract.
On an RTX 5090 with NetworkX 3.6.1 and nx-cugraph 26.08, all 270 comparisons
passed. Application speedup uses the same calculation as the measure tables:
median CPU time divided by median GPU time for each graph/seed case, followed
by the median of the nine case ratios at each size.

.. csv-table:: Application median speedup
   :header: "Workload", "1,000 nodes", "5,000 nodes", "20,000 nodes"
   :widths: 40, 20, 20, 20

   "Betweenness attack", "5.82x", "17.00x", "41.44x"
   "Eigenvector attack", "0.91x", "2.35x", "3.06x"
   "PageRank attack", "0.59x", "1.52x", "2.60x"
   "PageRank defense", "0.56x", "1.60x", "2.62x"
   "Motter--Lai", "3.42x", "6.24x", "9.13x"
   "SIR", "0.48x", "0.80x", "0.98x"
   "SIS", "0.48x", "0.93x", "0.88x"
   "Independent cascade", "0.62x", "0.96x", "1.01x"
   "Linear threshold", "0.74x", "1.03x", "1.02x"
   "Competitive cascade", "0.47x", "0.80x", "1.06x"

The results support automatic GPU selection for centrality attacks and
defenses and for Motter--Lai at their documented crossover sizes. Full-state
epidemic and influence models remain near parity because every recorded node
state returns to the CPU, so automatic execution keeps them on CPU. Explicit
GPU requests and ``min_gpu_nodes`` overrides remain available for local
experiments.
