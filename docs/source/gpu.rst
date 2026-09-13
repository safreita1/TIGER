GPU acceleration
================

TIGER can evaluate its adjacency- and Laplacian-spectrum robustness measures
with SciPy on the CPU or CuPy on an NVIDIA GPU. GPU execution is optional.
Small graphs commonly finish faster on the CPU because creating device arrays
and transferring the matrix can cost more than the eigensolver saves.

Install CuPy
------------

Install a CuPy package that matches the CUDA runtime on the machine. For
example, a CUDA 12 installation normally uses:

.. code-block:: console

   python -m pip install cupy-cuda12x

Confirm the selected package and driver requirements in the `CuPy installation
guide <https://docs.cupy.dev/en/stable/install.html>`_. TIGER does not install
CuPy automatically because the correct package depends on the local CUDA
version.

Check the device
----------------

``gpu_status`` verifies more than package installation. It asks the CUDA
runtime for a device, reads its available memory, completes a small allocation,
and reports why acceleration is unavailable when any of those checks fail.

.. code-block:: python

   from graph_tiger.utils import gpu_status

   print(gpu_status())

A ready device reports ``available=True`` along with its name and memory.
Restart Python after changing a CUDA or CuPy installation.

Choose a backend
----------------

Every call to ``run_measure`` accepts ``backend``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Behavior
   * - ``"cpu"``
     - Always use the SciPy implementation. This remains the default and
       preserves existing behavior.
   * - ``"gpu"``
     - Require a working CUDA device. TIGER raises an error instead of silently
       substituting the CPU when the requested device is unavailable.
   * - ``"auto"``
     - Use the GPU only when it is ready, the graph meets
       ``min_gpu_nodes``, and the estimated eigensolver working set fits within
       half of currently free device memory.

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
       "spectral_radius",
       backend="auto",
       min_gpu_nodes=1000
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
       min_gpu_nodes=1000
   )
   print(decision["selected"], decision["reason"])

The node threshold is a starting policy, not a universal crossover point.
Calibrate it using the benchmark suite on the same GPU, graph families, and
spectral requests used by the intended study.

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

A finite ``k`` requests a sparse partial spectrum where the measure supports
approximation. An infinite ``k`` requests the complete dense spectrum. Partial
spectra generally scale to larger networks, while complete spectra require
quadratic matrix memory and substantially more computation. CPU and GPU calls
must use the same ``k`` when their values and runtimes are compared.

Benchmark CPU and GPU
---------------------

The benchmark exercises all eight supported measures on Barabasi-Albert,
Watts-Strogatz, and Erdos-Renyi graphs. Its default study uses 1,000, 5,000,
and 20,000 requested nodes, three graph realizations per case, two warm-up
runs, and seven timed repetitions per backend.

.. code-block:: console

   python experiments/robustness/gpu_benchmarks.py \
       --output gpu-benchmark-results.csv

The CSV records:

* graph family, realized node and edge counts, density, and random seed;
* the measure and partial-spectrum size;
* cold-start CPU and GPU time;
* median and 10th--90th percentile warm runtime;
* every individual timed observation;
* CPU-to-GPU speedup;
* absolute and relative result error;
* the numerical-parity decision; and
* the backend that automatic selection would choose.

The process exits unsuccessfully if any CPU--GPU comparison violates the
declared numerical tolerances. This makes correctness a prerequisite for
reporting performance.

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
