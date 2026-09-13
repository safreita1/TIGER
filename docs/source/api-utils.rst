Utilities API
=============

GPU readiness and backend selection
-----------------------------------

``system_gpu_status`` detects NVIDIA hardware through ``nvidia-smi`` without
requiring CuPy. ``gpu_status`` then verifies that CuPy can see a device,
allocate memory, and synchronize work. ``networkx_gpu_status`` additionally
checks that nx-cugraph is registered with NetworkX.

Automatic selection uses measured per-operation thresholds. ``min_gpu_nodes``
is an optional, nonnegative node-count override used only by
``backend='auto'``: ``None`` selects TIGER's measured operation-specific
default, while an integer makes graphs with at least that many current nodes
GPU-eligible. Runtime availability and spectral memory checks still apply.
The override does not affect sampling, numerical accuracy, or random seeds.
Explicit ``backend='cpu'`` and ``backend='gpu'`` requests ignore the crossover;
GPU requests remain strict and raise when the required runtime is unavailable.
See :doc:`gpu` for examples, workload thresholds, and propagation through
attacks, defenses, cascades, epidemics, and influence simulations.

.. automodule:: graph_tiger.utils
   :members:
   :undoc-members:

GPU diagnostic command
----------------------

.. automodule:: graph_tiger.gpu
   :members:
