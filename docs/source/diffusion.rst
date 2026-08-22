Diffusion
=========

``Diffusion`` implements synchronous, discrete-time stochastic SIS and SIR
processes on a contact network. Transmission is sampled independently for each
infected-susceptible edge, recovery applies to nodes infected at the start of
the step, and newly infected nodes act beginning in the following step.

.. automodule:: graph_tiger.diffusion
   :members:
   :undoc-members:
   :show-inheritance:
