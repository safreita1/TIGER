Tutorial 4: Cascading Failures
=========================================

A cascade begins with an initial component failure and continues when the
resulting change in load overloads additional components. The appropriate TIGER
model depends on how load moves after the trigger.

- ``motter_lai`` recomputes global shortest-path load and removes overloaded nodes.
- ``crucitti`` reroutes over most-efficient paths and degrades service through
  congested nodes without removing them.
- ``local_load_sharing`` transfers a failed node's load only to its functioning
  neighbors.

Local load sharing
------------------

The local model follows the degree-weighted redistribution rule of
:cite:`wei2012analysis`. A node starts with load equal to its intact degree and
capacity :math:`(1+r)L_i(0)`. When it fails, a neighboring node :math:`j`
receives

.. math::

   \Delta L_{i\rightarrow j}
   =
   L_i
   \frac{k_j^\beta}
   {\sum_{u\in N_i^+} k_u^\beta}.

With ``beta=0``, every functioning neighbor receives the same share. With
``beta=1``, a neighbor of degree 3 receives three times as much as a neighbor
of degree 1. Larger values concentrate the failed load more strongly on
high-degree neighbors.

The following simulation attacks one high-degree node and applies linear
degree preference:

.. code-block:: python
   :name: local-load-sharing

   from graph_tiger.cascading import Cascading
   from graph_tiger.graphs import graph_loader

   graph = graph_loader('electrical')
   cascade = Cascading(
       graph,
       model='local_load_sharing',
       beta=1,
       r=0.2,
       attack='id_node',
       k_a=1,
       defense=None,
       k_d=0,
       runs=1,
       steps=20,
       seed=7,
       plot_transition=False,
       gif_animation=False
   )

   trajectory = cascade.run_single_sim()
   failed_by_step = [cascade.sim_info[t]['failed']
                     for t in range(len(trajectory))]
   shed_by_step = [cascade.sim_info[t]['shed_load']
                   for t in range(len(trajectory))]

Comparing equal and preferential sharing
------------------------------------------

Keep the graph, attack, capacity margin, and seed fixed, then change only
``beta``:

.. code-block:: python
   :name: compare-local-sharing

   results = {}
   for beta in [0, 1, 2]:
       cascade = Cascading(
           graph, model='local_load_sharing', beta=beta, r=0.2,
           attack='id_node', k_a=1, defense=None, k_d=0,
           runs=1, steps=20, seed=7,
           plot_transition=False, gif_animation=False
       )
       cascade.run_single_sim()
       results[beta] = cascade.sim_info[20]['failed']

This comparison isolates the allocation rule. It does not assume that larger
``beta`` is always safer: concentrating load on hubs can prevent low-degree
neighbors from failing, but it can also overload an already stressed hub.

State and interpretation
------------------------

The attacked state is recorded at index 0, followed by one state for each
requested transition. ``failed`` gives the cumulative number of failed nodes.
``status`` records node loads, and ``shed_load`` records load from failed nodes
that had no functioning neighbor. The input graph is not modified.
