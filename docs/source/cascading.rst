Cascading
=========

``Cascading`` supports three primary models and one compatibility model.

``motter_lai``
    The default node-overload model. Initial shortest-path load determines
    fixed capacity, attacked nodes are removed, and overloaded nodes fail
    synchronously :cite:`motter2002cascade`.

``crucitti``
    The weighted efficiency-dynamics model from
    :cite:`crucitti2004model`. The initial node attack changes the load
    distribution. Overloaded nodes remain active, but every incident edge is
    assigned the capacity-to-load efficiency ratio. Traffic is synchronously
    rerouted over the most efficient weighted paths. The result at each step is
    average network efficiency. In TIGER, ``1 + r`` is the paper's tolerance
    parameter and ``r`` must be positive.

``local_load_sharing``
    Local redistribution with a selectable allocation policy. By default,
    ``allocation='degree'`` uses the preferential rule of
    :cite:`wei2012analysis`. Initial node load is degree and fixed capacity is
    :math:`C_i=(1+r)L_i(0)`. When node :math:`i` fails, its complete current
    load is distributed among its functioning neighbors :math:`N_i^+`:

    .. math::

       \Delta L_{i\rightarrow j}
       =
       L_i
       \frac{k_j^\beta}
       {\sum_{u\in N_i^+} k_u^\beta}.

    The degree :math:`k_j` is measured in the intact graph. The update is
    synchronous: all transfers in one round are accumulated before new
    overloads are identified. If a failed node has no functioning neighbor, its
    load is recorded as ``lost_load``. ``shed_load`` remains a compatibility alias.

    ``allocation='greedy'`` fills the largest pre-round headrooms first for
    each source independently, then splits overflow equally. ``proportional``
    divides the full workload in proportion to pre-round headroom, with equal
    sharing if all headrooms are zero. ``max_flow`` coordinates a headroom-fitting
    pass across every failed source, then sends each source's remainder equally
    to its eligible recipients. These policies do not deliberately discard work
    or cap complete transfers at capacity. See :ref:`local-allocation-policies`.

    The exponent ``beta`` controls the allocation. With ``beta=0``, all
    weights equal one and the model reduces to equal sharing. With ``beta=1``,
    load is proportional to degree; larger values concentrate more load on
    high-degree neighbors. For a failed load of 12 and neighbors with degrees
    1, 2, and 3, ``beta=0`` assigns 4 to each neighbor, whereas ``beta=1``
    assigns 2, 4, and 6.

``legacy_redistribution``
    The corrected historical TIGER redistribution rule. It retains the former
    random betweenness-based initialization for compatibility and is not
    presented as a published model.

All models preserve the caller-owned graph and return the attacked state at
index 0 followed by one state per requested transition. Crucitti state records
also expose ``overloaded`` nodes and ``edge_efficiency`` values. Local
load-sharing records expose cumulative ``lost_load`` and its ``shed_load`` alias.
``last_transfers`` maps each ``(source, recipient)`` pair to the amount sent in
the latest redistribution round.

For local sharing, ``initial_load`` and ``capacities`` can supply complete
node-to-value dictionaries of finite nonnegative numbers. Missing load inputs
default to intact degree; missing capacities default to ``(1+r)`` times initial
load. Explicit capacities are used directly, not multiplied again. Node-defense
options can subsequently increase selected capacities. Inputs are copied and
restored on reset. ``initial_failures`` replaces attack selection and ``k_a``;
an empty list means no initiating failures. These options are rejected for the
other cascade models.

Greedy ties use original graph node insertion order. The maximum-flow solver is
Edmonds-Karp, with sources and recipients inserted in that same order. Record
graph insertion order and NetworkX version: tied maximum flows can produce
different later cascades. The objective is first-pass capacity fitting, not a
guarantee of minimum failures or optimal long-term survival.

A local load-sharing simulation can be created directly:

.. code-block:: python

   cascade = Cascading(
       graph,
       model='local_load_sharing',
       beta=1,
       r=0.2,
       attack='id_node',
       k_a=1,
       runs=1,
       steps=20,
       seed=7
   )
   trajectory = cascade.run_single_sim()

.. automodule:: graph_tiger.cascading
   :members:
   :undoc-members:
   :show-inheritance:
