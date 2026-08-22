Cascading
=========

``Cascading`` supports three explicitly named models.

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
    average network efficiency, so it is already normalized to the interval
    from 0 to 1. In TIGER, ``1 + r`` is the paper's tolerance parameter and
    ``r`` must be positive.

``legacy_redistribution``
    The corrected historical TIGER redistribution rule. It is TIGER-specific
    and is not presented as either published model.

All models preserve the caller-owned graph and return the attacked state at
index 0 followed by one state per requested transition. Crucitti state records
also expose ``overloaded`` nodes and ``edge_efficiency`` values.

.. automodule:: graph_tiger.cascading
   :members:
   :undoc-members:
   :show-inheritance:
