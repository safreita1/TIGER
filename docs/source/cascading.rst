Cascading
=========

``Cascading`` uses the Motter-Lai node-overload model by default: initial
shortest-path load determines fixed capacity, attacked nodes are removed, and
loads are recomputed synchronously until the cascade stabilizes. The historical
TIGER redistribution rule is available only as ``legacy_redistribution`` and is
not presented as the Crucitti model.

.. automodule:: graph_tiger.cascading
   :members:
   :undoc-members:
   :show-inheritance:
