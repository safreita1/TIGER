Introduction
============

.. raw:: html

   <div class="tag">TIGER documentation</div>
   <p class="lead">TIGER is a Python toolkit for robustness, vulnerability, and propagation experiments on NetworkX graphs. It connects structural measures with node and edge attacks, defenses, epidemics, influence processes, and cascading failures so that each result answers a stated operational question.</p>
   <div class="call"><strong>Start with the study question.</strong> Robustness is not one universal score. Specify what can fail or spread, what outcome represents service, and what attack or intervention budget is available. Then select the model and measures that match those choices.</div>

.. _index-section-1:

.. raw:: html

   <span id="section-1"></span>

What you can study
------------------

.. raw:: html

   <div class="grid"><div class="card"><p class="card-title"><a href="measures.html">Measure network structure</a></p><p>Evaluate connectivity, reachability, redundancy, path concentration, and spectral structure—and understand what each measure can and cannot establish.</p></div><div class="card"><p class="card-title"><a href="attacks.html">Stress-test the network</a></p><p>Compare random and targeted node or edge attacks under the same removal budget, including policies that recalculate priorities as damage accumulates.</p></div><div class="card"><p class="card-title"><a href="defenses.html">Evaluate interventions</a></p><p>Test node protection, edge addition, and rewiring against a fixed threat model rather than assuming that a structural improvement produces an operational benefit.</p></div><div class="card"><p class="card-title"><a href="epidemics.html">Simulate epidemics</a></p><p>Run synchronous SIS and SIR processes on contact networks and compare intervention outcomes over repeated realizations.</p></div><div class="card"><p class="card-title"><a href="influence.html">Simulate information diffusion</a></p><p>Compare independent-cascade, threshold, voter, and competitive-message dynamics under explicit update rules.</p></div><div class="card"><p class="card-title"><a href="cascades.html">Simulate cascading failures</a></p><p>Use global rerouting or local load sharing when component failures redistribute load through the network.</p></div></div>

.. _index-section-2:

.. raw:: html

   <span id="section-2"></span>

From a graph to a defensible result
-----------------------------------

.. raw:: html

   <ol><li><strong>Load and check the graph.</strong> Use a built-in network, a generator, or your own NetworkX graph. Confirm direction, weights, connectedness, and node attributes before analysis.</li><li><strong>Define the outcome.</strong> Choose a quantity tied to retained service, such as largest-component size, reachability, efficiency, epidemic prevalence, or failed load.</li><li><strong>Specify the disruption.</strong> State the failure, attack, epidemic, or cascade mechanism and its budget or parameters.</li><li><strong>Choose a comparison.</strong> Compare with a random baseline, an alternative attack, or a defense evaluated on the same graph and under the same conditions.</li><li><strong>Repeat and report.</strong> Use multiple runs for stochastic methods, preserve seeds and settings, and show trajectories or distributions rather than only one endpoint.</li></ol>

.. _index-section-3:

.. raw:: html

   <span id="section-3"></span>

Where to go next
----------------

.. raw:: html

   <p>Begin with <a href="installation.html">installation and first steps</a>, then learn how to <a href="network-inputs.html">load graphs</a> and <a href="measures.html">choose robustness measures</a>. For large spectral calculations, see <a href="gpu.html">GPU acceleration</a>. Continue to the attack, defense, epidemic, influence, or cascade guide that matches the process you need to study. The <a href="visualization.html">visualization guide</a> shows how to inspect network states and results, while the <a href="reproducibility.html">reproducibility guide</a> records the conventions used by the documentation experiments.</p>

.. toctree::
   :maxdepth: 3
   :hidden:
   :includehidden:

   self
   installation
   network-inputs
   measures
   gpu
   attacks
   defenses
   epidemics
   influence
   cascades
   visualization
   reproducibility
   historical-figures
   references
   api
