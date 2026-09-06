Reproduce the results
=====================

.. raw:: html

   <p>A seed identifies one random realization within a specified software environment. It is not a substitute for repeated runs, recorded graph preprocessing, or a declared stopping rule.</p>

.. _reproducibility-section-1:

.. raw:: html

   <span id="section-1"></span>

Run the documented examples
---------------------------

.. raw:: html

   <p>Download these three files into the same directory: <a href="guide-results/run-guide-studies.py">run-guide-studies.py</a>, <a href="guide-results/run-extended-studies.py">run-extended-studies.py</a>, and <a href="guide-results/run_guide_bootstrap.py">run_guide_bootstrap.py</a>. Then create an environment and install the <a href="guide-results/requirements-lock.txt">recorded dependency versions</a>. The recorded environment used Python 3.12.</p><pre><code>python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements-lock.txt
   .venv\Scripts\python run-guide-studies.py defenses
   .venv\Scripts\python run-extended-studies.py</code></pre><p>On macOS/Linux use <code>.venv/bin/python</code>. Omitting <code>defenses</code> runs the five core guide studies; other choices are <code>attacks</code>, <code>epidemics</code>, <code>cascades</code>, and <code>visualization</code>. Outputs go to <code>site/guide-results</code> beside the scripts. The approximation study has its own <a href="approximation-results/run-approximation.py">runner</a> and <a href="approximation-results/render-approximation.py">renderer</a>; keep both together.</p>

.. _reproducibility-section-2:

.. raw:: html

   <span id="section-2"></span>

What is recorded?
-----------------

.. raw:: html

   <p>Each figure links its observations and parameters. The <a href="guide-results/manifest.json">manifest</a> adds SciPy and other dependency versions, dataset SHA-256 hashes, preprocessing, and experiment-script hashes. Cross-version random streams and spectral calculations can differ. Rebuilt figures use saved observations where available; prose summaries are calculated from those observations during the documentation build.</p>

.. _reproducibility-section-3:

.. raw:: html

   <span id="section-3"></span>

Repeated runs and uncertainty
-----------------------------

.. raw:: html

   <p>A Monte Carlo study samples independent realizations to estimate an outcome’s distribution. Our shaded bands are empirical 10th–90th percentiles, not confidence intervals for the mean. The same integer seed across interventions does not guarantee identical random exposure or transmission draws, because an intervention can consume random numbers or change the graph. Record realized initial infections when pairing comparisons.</p>

.. _reproducibility-section-4:

.. raw:: html

   <span id="section-4"></span>

Outputs and history
-------------------

.. raw:: html

   <div class="table-scroll"><table><thead><tr><th>Call / process</th><th>Returned value</th><th>History</th></tr></thead><tbody><tr><td>run_simulation()</td><td>Mean trajectory, steps+1 entries</td><td>Resets after every realization, including the last.</td></tr><tr><td>run_single_sim()</td><td>One trajectory, steps+1 entries</td><td>Read that realization before resetting; calling again continues current state rather than starting a fresh run.</td></tr><tr><td>Attack / Defense / Motter–Lai / local load sharing</td><td>Chosen measure; raw LCC is a node count</td><td>Normalize by original n when tracking original-network service.</td></tr><tr><td>SIS / SIR</td><td>SIS: infected counts; SIR: recovered counts</td><td>SIR recovered includes pre-vaccinated nodes.</td></tr><tr><td>Crucitti</td><td>Weighted efficiency over surviving pairs</td><td>Report its denominator; it is not a functioning-node fraction.</td></tr></tbody></table></div><p>Mutable set fields such as SIR <code>protected</code> can share a live object with later states. Copy sets when capturing a transition, or rerun the same seed to a fixed horizon and read the current state. Counts and copied status lists are safer for time-series extraction.</p>
    <p>For layouts, plotting options, and export examples, see the <a href="visualization.html">Visualization guide</a>.</p><script>(function(){const anchors=["plotting-options","visualization-layouts","visualization-controls","visualization-export","visualization-compare","visualization-layout-comparison","visualization-edge-comparison","visualization-transition-comparison","visualization-animation-comparison","visualization-gallery-files"];function redirect(){const hash=decodeURIComponent(location.hash.slice(1));if(anchors.includes(hash))location.replace("visualization.html#"+encodeURIComponent(hash));}addEventListener("hashchange",redirect);redirect();})();</script>
