Epidemic simulations
====================

.. raw:: html

   <div class="tag">Guide</div>
   <p class="lead">Use SIS when recovery returns a node to susceptibility and reinfection is possible. Use SIR when recovery produces lasting immunity. These histories require different outcomes and interpretations.</p>
   <p>The classical SIR framework was introduced by <a href="references.html#ref-kermack">Kermack and McKendrick (1927)</a>. TIGER implements discrete-time network transitions; its per-step probabilities are distinct from the rates in the original continuous-time equations.</p>

.. _epidemics-models:

.. raw:: html

   <span id="models"></span>

Choose an epidemic model
------------------------

.. _epidemics-sis-model:

.. raw:: html

   <span id="sis-model"></span>

SIS: reinfection is possible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Susceptible nodes can become infected, and recovered nodes immediately become susceptible again. Use SIS when infection can recur and current prevalence or extinction is the primary outcome.</p>

.. _epidemics-sir-model:

.. raw:: html

   <span id="sir-model"></span>

SIR: recovery is lasting
~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Recovered nodes enter an absorbing state and cannot be infected again. Use SIR when the questions concern current prevalence, the timing of the outbreak, or the total fraction eventually infected.</p>

.. _epidemics-si-limit:

.. raw:: html

   <span id="si-limit"></span>

SI: the no-recovery limit
~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>SI permits transmission but no recovery, so the infected set can only grow. TIGER exposes <code>SIS</code> and <code>SIR</code> as named models; use <code>model="SIS"</code> with <code>d=0</code> for the same SI state transitions. <code>model="SI"</code> is not an accepted value.</p>
   <div class="table-scroll"><table><thead><tr><th>Setting</th><th>Model</th><th>Report</th><th>Interpretation</th></tr></thead><tbody>
   <tr><td>No lasting immunity</td><td><code>SIS</code></td><td>Infected fraction over time; persistence/extinction</td><td>A finite realization reaching zero cannot restart without external infection.</td></tr>
   <tr><td>Lasting immunity</td><td><code>SIR</code></td><td>Prevalence and final recovered fraction</td><td>Final recovered fraction is outbreak size, not current prevalence.</td></tr></tbody></table></div>
   <div class="call"><strong>Parameters.</strong> <code>b</code> is the per-step transmission probability to each susceptible neighbor, <code>d</code> is the per-step recovery probability for each infected node, and <code>c</code> is the initially infected fraction. State what one time step represents.</div>
   <pre><span class="kw">from</span> graph_tiger.diffusion <span class="kw">import</span> Diffusion
   <span class="kw">from</span> graph_tiger.graphs <span class="kw">import</span> karate

   G = karate()
   sis = Diffusion(G, model=<span class="st">"SIS"</span>, runs=100, steps=150,
                   b=0.08, d=0.20, c=0.05, seed=17)
   sir = Diffusion(G, model=<span class="st">"SIR"</span>, runs=100, steps=150,
                   b=0.08, d=0.20, c=0.05, seed=17)
   sis_curve = sis.run_simulation()
   sir_curve = sir.run_simulation()</pre>
   <p><code>sis_curve</code> contains mean infected counts; <code>sir_curve</code> contains mean recovered counts. Divide by <code>len(G)</code> to obtain fractions. A SIR prevalence curve requires infected counts from the per-step simulation state.</p>
   <div class="figure"><svg viewBox="0 0 900 290" width="100%" aria-label="SIS and SIR trajectories">
   <g transform="translate(35 20)"><text x="175" y="15" text-anchor="middle" font-weight="700">SIS: infection may persist</text><path d="M0 230H360M0 230V35" stroke="#607a80" stroke-width="2"/><path d="M0 215C55 65 105 68 145 135S225 183 270 130S330 98 360 122" fill="none" stroke="#c54c54" stroke-width="7"/><text x="155" y="270">time</text><text x="18" y="55" fill="#c54c54">infected</text></g>
   <g transform="translate(505 20)"><text x="175" y="15" text-anchor="middle" font-weight="700">SIR: outbreak ends</text><path d="M0 230H360M0 230V35" stroke="#607a80" stroke-width="2"/><path d="M0 218C75 205 95 48 170 72S245 197 360 222" fill="none" stroke="#c54c54" stroke-width="7"/><path d="M0 227C95 222 130 168 190 110S295 64 360 62" fill="none" stroke="#3979aa" stroke-width="7"/><text x="155" y="270">time</text><text x="250" y="47" fill="#3979aa">recovered</text><text x="218" y="202" fill="#c54c54">infected</text></g></svg>
   <div class="cap">Schematic trajectories, not simulation results. SIS permits reinfection; SIR recovery accumulates as the outbreak ends.</div></div>

.. _epidemics-simulation-rule:

.. raw:: html

   <span id="simulation-rule"></span>

Simulation rule and time scale
------------------------------

.. _epidemics-transition:

.. raw:: html

   <span id="transition"></span>

Synchronous transitions
~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>TIGER uses synchronous discrete-time updates. If a susceptible node has m infected neighbors at the beginning of a step, independent transmission attempts give infection probability 1 − (1−b)<sup>m</sup>. Nodes already infected recover independently with probability d. All changes are then applied together, so newly infected nodes begin transmitting on the next step. Recovery returns a node to S in SIS or to the absorbing R state in SIR.</p>

.. _epidemics-time-calibration:

.. raw:: html

   <span id="time-calibration"></span>

Calibrate the time step
~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>The probabilities <code>b</code> and <code>d</code> apply once per simulation step, so define what one step represents. Reusing a daily probability in an hourly simulation creates 24 opportunities per day and changes the process. Under constant transmission and recovery rates λ and μ, convert rates to probabilities for a step of duration Δt:</p><div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Transmission probability for one time step"><mrow><msub><mi>b</mi><mrow><mi>Δ</mi><mi>t</mi></mrow></msub><mo>=</mo><mn>1</mn><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>λ</mi><mi>Δ</mi><mi>t</mi></mrow></msup></mrow></math><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Recovery probability for one time step"><mrow><msub><mi>d</mi><mrow><mi>Δ</mi><mi>t</mi></mrow></msub><mo>=</mo><mn>1</mn><mo>−</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>μ</mi><mi>Δ</mi><mi>t</mi></mrow></msup></mrow></math></div><p>TIGER does not perform this conversion or attach physical units to a step; supply probabilities calibrated to the time scale of the study.</p>

.. _epidemics-strength:

.. raw:: html

   <span id="strength"></span>

Effective strength
~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Effective epidemic strength"><mrow><mi>s</mi><mo>=</mo><msub><mi>λ</mi><mtext>max</mtext></msub><mo>(</mo><mi>A</mi><mo>)</mo><mfrac><mi>b</mi><mi>d</mi></mfrac><mo>,</mo><mspace width='1em'/><mi>d</mi><mo>&gt;</mo><mn>0</mn></mrow></math></div><p>The value combines network connectivity with transmission relative to recovery. A threshold near s=1 comes from a linearized mean-field description; it is not an exact persistence guarantee for a finite stochastic network. A finite closed SIS process can eventually go extinct even when it remains active for a long time.</p><p><code>get_effective_strength()</code> computes the score from the simulation graph. With node vaccination, vaccinated nodes remain in that graph, so the score does not automatically measure the susceptible contact subgraph.</p>

.. _epidemics-outputs:

.. raw:: html

   <span id="outputs"></span>

Initial conditions, runs, and outputs
-------------------------------------

.. raw:: html

   <p><code>c</code> is the initially infected fraction; TIGER selects <code>floor(c × n)</code> nodes uniformly at random before applying vaccination. <code>run_single_sim()</code> returns one trajectory and leaves its per-step state in <code>sim_info</code>. <code>run_simulation()</code> runs the requested number of realizations and returns their mean primary trajectory.</p><p>For SIS, the primary trajectory is the current infected count. For SIR, it is the recovered count; read <code>sim_info[t]["failed"]</code> for current infection. Use repeated runs when reporting extinction probability, peak prevalence, outbreak size, or uncertainty. A mean curve alone does not show the distribution of outcomes.</p>

.. _epidemics-interventions:

.. raw:: html

   <span id="interventions"></span>

Interventions
-------------

.. raw:: html

   <p>To suppress infection, use node selection for vaccination or edge removal to reduce contacts. To promote a desirable diffusion process, use edge addition or rewiring to create transmission opportunities. Compare interventions with the same model, graph, time scale, initial infection rule, run count, and action budget.</p>

.. _epidemics-sis-studies:

.. raw:: html

   <span id="sis-studies"></span>

SIS studies
-----------

.. _epidemics-intervention-study:

.. raw:: html

   <span id="intervention-study"></span>

Vaccination and transmission sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <p>Compare no vaccination, random vaccination, and NetShield vaccination at the same budget. Vary b while keeping recovery, graph, initial infection fraction, duration, and run count fixed.</p><div class="study-introduction" data-study="epidemic-results"><p>The contact network is one fixed 100-node Barabási–Albert graph (m=3, graph seed 17). We compare no vaccination, five randomly vaccinated nodes, and five NetShield-selected nodes. Transmission probabilities are 0.02, 0.04, and 0.08 per infected–susceptible contact per step; recovery probability is 0.2.</p><p>For each setting, the code runs 20 seeds for 150 steps with an initial infection fraction of 0.1. It reads infected counts from the SIS trajectory and divides by 100. Compare the same transmission color across the three panels. Vaccinated initial infections are removed rather than replaced, so the intervention can change both initial prevalence and later spread.</p></div><details class="study-code" data-for-figure="epidemic-results"><summary>Run the SIS vaccination study — code</summary><pre><code>import numpy as np
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import graph_loader
   from graph_tiger.diffusion import Diffusion

   G = graph_loader("BA", n=100, m=3, seed=17)
   for method in [None, "rnd_node", "ns_node"]:
       plt.figure()
       for b in [0.02, 0.04, 0.08]:
           trials = []
           for seed in range(20):
               sim = Diffusion(G, model="SIS", b=b, d=0.2, c=0.1,
                               runs=1, steps=150, seed=seed,
                               diffusion=None if method is None else "min",
                               method=method, k=0 if method is None else 5)
               trials.append(np.asarray(sim.run_single_sim()) / len(G))
           values = np.asarray(trials)
           x = np.arange(values.shape[1])
           plt.plot(x, values.mean(axis=0), label=f"b={b:.2f}")
           plt.fill_between(x, *np.quantile(values, [0.1, 0.9], axis=0), alpha=0.12)
       plt.title(method or "No vaccination")
       plt.xlabel("Simulation steps")
       plt.ylabel("Infected fraction")
       plt.ylim(0, 1)
       plt.legend()
   plt.show()</code></pre></details><figure class="study-figure" id="epidemic-results"><a href="guide-results/epidemics-none.svg" target="_blank"><img src="guide-results/epidemics-none.svg" alt="SIS prevalence without vaccination for three transmission probabilities" loading="lazy"></a><a href="guide-results/epidemics-rnd_node.svg" target="_blank"><img src="guide-results/epidemics-rnd_node.svg" alt="SIS prevalence with five randomly vaccinated nodes" loading="lazy"></a><a href="guide-results/epidemics-ns_node.svg" target="_blank"><img src="guide-results/epidemics-ns_node.svg" alt="SIS prevalence with five NetShield vaccinated nodes" loading="lazy"></a><figcaption>Means and 10th–90th percentile ranges across seeds 0–19; b=0.02, 0.04, 0.08, d=0.2, c=0.1. All panels share the same axes. Vaccinated initial infections are removed rather than replaced, so vaccination can change both initial prevalence and subsequent transmission. <a href="guide-results/epidemics.csv">Run data (CSV)</a> · <a href="guide-results/epidemics.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">Higher curves mean more infection is present at that time, not that more distinct people have ever been infected: SIS permits reinfection. Shading shows variability between runs, and a finite-time plateau is not proof of indefinite persistence.</p><p>At b=0.04, mean prevalence at step 150 is 17.4% without vaccination, 15.3% with random vaccination, and 0.0% with NetShield. These finite-horizon results are not universal epidemic thresholds.</p><details><summary>Optional larger AS-733 benchmark</summary><p>This longer setup is retained for further study; the figures above are not its results.</p><pre><code>import numpy as np
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import graph_loader
   from graph_tiger.diffusion import Diffusion

   G = graph_loader("as_733")
   fig, axes = plt.subplots(1, 3, figsize=(14, 4))
   for ax, method in zip(axes, [None, "rnd_node", "ns_node"]):
       for b in [0.0, 0.001, 0.002, 0.003, 0.004]:
           sim = Diffusion(G, model="SIS", b=b, d=0.01, c=1,
                           runs=10, steps=5000, seed=17,
                           diffusion=None if method is None else "min",
                           method=method, k=0 if method is None else 5)
           prevalence = np.asarray(sim.run_simulation()) / len(G)
           ax.plot(prevalence, label=f"b={b:.3f}")
       ax.set(title=method or "No vaccination", xlabel="Steps",
              ylabel="Infected fraction", ylim=(0, 1))
       ax.legend()
   fig.tight_layout()
   fig.savefig("epidemic-interventions.png", dpi=180)</code></pre></details><p>The optional benchmark uses 5,000 steps on AS-733. The <code>as_733</code> loader selects the AS-733 snapshot, which is distinct from the separately available Oregon-1 dataset. Initially selected infected nodes that are vaccinated are removed from the infected set, so realized initial infection counts can differ between interventions.</p>

.. _epidemics-added-contact-study:

.. raw:: html

   <span id="added-contact-study"></span>

Added-contact comparison
~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="study-introduction" data-study="spread-results"><p>Keep the karate network and SIS probabilities fixed (transmission 0.08, recovery 0.2, initial infection fraction 0.1), then compare its original contacts with five randomly added edges. The question is whether more contact opportunities increase current prevalence, which may be desirable for information spread but harmful for infection.</p><p>The code adds contacts before each simulation and runs both conditions for 150 steps over 20 seeds. It averages infected fractions at each step. Only edge addition is used here; this is not a comparison of all possible rewiring or targeting policies.</p></div><details class="study-code" data-for-figure="spread-results"><summary>Run the added-contact comparison — code</summary><pre><code>import numpy as np
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import karate
   from graph_tiger.diffusion import Diffusion

   G = karate()
   for method in [None, "add_edge_random"]:
       trials = []
       for seed in range(20):
           sim = Diffusion(G, model="SIS", b=0.08, d=0.2, c=0.1,
                           diffusion="max" if method else None,
                           method=method, k=5 if method else 0,
                           runs=1, steps=150, seed=seed)
           trials.append(np.asarray(sim.run_single_sim()) / len(G))
       plt.plot(np.mean(trials, axis=0), label=method or "Unchanged network")
   plt.xlabel("Simulation steps")
   plt.ylabel("Infected fraction")
   plt.legend()
   plt.show()</code></pre></details><figure class="study-figure" id="spread-results"><a href="guide-results/epidemics-spread.svg" target="_blank"><img src="guide-results/epidemics-spread.svg" alt="SIS prevalence on unchanged karate network and with five extra random edges" loading="lazy"></a><figcaption>Twenty runs, seeds 0–19, b=0.08, d=0.2 and c=0.1. The same parameters are used with and without five added edges. Shading shows run-to-run variability; this small experiment does not establish a universally beneficial intervention. <a href="guide-results/epidemics-spread.csv">Run data (CSV)</a> · <a href="guide-results/epidemics-spread.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">The mean curves summarize the observed direction, while their overlapping run ranges show that individual outcomes vary substantially. A change in mean prevalence is not a universal guarantee for every seed or every network.</p><p>Mean step-150 prevalence is 32.2% without added links and 38.4% with five random added links. Wide run ranges caution against judging an intervention from a single realization.</p>

.. _epidemics-sir-studies:

.. raw:: html

   <span id="sir-studies"></span>

SIR studies
-----------

.. _epidemics-sir-history:

.. raw:: html

   <span id="sir-history"></span>

One SIR trajectory
~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="study-introduction" data-study="sir-results"><p>This small example uses the 34-node karate contact graph, three initially infected nodes, transmission probability 0.08, and recovery probability 0.2. There is no vaccination. Simulation seed 17 selects one realization, followed for 150 steps.</p><p>For SIR, the returned trajectory counts recovered nodes rather than currently infected nodes. The code therefore reads both fields from the step history, divides by 34, and obtains the susceptible fraction as one minus infected minus recovered. The three curves describe mutually exclusive states and sum to one.</p></div><details class="study-code" data-for-figure="sir-results"><summary>Run and read one SIR realization — code</summary><pre><code>import numpy as np
   import matplotlib.pyplot as plt
   from graph_tiger.graphs import karate
   from graph_tiger.diffusion import Diffusion

   G = karate()
   sim = Diffusion(G, model="SIR", b=0.08, d=0.2, c=0.1,
                   runs=1, steps=150, seed=17)
   sim.run_single_sim()
   infected = np.array([sim.sim_info[t]["failed"] for t in range(151)]) / len(G)
   recovered = np.array([sim.sim_info[t]["recovered"] for t in range(151)]) / len(G)
   plt.plot(1 - infected - recovered, label="Susceptible")
   plt.plot(infected, label="Infected")
   plt.plot(recovered, label="Recovered")
   plt.xlabel("Simulation steps")
   plt.ylabel("Fraction of original nodes")
   plt.legend()
   plt.show()</code></pre></details><figure class="study-figure" id="sir-results"><a href="guide-results/epidemics-sir.svg" target="_blank"><img src="guide-results/epidemics-sir.svg" alt="Susceptible infected and recovered fractions in one karate-club SIR realization" loading="lazy"></a><figcaption>A single realization, seed 17, on the 34-node karate-club graph. b=0.08, d=0.2, c=0.1; no vaccination. Susceptible, infected and recovered fractions sum to one at every step. The recovered curve accumulates past infections; it is not current prevalence. <a href="guide-results/epidemics-sir.csv">Run data (CSV)</a> · <a href="guide-results/epidemics-sir.json">Parameters and versions</a> · <a href="guide-results/run-guide-studies.py">Complete experiment script</a>. Select a plot to open it at full size.</figcaption></figure><p class="study-interpretation">In this realization the initially infected nodes recover without transmitting onward. That is a valid stochastic fadeout, not evidence that these parameters always prevent an outbreak. The repeated-run section below tests how representative this single outcome is.</p><p>This realization finishes with 8.8% recovered and 0.0% infected. The repeated-run study below places the single trajectory in its outbreak-size distribution.</p><p>With pre-vaccination, the recovered field includes vaccinated nodes. Subtract the initially vaccinated count when measuring cumulative recoveries caused by the simulated outbreak. If infection remains at the last step, the final recovered fraction is not yet the completed outbreak size.</p>

.. _epidemics-outbreak-distribution:

.. raw:: html

   <span id="outbreak-distribution"></span>

Repeated SIR outbreaks
~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="study-introduction" data-study="sir-ensemble-results"><p>Use the same 34-node karate topology and SIR probabilities as the single-run example, explicitly set all contact weights to one, and run seeds 0–99. Simulations continue for 300 steps so the recorded recovered fraction can be interpreted as completed outbreak size only after infection has disappeared.</p><p>The code collects two different outputs: one final recovered fraction per run for the histogram, and a full infected-fraction trajectory per run for the prevalence plot. Histogram height counts runs; the time-series line is mean current infection and its band is the 10th–90th percentile range. Only the first 80 time steps are displayed, but the CSV includes all 301 recorded states.</p></div><details class="study-code" data-for-figure="sir-ensemble-results"><summary>Run the repeated SIR experiment — code</summary><pre><code>import numpy as np
   import networkx as nx
   from graph_tiger.diffusion import Diffusion

   G = nx.karate_club_graph()
   nx.set_edge_attributes(G, 1.0, &quot;weight&quot;)
   outbreaks, infected_runs = [], []
   for seed in range(100):
       sim = Diffusion(G, model=&quot;SIR&quot;, b=0.08, d=0.2, c=0.1,
                       runs=1, steps=300, seed=seed)
       sim.run_single_sim()
       outbreaks.append(sim.sim_info[300][&quot;recovered&quot;] / len(G))
       infected_runs.append([sim.sim_info[t][&quot;failed&quot;] / len(G) for t in range(301)])</code></pre></details><figure class="study-figure" id="sir-ensemble-results"><a href="guide-results/sir-outbreak-distribution.svg" target="_blank"><img src="guide-results/sir-outbreak-distribution.svg" alt="Distribution of final recovered fractions across 100 SIR runs" loading="lazy"></a><a href="guide-results/sir-ensemble-prevalence.svg" target="_blank"><img src="guide-results/sir-ensemble-prevalence.svg" alt="Mean SIR prevalence and run-to-run interval" loading="lazy"></a><figcaption>Seeds 0–99; b=0.08, d=0.2, c=0.1. The prevalence plot shows the first 80 of 300 simulated steps. <a href="guide-results/sir-prevalence.csv">Full prevalence trajectories (CSV)</a> contain all 301 timesteps for each of the 100 seeds, sufficient to reconstruct the mean and percentile band. The data link below contains final outbreak sizes. <a href="guide-results/sir-ensemble.csv">Data (CSV)</a> · <a href="guide-results/sir-ensemble.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure><p class="study-interpretation">These panels answer complementary questions: how large an outbreak ultimately becomes, and when infection is present. A modest mean prevalence curve can coexist with large final outbreaks because infections occur at different times and recovered nodes accumulate.</p><p>5 of 100 runs produced no secondary infections. Mean final outbreak size was 53.1%. All runs had zero infected nodes at the last step. Reporting the distribution avoids confusing one fadeout with the typical outcome.</p><p>For the spectral effective-strength heuristic, use a spectral radius consistent with the contact model’s weights. <a href="references.html#ref-wang2003epidemic">Reference</a></p>

.. _epidemics-epidemic-snapshots:

.. raw:: html

   <span id="epidemic-snapshots"></span>

SIR network states
~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="study-introduction" data-study="epidemic-state-results"><p>This illustration uses a synthetic 60-node Barabási–Albert contact graph, with two links introduced per new node and graph seed 17. Initially six nodes are infected. Each infected–susceptible contact transmits with probability 0.15 per step, and infected nodes recover with probability 0.08. We inspect steps 0, 3, 10, and 30 of simulation seed 17.</p><p>The code stores one layout and obtains each state by rerunning that same seeded realization to the requested step. It colors each node from its state at that time: teal is susceptible, orange infected, and blue recovered. The plotting loop below uses the stored colors without recomputing positions. These are topological contact locations, not geographic locations.</p></div><details class="study-code" data-for-figure="epidemic-state-results"><summary>Generate and draw the SIR states — code</summary><pre><code>import networkx as nx
   from graph_tiger.diffusion import Diffusion

   G = nx.barabasi_albert_graph(60, 2, seed=17)
   pos = nx.spring_layout(G, seed=17, weight=None)
   states = {}
   counts = []
   for t in [0, 3, 10, 30]:
       sim = Diffusion(G, model=&quot;SIR&quot;, b=0.15, d=0.08, c=0.1,
                       runs=1, steps=t, seed=17)
       sim.run_single_sim()
       recovered = set(sim.vaccinated)
       infected = {n: n in sim.infected for n in G}
       counts.append(dict(step=t, infected=len(sim.infected), recovered=len(recovered)))
       states[t] = {n: &quot;#5765b0&quot; if n in recovered else
                    &quot;#cf6636&quot; if infected[n] else &quot;#157a79&quot; for n in G}

   legend = [("#157a79", "Susceptible"), ("#cf6636", "Infected"), ("#5765b0", "Recovered")]
   import matplotlib.pyplot as plt
   from matplotlib.lines import Line2D

   for t, colors in states.items():
       fig, ax = plt.subplots(figsize=(8, 6.5))
       nx.draw(G, pos, ax=ax, node_color=[colors[n] for n in G],
               node_size=140, edge_color="#b6c3c8", width=1.2, with_labels=False)
       ax.set_title(f"State at step {t}")
       ax.set_aspect("equal")
       ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=color,
                                 label=label) for color, label in legend])
       plt.show()</code></pre></details><figure class="study-figure" id="epidemic-state-results"><a href="guide-results/sir-state-0.svg" target="_blank"><img src="guide-results/sir-state-0.svg" alt="SIR node states at step 0" loading="lazy"></a><a href="guide-results/sir-state-3.svg" target="_blank"><img src="guide-results/sir-state-3.svg" alt="SIR node states at step 3" loading="lazy"></a><a href="guide-results/sir-state-10.svg" target="_blank"><img src="guide-results/sir-state-10.svg" alt="SIR node states at step 10" loading="lazy"></a><a href="guide-results/sir-state-30.svg" target="_blank"><img src="guide-results/sir-state-30.svg" alt="SIR node states at step 30" loading="lazy"></a><figcaption>Graph seed 17, simulation seed 17; b=0.15, d=0.08, c=0.1. This is a separate illustration, not the karate outbreak-size study. <a href="guide-results/sir-snapshots.csv">Data (CSV)</a> · <a href="guide-results/sir-snapshots.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure><p class="study-interpretation">At the four displayed steps, infected counts are 6, 21, 29, 3 and recovered counts are 0, 3, 21, 51. Orange nodes show current infection, while blue nodes record past infections that have recovered. A decline in orange therefore need not mean a small outbreak: many nodes may already be blue. This illustration uses a different graph and parameters from the karate-club outbreak-size experiment.</p><p><a href="visualization.html#plotting-options">Shared plotting and export options</a>.</p>
