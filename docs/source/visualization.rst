Visualization
=============

.. raw:: html

   <span id="plotting-options"></span><p>Use network drawings to locate changes and result plots to measure their consequences. This guide covers layouts, node colors, edge styles, snapshots, and animation; complete examples and figures remain with the <a href="attacks.html#road-snapshots">attack</a>, <a href="defenses.html#defense-snapshots">defense</a>, <a href="epidemics.html#epidemic-snapshots">epidemic</a>, and <a href="cascades.html#cascade-snapshots">cascade</a> studies.</p>

.. _visualization-visualization-layouts:

.. raw:: html

   <span id="visualization-layouts"></span>

Layouts and node states
-----------------------

.. raw:: html

   <p>Network-state figures now accompany their attack, defense, epidemic, or cascade experiment. Across all of them, compute positions once and reuse them across states; use physical coordinates only when supplied by the dataset. A spring layout shows connectivity, not geographic distance. Keep colors consistent and distinguish failed components from functioning links.</p><p>Map colors to explicit states, such as susceptible, infected, and recovered, and use the same mapping at every step. Highlight added edges with a distinct line style; if failed nodes or original edges remain as context, make that clear in the legend. Avoid interpreting a layout’s apparent clusters or edge lengths as measured physical properties.</p>

.. _visualization-visualization-controls:

.. raw:: html

   <span id="visualization-controls"></span>

TIGER plotting options
----------------------

.. raw:: html

   <div class="table-scroll"><table><thead><tr><th>Option</th><th>Behavior</th></tr></thead><tbody><tr><td><code>plot_transition</code></td><td>Saves selected network snapshots from the first simulation run.</td></tr><tr><td><code>node_style=<wbr>"force_atlas"</code></td><td>Uses the optional ForceAtlas2 layout; <code>fa_iter</code> controls the number of layout iterations.</td></tr><tr><td><code>edge_style=<wbr>"bundled"</code></td><td>Bundles nearby edges when the optional visualization dependencies are installed.</td></tr><tr><td><code>gif_animation</code></td><td>Writes an MP4 through FFmpeg on supported non-Windows platforms.</td></tr><tr><td><code>gif_snaps</code></td><td>Saves individual animation frames while the animation workflow runs.</td></tr></tbody></table></div>

.. _visualization-visualization-export:

.. raw:: html

   <span id="visualization-export"></span>

Result plots and export
-----------------------

.. raw:: html

   <p>TIGER saves built-in plots under <code>plots/</code> in the current working directory. The worked examples use Matplotlib for explicit axes, legends, and run-to-run bands. With Matplotlib, call <code>fig.savefig("result.svg", bbox_inches="tight")</code> before displaying the figure for scalable output, or use a <code>.png</code> filename with <code>dpi=180</code>. A simulation trajectory includes <code>steps + 1</code> states; label whether state zero is intact or already attacked, and state the normalization rather than relying on plotting defaults.</p>

.. _visualization-visualization-compare:

.. raw:: html

   <span id="visualization-compare"></span>

Compare the visualization options
---------------------------------

.. raw:: html

   <p>Every drawing below uses the same unweighted karate graph: 34 nodes and 78 edges. The layout and edge-style comparisons change only the drawing, not the network or a robustness measure. Orange node 0 and purple node 33 are landmarks for following the same nodes between layouts; they are not failed or defended states. Colors acquire SIR meanings only in the state examples below.</p>

.. _visualization-visualization-layout-comparison:

.. raw:: html

   <span id="visualization-layout-comparison"></span>

Node positions and layout iterations
------------------------------------

.. raw:: html

   <p>Without complete node-position attributes, <code>node_style=None</code> uses a spectral layout. <code>node_style="force_atlas"</code> uses ForceAtlas2, and <code>fa_iter</code> sets its iteration count. Complete <code>pos</code> attributes take precedence over either setting: the circular example supplies positions deliberately and is not a map.</p><div class="vis-grid"><figure><a href="guide-results/vis-layout-spectral.svg" target="_blank"><img src="guide-results/vis-layout-spectral.svg" alt="Default spectral layout; no supplied positions" loading="lazy"></a><figcaption>Default spectral layout; no supplied positions · <a href="guide-results/vis-layout-spectral.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-layout-provided.svg" target="_blank"><img src="guide-results/vis-layout-provided.svg" alt="Supplied circular positions; retained even with ForceAtlas2 requested" loading="lazy"></a><figcaption>Supplied circular positions; retained even with ForceAtlas2 requested · <a href="guide-results/vis-layout-provided.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-layout-force-20.svg" target="_blank"><img src="guide-results/vis-layout-force-20.svg" alt="ForceAtlas2 after 20 iterations" loading="lazy"></a><figcaption>ForceAtlas2 after 20 iterations · <a href="guide-results/vis-layout-force-20.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-layout-force-200.svg" target="_blank"><img src="guide-results/vis-layout-force-200.svg" alt="ForceAtlas2 after 200 iterations" loading="lazy"></a><figcaption>ForceAtlas2 after 200 iterations · <a href="guide-results/vis-layout-force-200.pdf">PDF</a></figcaption></figure></div><p>Compare the separation of nodes and the ease of tracing connections—not the apparent geographic shape. Spectral positions can place some nodes very close together. More ForceAtlas2 iterations change the layout but do not add links or demonstrate greater robustness. The iteration comparison fixes the initialization with <code>fa2</code> seed 17; TIGER does not pass its simulation seed to the layout engine. Colors, line widths, and titles use common documentation styling rather than TIGER’s default plot formatting.</p><details class="study-code" data-gallery-example="layouts"><summary>Try the node-layout options — code</summary><pre><code>from pathlib import Path
   import matplotlib.pyplot as plt
   import networkx as nx
   from graph_tiger.graphs import graph_loader
   from graph_tiger.attacks import Attack

   G = graph_loader(&quot;karate&quot;)
   nx.set_edge_attributes(G, 1.0, &quot;weight&quot;)
   for style, iterations in [(None, 200), (&quot;force_atlas&quot;, 20), (&quot;force_atlas&quot;, 200)]:
       sim = Attack(G.copy(), attack=&quot;rnd_node&quot;, runs=1, steps=3, seed=17,
                    node_style=style, fa_iter=iterations, plot_transition=True)
       sim.save_dir = str(Path(&quot;plots&quot;) / f&quot;layout-{style or 'spectral'}-{iterations}&quot;)
       Path(sim.save_dir).mkdir(parents=True, exist_ok=True)
       sim.run_simulation()
       plt.close(&quot;all&quot;)</code></pre><p>Each setting saves into its own directory under <code>plots/</code>, retaining all three comparisons. Rerunning a setting replaces its own outputs. The downloadable gallery runner fixes ForceAtlas2 initialization explicitly and saves the coordinates used here.</p></details>

.. _visualization-visualization-edge-comparison:

.. raw:: html

   <span id="visualization-edge-comparison"></span>

Straight versus bundled edges
-----------------------------

.. raw:: html

   <p>Both views below use the same 200-iteration ForceAtlas2 positions and identical axis limits. <code>edge_style=None</code> draws straight links; <code>edge_style="bundled"</code> uses Datashader’s bundling routine to route nearby lines together.</p><div class="vis-grid"><figure><a href="guide-results/vis-edges-straight.svg" target="_blank"><img src="guide-results/vis-edges-straight.svg" alt="Straight links: easier to follow individual edges" loading="lazy"></a><figcaption>Straight links: easier to follow individual edges · <a href="guide-results/vis-edges-straight.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-edges-bundled.svg" target="_blank"><img src="guide-results/vis-edges-bundled.svg" alt="Bundled links: shared visual corridors, unchanged topology" loading="lazy"></a><figcaption>Bundled links: shared visual corridors, unchanged topology · <a href="guide-results/vis-edges-bundled.pdf">PDF</a></figcaption></figure></div><p>Bundling can reduce a dense tangle of lines, but can also make individual links harder to count. Curves joining visually are not new graph junctions: no nodes or edges were added. ForceAtlas2 and bundling require the optional visualization dependencies.</p><details class="study-code" data-gallery-example="edges"><summary>Hold positions fixed and change edge style — code</summary><pre><code>from pathlib import Path
   import matplotlib.pyplot as plt
   import networkx as nx
   from graph_tiger.graphs import graph_loader
   from graph_tiger.attacks import Attack

   G = graph_loader(&quot;karate&quot;)
   nx.set_edge_attributes(G, 1.0, &quot;weight&quot;)
   pos = nx.spring_layout(G, seed=17, weight=None)
   nx.set_node_attributes(G, pos, &quot;pos&quot;)
   for edge_style in [None, &quot;bundled&quot;]:
       sim = Attack(G.copy(), attack=&quot;rnd_node&quot;, runs=1, steps=3, seed=17,
                    edge_style=edge_style, plot_transition=True)
       sim.save_dir = str(Path(&quot;plots&quot;) / f&quot;edges-{edge_style or 'straight'}&quot;)
       Path(sim.save_dir).mkdir(parents=True, exist_ok=True)
       sim.run_simulation()
       plt.close(&quot;all&quot;)</code></pre><p>This compact example supplies spring positions; the figures use saved ForceAtlas2 positions. In either case, positions stay fixed and the two settings save into separate directories. Bundled edges currently require consecutive integer node labels aligned with graph iteration order. Use straight edges when the graph has arbitrary or reordered labels.</p></details>

.. _visualization-visualization-transition-comparison:

.. raw:: html

   <span id="visualization-transition-comparison"></span>

Selected transition snapshots
-----------------------------

.. raw:: html

   <p><code>plot_transition=True</code> saves selected states from the first run; it does not change the layout or dynamics. Here a SIR simulation uses b=0.2, d=0.08, c=0.15, seed 17, and 30 steps on the same karate graph. TIGER’s transition selector chooses steps 0, 1, 2, 25, 30. Its middle snapshot is chosen from the change in infected count, not necessarily halfway through elapsed time.</p><div class="vis-grid"><figure><a href="guide-results/vis-state-0.svg" target="_blank"><img src="guide-results/vis-state-0.svg" alt="Transition snapshot at step 0" loading="lazy"></a><figcaption>Transition snapshot at step 0 · <a href="guide-results/vis-state-0.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-state-1.svg" target="_blank"><img src="guide-results/vis-state-1.svg" alt="Transition snapshot at step 1" loading="lazy"></a><figcaption>Transition snapshot at step 1 · <a href="guide-results/vis-state-1.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-state-2.svg" target="_blank"><img src="guide-results/vis-state-2.svg" alt="Transition snapshot at step 2" loading="lazy"></a><figcaption>Transition snapshot at step 2 · <a href="guide-results/vis-state-2.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-state-25.svg" target="_blank"><img src="guide-results/vis-state-25.svg" alt="Transition snapshot at step 25" loading="lazy"></a><figcaption>Transition snapshot at step 25 · <a href="guide-results/vis-state-25.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-state-30.svg" target="_blank"><img src="guide-results/vis-state-30.svg" alt="Transition snapshot at step 30" loading="lazy"></a><figcaption>Transition snapshot at step 30 · <a href="guide-results/vis-state-30.pdf">PDF</a></figcaption></figure></div><p>Teal is susceptible, orange infected, and purple recovered. Positions stay fixed, while colors track the process. Fewer orange nodes late in the run can reflect recovery rather than a small total outbreak. The images use TIGER’s recorded states and transition selection with a consistent documentation palette.</p><details class="study-code" data-gallery-example="transition"><summary>Save transition snapshots — code</summary><p>The example reruns the same seeded realization to each requested horizon, so every snapshot reads the state at its own time step while preserving a consistent layout.</p><pre><code>from pathlib import Path
   import matplotlib.pyplot as plt
   import networkx as nx
   from graph_tiger.graphs import graph_loader
   from graph_tiger.diffusion import Diffusion

   G = graph_loader(&quot;karate&quot;)
   nx.set_edge_attributes(G, 1.0, &quot;weight&quot;)
   sim = Diffusion(G, model=&quot;SIR&quot;, b=0.2, d=0.08, c=0.15,
                   seed=17, runs=1, steps=30, plot_transition=True)

   track = sim.track_simulation
   def record_snapshot(step):
       track(step)
       sim.sim_info[step][&quot;protected&quot;] = sim.sim_info[step][&quot;protected&quot;].copy()
   sim.track_simulation = record_snapshot
   record_snapshot(0)

   sim.save_dir = str(Path(&quot;plots&quot;) / &quot;sir-transitions&quot;)
   Path(sim.save_dir).mkdir(parents=True, exist_ok=True)
   sim.run_simulation()
   plt.close(&quot;all&quot;)
   print(sim.save_dir)</code></pre></details>

.. _visualization-visualization-animation-comparison:

.. raw:: html

   <span id="visualization-animation-comparison"></span>

Animation and individual frames
-------------------------------

.. raw:: html

   <p><code>gif_animation=True</code> requests TIGER’s MP4 export on supported platforms with FFmpeg installed. Despite the option name, that exporter creates an MP4, not a GIF. Adding <code>gif_snaps=True</code> also saves its individual animation frames as PDFs; it does not create another layout. For diffusion, the exporter uses every tenth step plus the final step, so these frame times differ from the transition snapshots above.</p><div class="vis-grid"><figure><a href="guide-results/vis-state-0.svg" target="_blank"><img src="guide-results/vis-state-0.svg" alt="Animation-frame state at step 0" loading="lazy"></a><figcaption>Animation-frame state at step 0 · <a href="guide-results/vis-state-0.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-state-10.svg" target="_blank"><img src="guide-results/vis-state-10.svg" alt="Animation-frame state at step 10" loading="lazy"></a><figcaption>Animation-frame state at step 10 · <a href="guide-results/vis-state-10.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-state-20.svg" target="_blank"><img src="guide-results/vis-state-20.svg" alt="Animation-frame state at step 20" loading="lazy"></a><figcaption>Animation-frame state at step 20 · <a href="guide-results/vis-state-20.pdf">PDF</a></figcaption></figure><figure><a href="guide-results/vis-state-30.svg" target="_blank"><img src="guide-results/vis-state-30.svg" alt="Animation-frame state at step 30" loading="lazy"></a><figcaption>Animation-frame state at step 30 · <a href="guide-results/vis-state-30.pdf">PDF</a></figcaption></figure></div><details class="study-code"><summary>Play a portable preview of these four states</summary><p>This Pillow-generated GIF uses the same recorded states, slowed to 800 ms per frame for inspection. It is a Windows-compatible preview, not output from TIGER’s native MP4 path. Close this panel to hide playback.</p><img class="vis-playback" src="guide-results/vis-playback.gif" alt="Loop through SIR states at steps 0, 10, 20, and 30" loading="lazy"></details><p>Native animation export is unavailable on Windows, and <code>gif_snaps</code> does not export frames there. Static transition plots remain available. Use the gallery runner for the animated GIF shown here, or run native MP4 export on a supported non-Windows platform.</p><details class="study-code" data-gallery-example="animation"><summary>Native animation and frame export — supported platforms only</summary><p>This example requests the native MP4 and individual frame outputs produced by TIGER.</p><pre><code>from pathlib import Path
   import matplotlib.pyplot as plt
   import networkx as nx
   from graph_tiger.graphs import graph_loader
   from graph_tiger.diffusion import Diffusion

   G = graph_loader(&quot;karate&quot;)
   nx.set_edge_attributes(G, 1.0, &quot;weight&quot;)
   sim = Diffusion(G, model=&quot;SIR&quot;, b=0.2, d=0.08, c=0.15,
                   seed=17, runs=1, steps=30, gif_animation=True, gif_snaps=True)

   track = sim.track_simulation
   def record_snapshot(step):
       track(step)
       sim.sim_info[step][&quot;protected&quot;] = sim.sim_info[step][&quot;protected&quot;].copy()
   sim.track_simulation = record_snapshot
   record_snapshot(0)

   sim.save_dir = str(Path(&quot;plots&quot;) / &quot;sir-animation&quot;)
   Path(sim.save_dir).mkdir(parents=True, exist_ok=True)
   sim.run_simulation()
   plt.close(&quot;all&quot;)
   print(sim.save_dir)</code></pre></details>

.. _visualization-visualization-gallery-files:

.. raw:: html

   <span id="visualization-gallery-files"></span>

Reproduce these comparisons
---------------------------

.. raw:: html

   <p>Install <code>graph-tiger[visualization]==0.7.0</code>, then run <a href="guide-results/run-visualization-gallery.py">run-visualization-gallery.py</a>. Each SIR snapshot is rerun to its own horizon from the same seed, and the runner checks rendered color counts against the scalar trajectory. The <a href="guide-results/vis-gallery.json">gallery settings</a> record dependency versions, graph and simulation settings, exact layout coordinates, selected frame times, and file hashes. The <a href="guide-results/vis-states.csv">state-count CSV</a> lets you compare snapshot colors with simulated counts.</p><script>(function(){const routes={"road-snapshots":"attacks.html#road-snapshots","network-results":"attacks.html#network-results","defense-snapshots":"defenses.html#defense-snapshots","defense-network-results":"defenses.html#defense-network-results","epidemic-snapshots":"epidemics.html#epidemic-snapshots","epidemic-state-results":"epidemics.html#epidemic-state-results","cascade-snapshots":"cascades.html#cascade-snapshots","cascade-state-results":"cascades.html#cascade-state-results","section-1":"visualization.html#plotting-options"};function redirect(){const target=routes[decodeURIComponent(location.hash.slice(1))];if(target)location.replace(target);}addEventListener("hashchange",redirect);redirect();})();</script>
