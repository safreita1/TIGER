Information diffusion
=====================

.. raw:: html

   <p class="lead">Use these models when a node represents an adopter, opinion, message, or decision rather than an infected individual. The model determines whether activation is permanent, whether several neighbors reinforce one another, whether states can reverse, and how competing messages resolve a simultaneous arrival.</p><div class="table-scroll"><table><thead><tr><th>Model</th><th>Node change</th><th>Update schedule</th><th>Primary result</th></tr></thead><tbody><tr><th>Independent cascade</th><td>An active node gets one probabilistic attempt to activate each inactive target.</td><td>Synchronous rounds</td><td>Total active nodes</td></tr><tr><th>Linear threshold</th><td>A node activates when accumulated incoming influence reaches its threshold.</td><td>Synchronous rounds</td><td>Total active nodes</td></tr><tr><th>Voter</th><td>One selected node copies one selected neighbor, so either state can replace the other.</td><td>One asynchronous node update per step</td><td>Nodes in the tracked state</td></tr><tr><th>Competitive cascade</th><td>The first message to activate a node is permanent; a declared rule resolves same-round collisions.</td><td>Synchronous rounds</td><td>Nodes holding the tracked message</td></tr></tbody></table></div><p>The four models share the <code>Influence</code> class, but their parameters are not interchangeable. Directed independent, threshold, and competitive cascades move along outgoing edges. The voter implementation currently requires an undirected graph.</p>

.. _influence-independent-cascade:

.. raw:: html

   <span id="independent-cascade"></span>

Independent cascade
-------------------

.. raw:: html

   <p>Choose this model when each newly active source has one opportunity to transmit through each eligible edge. All attempts in a round read the same starting state. Successful targets form the next frontier and may attempt transmission in the following round; a failed edge is not retried by that source. The model convention follows <a href="references.html#ref-kempe2003maximizing">Kempe, Kleinberg, and Tardos</a>.</p><pre><code>import networkx as nx
   from graph_tiger.influence import Influence

   G = nx.DiGraph([("A", "B"), ("B", "C"), ("B", "D")])
   sim = Influence(G, model="independent_cascade", seeds={"A"},
                   probability=0.25, runs=100, steps=10, seed=17)
   mean_active = sim.run_simulation()</code></pre><p><code>probability</code> may be one number for every edge or the name of an edge attribute. With probability 1 in the figure, B activates in round 1 and only then attempts C and D in round 2. This deliberately deterministic setting exposes the update order; realistic probabilities require repeated runs.</p><figure class="study-figure"><a href="guide-results/influence-independent-cascade.svg" target="_blank"><img src="guide-results/influence-independent-cascade.svg" alt="Three network states showing an independent cascade moving one directed edge per round." loading="lazy"></a><figcaption>Orange nodes are active. Newly active nodes do not transmit until the next round. <a href="guide-results/influence.csv">Data (CSV)</a> · <a href="guide-results/influence.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure>

.. _influence-linear-threshold:

.. raw:: html

   <span id="linear-threshold"></span>

Linear threshold
----------------

.. raw:: html

   <p>Choose this model when several active neighbors can jointly produce adoption. Each active predecessor contributes its edge weight once. An inactive node becomes active when the accumulated incoming weight is at least its threshold, and activation is permanent. This is the weighted threshold convention described by <a href="references.html#ref-kempe2003maximizing">Kempe, Kleinberg, and Tardos</a>.</p><pre><code>import networkx as nx
   from graph_tiger.influence import Influence

   G = nx.DiGraph()
   G.add_weighted_edges_from([("A", "C", 0.35),
                              ("B", "C", 0.35),
                              ("C", "D", 1.0)])
   thresholds = {"A": 1.0, "B": 1.0, "C": 0.60, "D": 0.80}
   sim = Influence(G, model="linear_threshold", seeds={"A", "B"},
                   threshold=thresholds, runs=1, steps=2, seed=17)
   active = sim.run_single_sim()</code></pre><p>The two incoming weights at C sum to 0.70, exceeding its 0.60 threshold; neither 0.35 contribution would activate C alone. C therefore activates in round 1 and its weight of 1.0 activates D in round 2. <code>threshold</code> may be a scalar, a node-to-value mapping, or a node-attribute name. <code>weight</code> names the edge attribute and missing values default to 1.</p><figure class="study-figure"><a href="guide-results/influence-linear-threshold.svg" target="_blank"><img src="guide-results/influence-linear-threshold.svg" alt="Three weighted directed network states showing threshold activation by combined influence." loading="lazy"></a><figcaption>Edge labels show influence weights; the annotation below C shows its activation threshold. <a href="guide-results/influence.csv">Data (CSV)</a> · <a href="guide-results/influence.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure>

.. _influence-voter-model:

.. raw:: html

   <span id="voter-model"></span>

Voter model
-----------

.. raw:: html

   <p>Choose the voter model when opinions or labels can reverse. At each step, TIGER samples one node and then one of its neighbors uniformly; the selected node copies that neighbor’s current state immediately. One step is one node-update event, not a synchronous sweep over the graph. The process follows the local copying model of <a href="references.html#ref-clifford1973model">Clifford and Sudbury</a>.</p><pre><code>import networkx as nx
   from graph_tiger.influence import Influence

   G = nx.cycle_graph(20)
   initial = {node: int(node &gt;= 10) for node in G}
   sim = Influence(G, model="voter", initial_state=initial,
                   tracked_state=1, runs=40, steps=100, seed=17)
   mean_state_1 = sim.run_simulation()</code></pre><p>The example begins with equal-sized contiguous regions. Individual runs can move toward either consensus state, so the ensemble mean alone is incomplete. The network panels show one realization; the curve reports the mean and 10th–90th percentile interval over 40 independent seeds. Isolated nodes retain their state.</p><figure class="study-figure"><a href="guide-results/influence-voter.svg" target="_blank"><img src="guide-results/influence-voter.svg" alt="Three voter-model network states and an ensemble trajectory for the fraction of nodes in state one." loading="lazy"></a><figcaption>Orange and gray mark the two states. The wide band reflects stochastic local copying, not measurement error. <a href="guide-results/influence.csv">Data (CSV)</a> · <a href="guide-results/influence.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure>

.. _influence-competitive-cascade:

.. raw:: html

   <span id="competitive-cascade"></span>

Competitive cascades
--------------------

.. raw:: html

   <p>Use a competitive cascade when several messages spread at once and adoption is permanent. Each message has its own frontier. A previously inactive node first collects every successful same-round proposal; it then adopts one message using either a random tie break or the supplied priority order. Resolving proposals after the round prevents node-iteration order from deciding the winner. The implementation uses the competitive diffusion setting introduced by <a href="references.html#ref-bharathi2007competitive">Bharathi, Kempe, and Salek</a> with this explicit collision rule.</p><pre><code>import networkx as nx
   from graph_tiger.influence import Influence

   G = nx.path_graph(9)
   sim = Influence(
       G, model="competitive_cascade",
       message_seeds={"claim": {0}, "correction": {8}},
       probability=1.0, tracked_state="correction",
       tie_break="priority", priority=["correction", "claim"],
       runs=1, steps=4, seed=17)
   correction_reach = sim.run_single_sim()</code></pre><p>Both messages advance one edge per round. At round 4 they reach node 4 simultaneously, so the declared priority assigns that node to the correction. A random tie rule samples among the messages that actually reached the node. <code>probability</code> may also map each message to its own scalar or edge-attribute name.</p><figure class="study-figure"><a href="guide-results/influence-competitive-cascade.svg" target="_blank"><img src="guide-results/influence-competitive-cascade.svg" alt="Three path-network states showing two competitive messages meeting at the center." loading="lazy"></a><figcaption>Red is the claim, blue is the correction, and gray is inactive. The priority rule—not drawing order—settles the center node. <a href="guide-results/influence.csv">Data (CSV)</a> · <a href="guide-results/influence.json">Parameters</a> · <a href="reproducibility.html">Rerun this study</a>.</figcaption></figure>

.. _influence-outputs:

.. raw:: html

   <span id="outputs"></span>

Runs, history, and stopping
---------------------------

.. raw:: html

   <p><code>run_single_sim()</code> returns <code>steps + 1</code> values and leaves that realization in <code>sim_info</code>. <code>run_simulation()</code> averages the same primary trajectory over <code>runs</code>. Progressive models return active-node counts; voter and competitive models return the count for <code>tracked_state</code>. Divide by the graph order when a fraction is required.</p><p>For each step, <code>sim_info[t]["status"]</code> follows the graph’s node iteration order, <code>counts</code> reports all states, and <code>changed</code> identifies nodes changed in that update. Progressive and competitive cascades stop changing when no frontier remains; the fixed-length return carries the terminal value forward. The voter model stops changing only at an absorbing state, such as consensus within every connected component. A finite step limit can end first, so report both the limit and the final state.</p><p>These simulations evaluate diffusion from user-supplied starting nodes. They do not solve the separate optimization problem of choosing a seed set to maximize expected reach.</p>
