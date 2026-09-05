# TIGER Correctness Remediation Plan

**Working branch:** `codex/correctness-remediation`  
**Baseline:** `8ddb87bda10168cea1a84a61c12b692383e667c9`  
**Scope:** algorithms, simulations, measures, graph utilities, packaging, visualizations, examples, documentation, and tests.

## Goals

1. Make every public implementation agree with its name, documentation, and cited literature.
2. Preserve the repository's current code and test style. Do not perform broad formatting, renaming, or refactoring.
3. Convert each audited defect into a small deterministic regression test before changing production code.
4. Keep public APIs compatible where the existing behavior is defensible; document and test intentional corrections where compatibility would preserve a wrong result.
5. Separate mathematically different models instead of presenting a TIGER-specific approximation as a published model.

## Working rules

- Fix one coherent issue cluster per commit.
- Read the surrounding implementation and tests before editing.
- Keep the existing test pattern: top-level `test_*` functions, ordinary parameter dictionaries, loops, direct `assert` statements, and a `main()` runner.
- Do not introduce a formatter, wholesale import sorting, pytest fixtures, parametrization, or a new test dependency.
- Prefer tiny graphs with analytically known answers over large golden-output fixtures.
- Stochastic tests must use fixed seeds and deterministic invariants; do not use fragile statistical thresholds.
- Do not mutate a caller-owned graph unless the public API explicitly says that it does.
- Every corrected formula must include a source citation in its docstring or adjacent documentation.
- The setup commit intentionally adds expected-behavior tests before production fixes. It is allowed to be red on the remediation branch; it must be green before merge.

## Scientific contracts

### SIS and SIR diffusion

TIGER's implementation is a synchronous, discrete-time stochastic network model. For every infected-susceptible edge, transmission occurs independently with probability `b` during a step. Every node infected at the start of the step recovers independently with probability `d`. In SIS, recovered nodes become susceptible; in SIR, they move permanently to the recovered set. Newly infected nodes do not transmit or recover until the next step.

The core transition currently follows this contract and should be preserved. Corrections are needed around validation, seeding, initialization, timeline semantics, and documentation. Parameters must satisfy `0 <= b <= 1`, `0 <= d <= 1`, `0 <= c <= 1`, `runs > 0`, and `steps >= 0`; `model` must be `SIS` or `SIR`. The documentation should call this a discrete-time network SIS/SIR process rather than imply that it directly integrates the Kermack-McKendrick differential equations.

### Cascading failure

The current implementation mixes betweenness-derived capacity with sampled loads and repeated local redistribution. That is neither the cited Crucitti model nor a standard Motter-Lai overload cascade.

The recommended corrected contract is a Motter-Lai-style node-overload cascade because it is closest to the existing public parameters and betweenness calculation:

1. Compute initial node load `L_i(0)` from shortest-path traffic on the intact graph.
2. Set capacity once as `C_i = (1 + r) L_i(0)`.
3. Remove attacked nodes from the functional graph.
4. Recompute loads on the surviving topology.
5. Fail every node whose recomputed load exceeds capacity, using a synchronous state update.
6. Repeat until no new failures occur or the requested step limit is reached.

Update the citation and documentation to Motter and Lai. If an actual Crucitti efficiency-dynamics model is desired, implement it as a separately named model with its own parameters and reference tests. Do not label the current local redistribution rule as Crucitti. For one compatibility cycle, expose the old rule only under an explicit `legacy_redistribution` option, correct its conservation defects, and document that it is TIGER-specific. The default remains the literature-aligned overload model.

### Attack and defense

`k_a` is the number of attacked nodes or edges and `k_d` is the defense budget. A node defense protects the intersection with the attack set, so removed nodes are `attacked - protected`. An edge attack removes edges. An edge-rewiring defense applies both its removal and addition to the simulation copy. The caller's graph remains unchanged.

### Measures

Measure names use standard graph-theoretic definitions. Exact defaults must be exact on supported graph sizes. Approximation parameters may reduce accuracy only when the user explicitly requests an approximation and the normalization must still use the graph order where the definition requires it.

## Issue ledger and implementation order

| ID | Severity | Location | Required correction | Verification |
| --- | --- | --- | --- | --- |
| DEF-01 | Critical | `graph_tiger/defenses.py:Defense.reset_simulation` | Remove `attacked - protected`, not `protected - attacked`. | Protected attacked nodes remain; unprotected attacked nodes are removed. |
| DEF-02 | Critical | `Defense.reset_simulation` | Use `remove_edges_from` for edge attacks. | Selected attack edges are absent from `graph_`. |
| DEF-03 | Critical | `Defense.run_single_sim` | Apply every rewiring removal to `graph_`; index the `removed` list correctly. | Added edges exist, removed edges do not, input graph is unchanged. |
| DEF-04 | Critical | `Defense.reset_simulation` | Generate defenses with `k_d`, not `steps`. | Protection count equals the requested budget. |
| CAS-01 | Critical | `graph_tiger/cascading.py` | Replace or explicitly rename the non-literature hybrid model using the scientific contract above. | Hand-computed path, star, and bridge cases match reference transitions. |
| CAS-02 | Critical | `Cascading.run_single_sim` | In `legacy_redistribution`, never redistribute the same failed load repeatedly. | A failed node contributes load at most once. |
| CAS-03 | Critical | `Cascading.run_single_sim` | In `legacy_redistribution`, divide among functioning recipients only and conserve load. | Total transferred load equals source load when recipients exist. |
| CAS-04 | Critical | `Cascading.reset_simulation` | Keep failed nodes and failed edges in separate state; never use edge tuples as node IDs. | Edge attacks complete without `KeyError` and without corrupting node state. |
| CAS-05 | Critical | `Cascading.reset_simulation` | An attacked node must be removed/failed regardless of its sampled pre-attack load. | Low-load attacked nodes still initiate the defined cascade transition. |
| CAS-06 | High | `Cascading` | Use a simulation-owned graph copy and restore it on every run. | Constructor and repeated runs leave the caller graph unchanged. |
| CAS-07 | High | `Cascading` timeline | Define `t=0` and return exactly `steps + 1` states, or return exactly `steps` transitions consistently across the framework. | Result length and plotted state count are stable and documented. |
| CAS-08 | Critical | `graph_tiger/cascading.py` | Add the separately named Crucitti weighted efficiency-dynamics model. | Analytical congestion, recovery, weighted-routing, and efficiency fixtures match the paper. |
| RNG-01 | High | `graph_tiger/simulations.py:Simulation.__init__` | Apply child `seed` before any random initialization. | Different seeds yield different initial states; equal seeds reproduce them. |
| RNG-02 | High | simulation dispatch | Do not reseed global generators inside every helper call or run. Prefer simulation-local Python and NumPy generators. | Runs in one ensemble are independent but reproducible as a sequence. |
| IMP-01 | High | `graph_tiger/simulations.py` | Lazy-load ForceAtlas2 and Datashader only when requested. | Core attacks, measures, and simulations import without optional visualization packages. |
| MEA-01 | High | `graph_tiger/utils.py:get_laplacian_spectrum` | Do not request `N-1` smallest eigenvalues when an exact full spectrum is required. Use a dense exact path or mathematically valid specialized formula. | `P_100` has one spanning tree and resistance `166650`. |
| MEA-02 | High | `graph_tiger/measures.py:natural_connectivity` | Normalize the exponential eigenvalue sum by `N`, not the number of computed eigenvalues. | Complete-graph result matches the closed form, including explicit approximations. |
| MEA-03 | High | `avg_vertex_betweenness` | Use the standard endpoint-excluding definition unless the API explicitly exposes an endpoint variant. | The average for `P_4` is `1`. |
| MEA-04 | Medium | measure functions | Give direct calls safe defaults instead of requiring `kwargs['use_gpu']`. | Direct calls work without wrapper-only keyword arguments. |
| MEA-05 | Medium | `largest_connected_component` | Reconcile count versus documented fraction and handle the empty graph. | Empty, disconnected, and connected fixtures have explicit results. |
| MEA-06 | Medium | `run_measure` | Validate measure names and avoid swallowing arbitrary programming errors. | Unknown names produce a clear exception; timeouts remain distinguishable. |
| MEA-07 | Medium | module scope | Remove global `np.seterr` mutation and avoid premature rounding in computational kernels. | Import does not alter caller numerical policy; precision tests pass. |
| ATK-01 | High | `graph_tiger/attacks.py:get_node_ns` | Map sparse-matrix indices back to original node labels and validate `k`. | Labeled paths return labels, not integer positions. |
| EDG-01 | High | `add_edge_rnd` | Precompute available nonedges and reject or cap impossible `k`; remove the unbounded loop. | Complete graphs terminate immediately with defined behavior. |
| EDG-02 | High | `add_edge_pref` | Add only nonedges, never self-loops, and return exactly the feasible requested count. | Every reported edge is valid on a path fixture. |
| EDG-03 | High | `rewire_edge_pref` | Validate the endpoints of the edge actually added. | Star rewiring cannot create a self-loop. |
| EDG-04 | High | `rewire_edge_pref_rnd` | Remove selected edges from the working graph and constrain replacement edges. | Edge count is conserved and replacements are valid nonedges. |
| EDG-05 | Medium | all edge defenses | Define behavior when fewer than `k` valid operations exist. | Boundary cases return a clear exception or documented capped result. |
| GRA-01 | High | `graph_tiger/graphs.py:get_graph_options` | Serialize dataset names, not function objects. | Output is valid JSON containing string lists. |
| GRA-02 | High | `graph_loader` / `download_dataset` | Do not download built-in `karate`; distinguish generated, bundled, and downloadable graphs. | `graph_loader('karate')` works offline. |
| GRA-03 | High | `graph_urls` | Correct the swapped `ca_hep_th` and `cit_hep_th` sources and verify file readers. | Small source metadata checks and one integration download check. |
| GRA-04 | Medium | graph models | Decide whether Watts-Strogatz means ordinary WS or connected-conditioned WS and document it. | Seeded generator result matches the chosen NetworkX contract. |
| GRA-05 | Medium | graph readers | Preserve or explicitly convert directionality according to dataset definitions. | Directed-source fixtures have documented graph type. |
| PKG-01 | High | `setup.py` | Remove nested `pip install` and the accidental package named `install`; declare runtime and optional dependencies through package metadata. | Wheel/sdist build in an isolated environment without network side effects. |
| PKG-02 | High | package metadata and CI | Support current stable Python releases through 3.14. | Core tests pass on Python 3.8 through 3.14 and visualization tests pass on Python 3.14. |
| PKG-03 | High | `graph_tiger/measures.py` | Remove the `stopit` dependency on unavailable `pkg_resources`. | Timeout behavior and all core imports pass on modern Python. |
| UTL-01 | High | `graph_tiger/utils.py` | Replace deprecated NetworkX sparse APIs, `np.float`, and pip private APIs. | Supported modern NumPy, SciPy, NetworkX, and pip versions import and run. |
| API-01 | Medium | public dispatchers | Validate unknown methods, invalid probabilities, negative sizes, and impossible budgets consistently. | Boundary tests assert one documented exception policy. |
| VIZ-01 | Medium | visualization paths | Use real node labels in coordinates/status mappings and normalize only measures whose definitions require it. | Arbitrary-label graphs render correctly; non-count measures are not divided by `N`. |
| VIZ-02 | Medium | plotting tests | Replace smoke-only tests with file/state assertions and keep heavy optional tests separately selectable. | Each visualization test proves an observable result. |
| DOC-01 | Medium | examples and docs | Remove unused parameters such as `capacity_approx`, correct formula claims, and state mutation/timeline semantics. | Every documented example executes under the supported environment. |
| TST-01 | High | `tests/` | Instantiate `Attack` and `Defense`, exercise direct utilities/graphs, arbitrary labels, large spectra, invalid inputs, and reproducibility. | Each audited failure has a focused regression. |

## Delivery phases

### Phase 0 - Baseline and contracts

- Record the baseline environment and current test results.
- Keep the initial expected-behavior regression suite in `tests/test_regressions.py`.
- Confirm the cascade contract and any compatibility alias before production changes.

### Phase 1 - Defense state correctness

- Fix DEF-01 through DEF-04 and the edge-defense invariants.
- Add focused checks to the existing defense golden tests only where corrected behavior changes their expected values.
- Run attacks, defenses, regressions, and a no-mutation check.

### Phase 2 - Cascading model

- Implement the Motter-Lai overload model as the default with explicit functional-node and failed-node state.
- Keep the corrected historical rule temporarily as `legacy_redistribution` and mark it as TIGER-specific.
- Make transitions synchronous and terminate early at a fixed point while preserving the documented output length.
- Add analytical fixtures for paths, stars, disconnected graphs, simultaneous overloads, zero-load nodes, no-recipient failures, edge attacks, and defenses.
- Update the class docstring, example, and citation in the same commit.

### Phase 3 - Diffusion, seeding, and simulation lifecycle

- Preserve the verified SIS/SIR transition logic.
- Add parameter validation, local random generators, repeatable ensemble semantics, and a consistent timeline.
- Test zero/one probabilities, simultaneous infection and recovery, vaccination overlap, invalid model names, `c=0`, `c=1`, `d=0`, empty graphs, and repeated runs.

### Phase 4 - Measures and numerical utilities

- Correct the standard betweenness definition and update the legacy golden values.
- Correct exact spectral computations and natural-connectivity normalization.
- Add closed-form checks for paths, cycles, complete graphs, trees, disconnected graphs, and graphs at sizes 99, 100, and 101.
- Modernize deprecated numerical APIs without changing unrelated style.

### Phase 5 - Attack and defense algorithms

- Restore arbitrary node labels in NetShield.
- Add shared validation for `k` and feasible nonedges.
- Repair preferential addition and rewiring invariants.
- Verify every reported addition/removal by applying it to a copy and checking simple-graph invariants.

### Phase 6 - Graph loading, packaging, and optional dependencies

- Separate built-in, bundled, and downloadable graph registries.
- Correct dataset source metadata and directionality.
- Move visualization imports behind their feature paths.
- Replace setup-time installation with declarative core and optional dependencies.
- Add an isolated packaging/import job to CI.

### Phase 7 - Visualizations, examples, and documentation

- Add observable assertions to visualization tests.
- Run every example with small deterministic parameters.
- Reconcile all docstrings, Sphinx pages, README claims, and citations with the corrected contracts.

### Phase 8 - Compatibility and release

- Run the complete suite on the supported Python dependency matrix.
- Document corrected numerical outputs and intentional behavior changes.
- Add release notes and bump the version according to the project's compatibility policy.
- Merge only after the full suite is green and the branch diff contains no unrelated formatting.

## Initial regression coverage

The setup commit adds direct tests for:

- labeled-node NetShield output;
- node protection set arithmetic;
- defense budget handling;
- edge-attack removal;
- rewiring removal/application and caller graph immutability;
- preferential addition and rewiring validity;
- exact SIS and SIR synchronous state transitions;
- diffusion parameter validation and requested seed handling;
- Motter-Lai initial load/capacity, cascade seed handling, edge-state separation, input immutability, legacy one-time redistribution, functioning-neighbor conservation, and timeline length;
- standard average vertex betweenness;
- exact large-path spanning-tree count and effective resistance;
- natural-connectivity normalization;
- JSON graph options and offline Karate loading.

## Implementation status

Completed on `codex/correctness-remediation` for the 0.4.0 release candidate.

- Phases 0 through 8 and every issue ID in the ledger are implemented.
- The core suite passes on Python 3.8 through 3.14.
- The optional visualization suite passes on Python 3.14 with ForceAtlas2, Datashader, image, and animation coverage.
- Motter-Lai remains the default cascading model, the weighted efficiency-dynamics implementation is available as `crucitti`, and the corrected historical TIGER rule is available only as `legacy_redistribution`.

## Acceptance criteria

- All existing and new tests pass without weakening expected values or hiding failures behind broad exception handling.
- Every issue in the ledger is fixed, explicitly deferred with rationale, or converted into a documented compatibility decision.
- Caller-owned graphs and global RNG/numerical state are not mutated unexpectedly.
- Core imports succeed without visualization extras.
- Published-model names, equations, state transitions, and citations agree.
- The final diff preserves the repository's established style and contains no unrelated reformatting.

## Local allocation follow-up

The local-allocation issue cluster extends the existing cascade contracts:

- CAS-09: add greedy, proportional, and coordinated maximum-flow allocation with complete overflow transfers, pre-round headroom, explicit insertion-order ties, and synchronous failures.
- CAS-10: accept copied application loads/capacities and exact initiating failures; expose lost service without breaking the older shed_load field.
- UTL-02: allow empty-graph simulation initialization through a zero-size sparse adjacency matrix.
- DOC-01 / TST-01: synchronize the tutorial, API reference, runnable figures, analytical examples, conservation invariants, and independent min-cut checks in the same PR.

## Primary references

- W. O. Kermack and A. G. McKendrick, "A Contribution to the Mathematical Theory of Epidemics," 1927.
- R. Pastor-Satorras and A. Vespignani, "Epidemic Spreading in Scale-Free Networks," 2001.
- A. E. Motter and Y.-C. Lai, "Cascade-Based Attacks on Complex Networks," Physical Review E 66, 065102, 2002.
- P. Crucitti, V. Latora, and M. Marchiori, "Model for Cascading Failures in Complex Networks," Physical Review E 69, 045104, 2004.
- H. Tong, B. A. Prakash, C. Tsourakakis, T. Eliassi-Rad, C. Faloutsos, and D. H. Chau, "On the Vulnerability of Large Graphs," KDD 2010.
- E. Estrada, "Network Robustness to Targeted Attacks. The Interplay of Expansibility and Degree Distribution," 2006.
