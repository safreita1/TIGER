# Changelog

## 0.8.0 - GPU acceleration

### Added

- Added optional nx-cuGraph dispatch for centrality attacks, defenses, and Motter--Lai load recomputation.
- Added parity-preserving CuPy execution for synchronous SIS/SIR, independent cascade, linear threshold, and competitive cascade models.
- Added a 270-case non-spectral GPU benchmark with exact selection/state gates and workload-specific crossover reporting.
- Added exact GPU implementations for diameter, average distance, average inverse distance, average clustering coefficient, and largest-connected-component size.
- Extended the robustness benchmark to cover every GPU-capable measure and to distinguish sampled betweenness from exact runs.
- Added CUDA 12 and CUDA 13 installation extras plus the ``tiger-gpu-status`` hardware and runtime diagnostic.

### Fixed

- Corrected nx-cuGraph approximate edge-betweenness scaling and stabilized centrality tie ordering across CPU and GPU backends.

### Changed

- Synchronous stochastic simulations now use a backend-independent counter-based random stream. Equal seeds reproduce exactly across CPU and GPU, but seeded trajectories can differ from releases that used Python's iteration-dependent random draws.
- Automatic GPU selection uses measured, operation-specific crossover defaults and stays on CPU when the measured advantage is marginal. Full-state epidemic and influence simulations remain CPU in automatic mode; explicit GPU selection remains available.
- Largest-connected-component measurement remains on CPU in automatic mode because its GPU end-to-end path was slower through 20,000 nodes; explicit GPU selection remains available.

## 0.7.0 - Information diffusion and influence

### Added

- Added independent-cascade and linear-threshold simulations with synchronous frontiers, directed-edge support, and scalar or attributed influence parameters.
- Added an asynchronous voter model with complete caller-supplied initial states.
- Added competitive message cascades with message-specific probabilities and explicit random or priority tie resolution.
- Added per-step state, count, frontier, and changed-node histories; normalized plots; reproducible resets; and fixed-length repeated-run trajectories.
- Added analytical and visualization tests plus a customer guide with four reproducible model figures, downloadable observations, parameters, and primary references.

## 0.6.0 - Capacity-aware local allocation

### Added

- Local cascades accept `allocation='greedy'`, `'proportional'`, or `'max_flow'`, alongside the existing default `'degree'` policy. All policies transfer full displaced workloads; greedy and maximum-flow allocations distribute overflow equally.
- Added complete `initial_load` and `capacities` mappings and an exact `initial_failures` set for local cascades. Supplied capacities are used directly and all inputs are copied for repeatable resets.
- Added `lost_load` for work with no functioning recipient, retaining `shed_load` as a compatible attribute/history alias, and `last_transfers` for inspecting individual handoffs.
- Added analytical and randomized conservation/max-flow tests, a runnable allocation study, and six tutorial figures with transfer data.

### Fixed

- Empty graphs now have a zero-by-zero sparse adjacency matrix rather than failing during simulation construction.

## 0.5.0 - Local load-sharing cascades

### Added

- Added `model='local_load_sharing'`, implementing degree-weighted local redistribution with `beta=0` for equal sharing and larger values favoring higher-degree neighbors.
- Local cascade states now report load that cannot reach a functioning neighbor as `shed_load`.
- Added analytical regressions for initialization, equal and preferential allocation, synchronous updates, stranded load, and parameter validation.

## 0.4.0 - Crucitti model and modern Python

### Added

- Added the Crucitti-Latora-Marchiori weighted efficiency-dynamics model as `model='crucitti'`.
- Added analytical tests for congestion, recovery, weighted route selection, network efficiency, model validation, and caller-graph preservation.
- Added core compatibility testing and package classifiers for Python 3.12, 3.13, and 3.14.

### Changed

- The visualization dependency stack now uses the current ForceAtlas2 package and is tested on Python 3.14 without legacy NumPy or Cython installation pins.
- Replaced the obsolete `stopit`/`pkg_resources` runtime path with a standard-library timeout wrapper compatible with modern Python.
- Crucitti simulations report average network efficiency and expose overloaded nodes and per-edge efficiencies in each recorded state.

## 0.3.0 - Correctness remediation

### Corrected

- Defense simulations now remove unprotected attacked nodes, apply edge attacks as edge removals, honor `k_d`, and apply complete rewiring operations to the simulation copy.
- Cascading simulations now default to a documented Motter-Lai overload model. The corrected historical redistribution rule is available as `legacy_redistribution`.
- SIS and SIR initialization now honors the requested seed, validates probabilities and model names, and exposes an initial state followed by synchronous transitions.
- NetShield preserves arbitrary node labels and attack/defense budgets are validated.
- Edge additions and rewiring terminate on impossible inputs and return valid simple-graph changes.
- Vertex betweenness, natural connectivity, spanning-tree count, and effective resistance now follow their standard definitions across the 100-node numerical threshold.
- Graph options are valid JSON, Karate loads offline, HepTh dataset metadata is no longer swapped, and directed-source conversion is explicit.

### Packaging and tests

- Removed setup-time nested `pip install` calls and declared core, test, and visualization dependencies.
- Core imports no longer require ForceAtlas2 or Datashader.
- Added deterministic regression coverage for audited algorithms, simulations, measures, graphs, random seeds, edge cases, and visualization artifacts.
- Updated CI to exercise the supported Python matrix and the optional visualization stack separately.

### Compatibility

- Simulation histories now contain `steps + 1` values: the initial state at index 0 and one state after each requested transition.
- `largest_connected_component` is explicitly a node count and returns 0 for an empty graph.
- Unknown methods, measures, graph names, invalid probabilities, and impossible budgets raise `ValueError` instead of printing and returning an ambiguous empty result.
