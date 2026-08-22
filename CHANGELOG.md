# Changelog

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
