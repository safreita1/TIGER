---
name: tiger-correctness-remediation
description: Repair and verify TIGER algorithms, simulations, measures, graph utilities, packaging, examples, and tests against their documented and literature-defined behavior while preserving the repository's existing style. Use for work on the correctness-remediation branch or any TIGER correctness review.
---

# TIGER Correctness Remediation

Use this skill when reviewing or fixing correctness in the TIGER repository.

## Source of truth

Read `../../../CORRECTNESS_REMEDIATION_PLAN.md` before making changes. Treat its scientific contracts, issue IDs, delivery phases, and acceptance criteria as the working specification.

## Required workflow

1. Confirm the active branch is `codex/correctness-remediation` or a child branch created for one issue cluster.
2. Inspect the target function, its docstring, neighboring functions, existing tests, examples, and cited source before editing.
3. Identify the exact contract: inputs, outputs, state transition, mutation policy, randomness, numerical precision, and failure behavior.
4. Add or enable the smallest deterministic regression that fails for the audited reason.
5. Make the smallest production change that satisfies the contract.
6. Run the focused test file, the relevant neighboring test files, and then the full suite.
7. Inspect the diff for unrelated formatting, import churn, or behavior changes.
8. Update documentation and citations in the same change whenever a mathematical contract changes.
9. Record the addressed issue IDs in the commit message or pull-request description.

## Style preservation

- Match nearby imports, blank lines, naming, dictionaries, loops, comments, and docstrings.
- Keep tests as top-level `test_*` functions with direct `assert` statements and the existing `main()` runner.
- Do not introduce pytest fixtures, decorators, parametrization, Hypothesis, formatters, or broad mechanical rewrites unless the owner explicitly requests them.
- Do not reorder or reformat unaffected code.
- Prefer a short local helper only when it removes repetition in the same style already used by the repository.

## Scientific verification

### Diffusion

- Preserve synchronous discrete-time SIS/SIR semantics.
- Transmission is independent per infected-susceptible edge with probability `b`.
- Recovery applies to nodes infected at the start of the step with probability `d`.
- SIS recovery returns to susceptible; SIR recovery is permanent.
- Newly infected nodes act starting in the next step.
- Validate model names, probabilities, initial fraction, runs, and steps.

### Cascading failure

- Do not call a model Crucitti or Motter-Lai unless its load, capacity, failure, recomputation/redistribution, and robustness equations match the cited source.
- Keep node failures and edge failures in distinct state.
- Use synchronous failure updates.
- Prove load conservation where a redistribution model is used.
- Never process the same failed load twice.
- Never mutate the caller's graph.

### Measures

- Verify formulas on analytically solvable toy graphs.
- Check algorithm branches on both sides of size thresholds.
- Preserve graph order in standard normalizations.
- Distinguish an explicit approximation from an exact default.
- Avoid global numerical-state changes and premature rounding.

### Attacks and defenses

- Preserve arbitrary node labels.
- Validate `k` against the feasible node, edge, or nonedge set.
- Every added edge must be a feasible nonedge and not a self-loop.
- Every removed edge must exist in the working graph.
- Rewiring must apply removal and addition to the same simulation copy.
- Node protection removes `attacked - protected`.

## Test design

- Use paths, cycles, stars, complete graphs, empty graphs, disconnected graphs, and small labeled graphs.
- Prefer exact closed forms and state sets to approximate aggregate inequalities.
- For random behavior, assert same-seed reproducibility and different-seed divergence on fixtures where the result is deterministic for those seeds.
- Add boundary cases for zero, one, maximum feasible, impossible, negative, and unknown inputs.
- Test caller-state preservation and repeated runs.
- Do not resolve a failing regression by weakening its assertion unless the scientific contract was explicitly revised.

## Completion checklist

- The focused regression fails before the fix and passes after it.
- Existing tests are updated only when they encoded the audited wrong behavior.
- Relevant examples and documentation agree with the corrected behavior.
- Optional visualization dependencies do not block core imports.
- The full suite passes.
- The diff contains no unrelated style changes.
- The issue ledger and release notes reflect the final decision.
