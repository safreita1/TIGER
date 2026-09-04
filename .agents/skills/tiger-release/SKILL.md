---
name: tiger-release
description: Prepare and publish a TIGER release by keeping implementation, tests, source and hosted documentation, citations, version metadata, GitHub, and PyPI artifacts synchronized. Use when adding a releasable TIGER feature or fix, preparing a version, merging its release PR, or publishing graph-tiger.
---

# TIGER Release

Use this skill for end-to-end changes that must reach both the TIGER repository and the `graph-tiger` package. A release is complete only when the merged source, documentation, hosted Read the Docs site, version metadata, GitHub release, and PyPI artifacts agree.

Publishing, merging, tagging, and creating releases are external mutations. Perform them only when the user explicitly authorizes the corresponding action.

## Inspect before editing

Read the current versions of:

- `setup.py`
- `CHANGELOG.md`
- `docs/source/conf.py`
- `.readthedocs.yaml`
- `.github/workflows/ci.yml`
- `.github/workflows/cd.yml`
- the affected production module, tests, API documentation, tutorial, README section, and bibliography entries

Check the latest GitHub release and the versions already present on PyPI. PyPI files are immutable: never reuse a published version.

For correctness changes governed by `tiger-correctness-remediation`, read and follow that skill as well.

## Keep the change synchronized

A public behavior change normally requires all of the following in the same feature branch:

1. Update the production implementation without unrelated reformatting.
2. Add deterministic regression tests for the scientific or API contract.
3. Update the affected docstrings and `docs/source` API page.
4. Update a tutorial or runnable example when users need new parameters or outputs.
5. Update the README when the public model or technique inventory changes.
6. Add or correct literature entries in `docs/source/refs.bib` when the implementation follows a published method.
7. Add a concise entry under the current changelog release section.

Documentation must describe the code that will actually ship. State initialization, update order, randomness, mutation policy, outputs, edge cases, and model-specific conventions when they affect results. Do not claim fidelity to a named method without tests for its defining behavior.

## Choose and set the version

Use semantic versioning relative to the latest published package:

- Patch release for backward-compatible fixes.
- Minor release for a new public model, parameter, output, or other backward-compatible feature.
- Major release for intentional breaking API or behavior changes.

Before merge, update every active version source:

- `version = "<version>"` in `setup.py`
- `release = '<version>'` in `docs/source/conf.py`
- Replace `## Unreleased` in `CHANGELOG.md` with a descriptive heading beginning `## <version>`

Search the repository for the previous version and inspect every match. Update only values that represent the current package or documentation release; retain historical changelog entries and old-version examples when they are intentionally historical.

## Validate the release candidate

Run or confirm all of the checks encoded by `.github/workflows/ci.yml`:

- Core tests across the supported Python matrix.
- Optional visualization tests.
- `python -m build`.
- `python -m twine check dist/*`.
- Install the built wheel in a clean location and import `graph_tiger`.
- Build the Sphinx HTML documentation using the dependencies and configuration declared in `.readthedocs.yaml`.

Review the complete PR diff and changed-file list. Confirm that package data, imports, citations, examples, and changelog statements match the implementation. Do not merge while any required job is pending or failing.

## Merge and release

When the user authorizes merge and release:

1. Reconfirm the PR head SHA and green CI.
2. Merge the reviewed PR, preferably by squash unless repository history or the user requires another method.
3. Wait for the `master` push CI to finish successfully.
4. Create a GitHub release whose tag exactly matches the package version, such as `0.5.0`, and target the reviewed merge commit.
5. Publish it as a normal release, not a draft or prerelease, unless the user requested otherwise.
6. Let `.github/workflows/cd.yml` build, validate, and upload the package using the repository's `PYPI_API_TOKEN` secret. Never retrieve or print that secret.
7. Confirm that Read the Docs starts a `latest` build from the current `master` commit. If no build appears, use an authenticated project-owner session to trigger `latest`; reconnect the Git integration when pushes no longer trigger builds automatically.

Create the GitHub release through an authenticated user or integration that emits a normal release event. A release created by a workflow using its own `GITHUB_TOKEN` does not trigger another workflow, so it will not automatically start the release-driven PyPI job.

If the available release surface cannot emit the publishing event, do not assume that a visible GitHub release means the package shipped. With explicit release authorization, use a one-purpose workflow on an isolated `codex/release-<version>` branch to:

- confirm the GitHub release exists;
- check out the exact merge commit;
- run the same build and `twine check` commands as `cd.yml`;
- upload with the existing `PYPI_API_TOKEN` secret; and
- delete the temporary branch after successful verification.

Keep the default publishing workflow unchanged. Do not merge the temporary release workflow into `master`.

## Verify the published result

A successful workflow is necessary but not sufficient. Verify all of the following:

- The PR is merged and its merge commit is on `master`.
- The `master` test workflow passed for that commit.
- The GitHub release is published under the intended tag and targets that commit.
- PyPI's JSON endpoint `https://pypi.org/pypi/graph-tiger/<version>/json` returns the intended version.
- PyPI lists both the wheel and source distribution.
- The package can be installed with `pip install --upgrade graph-tiger==<version>`.
- `https://graph-tiger.readthedocs.io/en/latest/` shows the intended release and the deployed revision corresponds to the current `master` commit.
- Changed public APIs and tutorials are present on the hosted pages, not only in `docs/source`.

Report the PR, merge commit, GitHub release, PyPI page, artifact names, and validation results. If any item fails, report the exact failing stage and leave the next irreversible action undone.

## Stop conditions

- Do not release from a failing or unreviewed commit.
- Do not publish when code, changelog, and version metadata disagree.
- Do not overwrite or attempt to replace an existing PyPI artifact.
- Do not expose repository or PyPI credentials in commands, logs, comments, or responses.
- Do not interpret a successful GitHub release as proof of a successful PyPI upload; verify PyPI directly.
- Do not report documentation deployment complete until the Read the Docs build succeeds and the live pages expose the released version and changed API.
