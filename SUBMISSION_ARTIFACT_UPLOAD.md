# Artifact Upload Checklist

Use this file when creating the artifact record for ICDE submission.

## Author Mode

The current `main.tex` includes the full author block. ICDE 2027 uses single-blind review, so this is the intended submission mode.

## File to Upload

Upload this zip file:

```text
packages/ICDE_TRAPS_github_code.zip
```

Do not upload local scratch directories or build byproducts.

## Suggested Artifact Metadata

Title:

```text
TRAPS: One-Sided Learned Admission Indexing for Flood-Resilient Password Verification
```

Short description:

```text
Reproducibility package for the ICDE submission. The artifact contains the LaTeX source, generated result CSVs, experiment scripts, ICDE-facing figures, artifact guide, package manifest, and executable audit for the seeded production frontier, shift grid, dynamic drift guard, ablation, LANL status-outcome replay, LANL invalid-heavy stress replay, and queueing experiments.
```

Suggested tags:

```text
approximate membership, learned index, admission control, password verification, ICDE artifact
```

License:

```text
Use the conference system's default anonymous-review license, or leave blank if the artifact service supports private/anonymous review links without a public license.
```

## Link to Enter in the Submission System

After upload, enter the artifact URL in the submission metadata.

Record the final URL here before submission:

```text
ARTIFACT_URL=https://anonymous.4open.science/r/ICDETRAPS20260609
```

## Final Local Verification

Run from the repository root after any documentation or package change:

```powershell
python experiments/package_submission.py
python experiments/audit_submission_requirements.py
```

The expected result is:

```text
PASS audit_frontier
PASS audit_shift_grid
PASS audit_ablation
PASS audit_lanl
PASS audit_queueing
PASS audit_manuscript_framing
PASS audit_latex_and_packages
All submission requirement audits passed.
```

The audit checks experiment coverage, manuscript framing, checksums, package cleanliness, stale-file protection, current author-mode consistency, and local-path leakage.
