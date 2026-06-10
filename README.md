# TRAPS Reproducibility Code

This repository contains the code artifact for the TRAPS ICDE submission:

**TRAPS: One-Sided Learned Admission Indexing for Flood-Resilient Password Verification**

The artifact is intentionally limited to experiment code and lightweight CSV outputs. It does not include the paper source, compiled PDFs, Overleaf package, or submission packaging scripts.

## Contents

- `experiments/run_icde_experiments.py`: deterministic synthetic workload, allocation, ablation, queueing, scale-update, and replay-cache experiments.
- `experiments/run_lanl_trace_experiment.py`: LANL authentication-trace replay and negative-control experiments.
- `experiments/results/*.csv`: committed result tables corresponding to the current submission experiments.
- `requirements.txt`: Python package dependencies.

Running the scripts may create derived figures under `figure/icde_experiments/` and generated LaTeX tables under `context/generated/`. Those derived files are ignored by Git.

## Setup

Use Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce Controlled Experiments

Run all deterministic controlled experiments:

```bash
python experiments/run_icde_experiments.py
```

Selected suites can be run independently:

```bash
python experiments/run_icde_experiments.py --suite frontier_production --seeds 0-29 --bits 4,6,8,10,12,14,16,20,24
python experiments/run_icde_experiments.py --suite shift_grid --seeds 0-9 --bits 12,16
python experiments/run_icde_experiments.py --suite traps_ablation --seeds 0-29 --bits 12,16
```

## Reproduce LANL Replay

The LANL replay downloads the public LANL authentication trace at runtime.

```bash
python experiments/run_lanl_trace_experiment.py
```

If the local Python certificate store cannot verify the LANL TLS chain, run:

```bash
LANL_AUTH_INSECURE=1 python experiments/run_lanl_trace_experiment.py
```

On Windows PowerShell:

```powershell
$env:LANL_AUTH_INSECURE='1'
python experiments/run_lanl_trace_experiment.py
```

For a quick smoke run, limit the trace size:

```bash
LANL_AUTH_MAX_LINES=50000 python experiments/run_lanl_trace_experiment.py
```

## Outputs

Both scripts write CSV files to `experiments/results/`. They also regenerate plotting outputs under `figure/icde_experiments/`; these figures are derived from the committed code and CSV data and are not tracked in this cleaned code package.
