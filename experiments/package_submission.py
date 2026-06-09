from __future__ import annotations

import zipfile
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
PACKAGES.mkdir(exist_ok=True)


USED_FIGURES = [
    "figure/LBF_pic/new_lbf_fig2.pdf",
    "figure/icde_experiments/icde_frontier_production.pdf",
    "figure/icde_experiments/icde_shift_grid_heatmap.pdf",
    "figure/icde_experiments/icde_scale_update_sweep.pdf",
    "figure/icde_experiments/icde_lanl_trace_suite.pdf",
    "figure/icde_experiments/icde_queueing_by_method.pdf",
    "figure/icde_experiments/icde_replay_cache_ci.pdf",
]


TOP_LEVEL = [
    "main.tex",
    "references.bib",
    "main.bbl",
    "main.pdf",
    "README_CODE_PACKAGE.md",
    "ARTIFACT_EVALUATION.md",
    "PACKAGE_MANIFEST.md",
    "SUBMISSION_ARTIFACT_UPLOAD.md",
]


EXPERIMENT_SCRIPTS = [
    "experiments/run_icde_experiments.py",
    "experiments/run_lanl_trace_experiment.py",
    "experiments/audit_submission_requirements.py",
    "experiments/package_submission.py",
]


def add_file(zf: zipfile.ZipFile, rel: str) -> None:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(rel)
    zf.write(path, rel)


def add_tree(zf: zipfile.ZipFile, root_rel: str, suffixes: set[str] | None = None) -> None:
    root = ROOT / root_rel
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or "_rendered_dashboards" in path.parts:
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        zf.write(path, path.relative_to(ROOT).as_posix())


def context_tex_rels() -> list[str]:
    rels: list[str] = []
    for path in sorted((ROOT / "context").rglob("*.tex")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == "context/Appendix.tex":
            continue
        rels.append(rel)
    return rels


def write_zip(path: Path, rels: list[str]) -> None:
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in rels:
            add_file(zf, rel)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums() -> Path:
    rels = [
        "main.pdf",
        "packages/ICDE_TRAPS_github_code.zip",
        "packages/ICDE_TRAPS_overleaf_minimal.zip",
    ]
    checksum_path = PACKAGES / "SHA256SUMS.txt"
    lines = [f"{sha256_file(ROOT / rel)}  {rel}\n" for rel in rels]
    checksum_path.write_text("".join(lines), encoding="utf-8")
    return checksum_path


def main() -> None:
    overleaf_rels = [
        "main.tex",
        "references.bib",
        "main.bbl",
        *context_tex_rels(),
        *USED_FIGURES,
    ]
    write_zip(PACKAGES / "ICDE_TRAPS_overleaf_minimal.zip", overleaf_rels)

    github_zip = PACKAGES / "ICDE_TRAPS_github_code.zip"
    if github_zip.exists():
        github_zip.unlink()
    with zipfile.ZipFile(github_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in TOP_LEVEL:
            add_file(zf, rel)
        for rel in context_tex_rels():
            add_file(zf, rel)
        for rel in EXPERIMENT_SCRIPTS:
            add_file(zf, rel)
        add_tree(zf, "experiments/results", suffixes={".csv"})
        for rel in USED_FIGURES:
            add_file(zf, rel)
        # Include generated PNG companions for quick inspection, but not old
        # dashboard artifacts or Python bytecode.
        for path in sorted((ROOT / "figure" / "icde_experiments").glob("*.png")):
            zf.write(path, path.relative_to(ROOT).as_posix())

    checksum_path = write_checksums()
    print(PACKAGES / "ICDE_TRAPS_github_code.zip")
    print(PACKAGES / "ICDE_TRAPS_overleaf_minimal.zip")
    print(checksum_path)


if __name__ == "__main__":
    main()
