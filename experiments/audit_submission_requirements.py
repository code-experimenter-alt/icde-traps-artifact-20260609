from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "figure" / "icde_experiments"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"missing file: {path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_no_type3_fonts(path: Path) -> None:
    result = subprocess.run(["pdffonts", str(path)], text=True, capture_output=True, check=True)
    offenders = [line for line in result.stdout.splitlines() if "Type 3" in line]
    require(not offenders, f"{path.relative_to(ROOT)} contains Type 3 font(s): {offenders}")


def floats(rows: list[dict[str, str]], key: str) -> set[float]:
    return {float(r[key]) for r in rows}


def ints(rows: list[dict[str, str]], key: str) -> set[int]:
    return {int(float(r[key])) for r in rows}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_checksums() -> None:
    checksum_path = ROOT / "packages" / "SHA256SUMS.txt"
    require(checksum_path.exists(), "missing packages/SHA256SUMS.txt")
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        require(len(parts) == 2, f"invalid checksum line: {line}")
        recorded[parts[1].replace("\\", "/")] = parts[0].lower()
    expected = [
        "main.pdf",
        "packages/ICDE_TRAPS_github_code.zip",
        "packages/ICDE_TRAPS_overleaf_minimal.zip",
    ]
    require(set(expected) <= set(recorded), "checksum manifest does not cover all submission artifacts")
    for rel in expected:
        require((ROOT / rel).exists(), f"checksum target missing: {rel}")
        require(recorded[rel] == sha256_file(ROOT / rel), f"stale checksum for {rel}")


def audit_frontier() -> None:
    rows = read_csv(RESULTS / "frontier_production.csv")
    summary = read_csv(RESULTS / "frontier_production_summary.csv")
    memory = read_csv(RESULTS / "frontier_production_memory.csv")
    expected_methods = {
        "No admission",
        "Global Bloom",
        "GlobalBloom+NegCache",
        "Hash partition",
        "Risk partition",
        "TRAPS learned",
        "TRAPS+ReplayCache",
        "Oracle AMQ",
        "Peppered tag",
        "Peppered tag-64",
        "Adaptive tag",
        "NegCache only",
        "Tag+NegCache",
        "RateLimit+Tag",
        "RateLimit+Tag+NegCache",
    }
    expected_bits = {4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0}
    methods = {r["method"] for r in rows}
    require(expected_methods <= methods, f"frontier missing methods: {sorted(expected_methods - methods)}")
    require(floats(rows, "bits_per_key") == expected_bits, "frontier bits/key grid is incomplete")
    require(ints(rows, "seed") == set(range(30)), "frontier seeds must be 0..29")
    required_columns = {
        "N_keys",
        "N_events",
        "N_valid",
        "N_invalid",
        "N_invalid_unique",
        "N_invalid_repeated",
        "valid_forwarded",
        "valid_rejected",
        "valid_throttled",
        "invalid_forwarded",
        "invalid_rejected_by_index",
        "invalid_rejected_by_cache",
        "invalid_throttled",
        "invalid_forwarded_unique",
        "invalid_forwarded_repeated",
        "backend_invocations",
        "raw_invalid_forward_rate",
        "first_seen_invalid_forward_rate",
        "replay_suppression_rate",
        "backend_work_reduction",
        "valid_rejection_rate",
    }
    require(required_columns <= set(rows[0]), f"frontier missing columns: {sorted(required_columns - set(rows[0]))}")
    summary_columns = {
        "mean_raw_invalid_forward_rate",
        "median_raw_invalid_forward_rate",
        "ci95_raw_lo",
        "ci95_raw_hi",
        "mean_first_seen_invalid_forward_rate",
        "median_first_seen_invalid_forward_rate",
        "ci95_first_seen_lo",
        "ci95_first_seen_hi",
        "zero_event_upper_95",
        "n_seeds",
        "total_invalid",
        "total_unique_invalid",
    }
    require(summary_columns <= set(summary[0]), f"frontier summary missing CI/denominator columns: {sorted(summary_columns - set(summary[0]))}")
    for row in rows:
        require(int(float(row["N_keys"])) >= 200_000, "frontier N_keys below Pro minimum")
        require(int(float(row["N_valid"])) >= 50_000, "frontier N_valid below Pro minimum")
        require(int(float(row["N_invalid"])) >= 1_000_000, "frontier N_invalid below Pro minimum")
        if row["method"] not in {"RateLimit+Tag", "RateLimit+Tag+NegCache"}:
            require(float(row["valid_rejection_rate"]) == 0.0, f"one-sided method rejects valid probes: {row['method']}")
    for row in summary:
        require(int(float(row["n_seeds"])) == 30, "frontier summary must aggregate 30 seeds")
        require(int(float(row["total_invalid"])) >= 30_000_000, "frontier summary denominator below Pro minimum")
        require(float(row["ci95_raw_lo"]) <= float(row["mean_raw_invalid_forward_rate"]) <= float(row["ci95_raw_hi"]), "frontier raw CI does not bracket mean")
        require(float(row["ci95_first_seen_lo"]) <= float(row["mean_first_seen_invalid_forward_rate"]) <= float(row["ci95_first_seen_hi"]), "frontier first-seen CI does not bracket mean")
    by_key = {(r["method"], float(r["bits_per_key"])): r for r in summary}
    require(by_key[("Peppered tag-64", 16.0)]["zero_event_upper_95"] != "", "zero-event tag-64 upper bound missing")
    require(any(float(r["counter_bits_per_key"]) > 0 for r in memory if r["method"].startswith("RateLimit")), "rate-limit counter memory missing")
    require({"index_or_tag_bits_per_key", "cache_bits_per_key", "counter_bits_per_key", "total_bits_per_key"} <= set(memory[0]), "frontier memory breakdown incomplete")

    grouped: dict[tuple[int, float], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(float(row["seed"])), float(row["bits_per_key"]))][row["method"]] = row
    for key, group in grouped.items():
        for cached, base in [
            ("TRAPS+ReplayCache", "TRAPS learned"),
            ("GlobalBloom+NegCache", "Global Bloom"),
            ("Tag+NegCache", "Peppered tag"),
            ("RateLimit+Tag+NegCache", "RateLimit+Tag"),
        ]:
            require(cached in group and base in group, f"cache/base pair missing for {key}: {cached}/{base}")
            require(
                group[cached]["invalid_forwarded_unique"] == group[base]["invalid_forwarded_unique"],
                f"cache changed first-seen invalid forwards for {cached} at {key}",
            )
            require(int(float(group[cached]["invalid_forwarded_repeated"])) == 0, f"cache failed to suppress repeats for {cached} at {key}")


def audit_shift_grid() -> None:
    rows = read_csv(RESULTS / "shift_grid.csv")
    sensitivity = read_csv(RESULTS / "shift_grid_sensitivity.csv")
    worst = read_csv(RESULTS / "shift_grid_worstcase.csv")
    dynamic = read_csv(RESULTS / "shift_dynamic_capacity.csv")
    dynamic_summary = read_csv(RESULTS / "shift_dynamic_capacity_summary.csv")
    require(floats(rows, "bits_per_key") == {12.0, 16.0}, "shift grid must include 12 and 16 bits/key")
    require(floats(rows, "target_zipf_alpha") == {0.4, 0.8, 1.2, 1.6}, "shift grid alpha sweep incomplete")
    require(floats(rows, "risk_shift") == {0.0, 0.25, 0.5, 0.75}, "shift grid risk-shift sweep incomplete")
    require(ints(rows, "seed") == set(range(10)), "shift grid seeds must be 0..9")
    require({"median_backend_reduction", "valid_rejects"} <= set(worst[0]), "shift worst-case table missing Pro columns")
    require(floats(sensitivity, "replay_fraction") == {0.0, 0.5, 0.9}, "replay sensitivity incomplete")
    require(floats(sensitivity, "occupancy_injection_fraction") == {0.0, 0.01, 0.05}, "occupancy sensitivity incomplete")
    dynamic_required = {
        "seed",
        "bits_per_key",
        "target_skew",
        "risk_drift",
        "method",
        "adaptation_mode",
        "monitor_window_events",
        "test_window_events",
        "region_bits_json",
        "raw_invalid",
        "unique_invalid",
        "raw_invalid_forwarded",
        "first_seen_invalid_forwarded",
        "raw_invalid_forward_rate",
        "first_seen_invalid_forward_rate",
        "valid_rejects",
        "rebuild_keys",
        "rebuild_bytes",
        "dual_query_fraction",
        "drift_metric",
        "fallback_triggered",
        "uses_future_labels",
    }
    require(dynamic_required <= set(dynamic[0]), f"dynamic shift CSV missing columns: {sorted(dynamic_required - set(dynamic[0]))}")
    require(
        {"TRAPS-frozen", "TRAPS-dynamic-next-epoch", "TRAPS-drift-guard"} <= {r["method"] for r in dynamic},
        "dynamic shift methods incomplete",
    )
    require(floats(dynamic, "bits_per_key") == {12.0, 16.0}, "dynamic shift bits/key incomplete")
    require(floats(dynamic, "target_skew") == {0.4, 0.8, 1.2, 1.6}, "dynamic shift target-skew sweep incomplete")
    require(floats(dynamic, "risk_drift") == {0.0, 0.25, 0.5, 0.75}, "dynamic shift risk-drift sweep incomplete")
    require(ints(dynamic, "seed") == set(range(10)), "dynamic shift seeds must be 0..9")
    require(all(int(float(r["valid_rejects"])) == 0 for r in dynamic), "dynamic shift has valid rejects")
    require(all(str(r["uses_future_labels"]).lower() == "false" for r in dynamic), "dynamic shift uses future labels")
    require(any(str(r["fallback_triggered"]).lower() == "true" for r in dynamic), "drift guard never triggers fallback")
    by_summary = {(r["method"], float(r["bits_per_key"])): r for r in dynamic_summary}
    frozen16 = float(by_summary[("TRAPS-frozen", 16.0)]["worst_first_seen_IFFR"])
    dynamic16 = float(by_summary[("TRAPS-dynamic-next-epoch", 16.0)]["worst_first_seen_IFFR"])
    guard16 = float(by_summary[("TRAPS-drift-guard", 16.0)]["worst_first_seen_IFFR"])
    global16 = next(float(r["worst_cell_IFFR"]) for r in worst if r["method"] == "Global Bloom" and float(r["bits_per_key"]) == 16.0)
    require(dynamic16 < frozen16, "dynamic next-epoch allocation does not reduce frozen worst-case IFFR")
    require(guard16 <= global16 * 1.01, "drift guard does not fall back near Global Bloom worst-case")


def audit_ablation() -> None:
    rows = read_csv(RESULTS / "traps_ablation.csv")
    expected_methods = {
        "TRAPS full",
        "No learned router",
        "No region allocation",
        "No false-forward cap",
        "No replay cache",
        "Oracle allocation",
    }
    require(expected_methods <= {r["method"] for r in rows}, "TRAPS ablation method set incomplete")
    require(floats(rows, "bits_per_key") == {12.0, 16.0}, "TRAPS ablation bits incomplete")
    require(ints(rows, "seed") == set(range(30)), "TRAPS ablation seeds must be 0..29")
    required_columns = {
        "raw_invalid_forward_rate",
        "first_seen_invalid_forward_rate",
        "backend_invocations",
        "p99_queue_delay_ms",
        "cap_exhaustions",
        "cap_violations",
        "valid_rejects",
        "memory_bits_per_key",
    }
    require(required_columns <= set(rows[0]), f"TRAPS ablation missing columns: {sorted(required_columns - set(rows[0]))}")
    require(any(int(float(r["cap_violations"])) > 0 for r in rows if r["method"] == "No false-forward cap"), "no-cap ablation does not expose cap violations")


def audit_lanl() -> None:
    rows = read_csv(RESULTS / "lanl_trace_conditioned.csv")
    denominators = read_csv(RESULTS / "lanl_trace_denominators.csv")
    controls = read_csv(RESULTS / "lanl_trace_negative_controls.csv")
    status_rows = read_csv(RESULTS / "lanl_status_outcome.csv")
    status_denominators = read_csv(RESULTS / "lanl_status_outcome_denominators.csv")
    require({"Global Bloom", "Hash partition", "Risk partition", "TRAPS learned", "Keyed tag", "Adaptive tag"} <= {r["method"] for r in rows}, "LANL method set incomplete")
    denom_cols = {
        "events_used",
        "unique_users",
        "unique_sources",
        "top_1pct_user_event_mass",
        "gini_user_events",
        "train_test_JS_divergence",
        "synthetic_valid_events",
        "synthetic_invalid_events",
        "synthetic_unique_invalid_pairs",
        "synthetic_replay_fraction",
    }
    require(denom_cols <= set(denominators[0]), f"LANL denominators missing columns: {sorted(denom_cols - set(denominators[0]))}")
    require(float(denominators[0]["synthetic_invalid_events"]) > float(denominators[0]["synthetic_valid_events"]), "LANL stress should be invalid-heavy")
    expected_controls = {"LANL-real", "LANL-shuffled-accounts", "LANL-shuffled-times", "LANL-uniform-risk"}
    require(expected_controls <= {r["control"] for r in controls}, "LANL negative controls incomplete")
    real_traps = next(float(r["invalid_forward_rate"]) for r in controls if r["control"] == "LANL-real" and r["method"] == "TRAPS learned")
    uniform_traps = next(float(r["invalid_forward_rate"]) for r in controls if r["control"] == "LANL-uniform-risk" and r["method"] == "TRAPS learned")
    global_rate = next(float(r["invalid_forward_rate"]) for r in controls if r["control"] == "LANL-real" and r["method"] == "Global Bloom")
    require(abs(uniform_traps - global_rate) < 5e-6, "uniform-risk negative control should collapse TRAPS near Global Bloom")
    require(real_traps < global_rate, "LANL-real TRAPS should not be worse than Global Bloom in the conditioned replay")
    status_methods = {"Global Bloom", "Hash partition", "Risk partition", "TRAPS learned", "Keyed tag", "Adaptive tag"}
    require(status_methods <= {r["method"] for r in status_rows}, "LANL status-outcome method set incomplete")
    status_denom_cols = {
        "lanl_success_events",
        "lanl_failure_events",
        "represented_success_test_events",
        "failure_test_events",
        "uses_lanl_status_as_outcome",
        "uses_plaintext_passwords",
    }
    require(status_denom_cols <= set(status_denominators[0]), f"LANL status denominators missing columns: {sorted(status_denom_cols - set(status_denominators[0]))}")
    require(int(float(status_denominators[0]["represented_success_test_events"])) >= 100_000, "LANL status replay has too few represented success events")
    require(int(float(status_denominators[0]["failure_test_events"])) >= 1_000, "LANL status replay has too few failure events")
    require(str(status_denominators[0]["uses_lanl_status_as_outcome"]).lower() == "true", "LANL status replay must use real Success/Fail outcomes")
    require(str(status_denominators[0]["uses_plaintext_passwords"]).lower() == "false", "LANL status replay must not use plaintext passwords")
    status_global = next(float(r["invalid_forward_rate"]) for r in status_rows if r["method"] == "Global Bloom")
    status_traps = next(float(r["invalid_forward_rate"]) for r in status_rows if r["method"] == "TRAPS learned")
    require(status_traps < status_global, "LANL status-outcome TRAPS should improve over Global Bloom")
    require(all(float(r["valid_fnr"]) == 0.0 for r in status_rows), "LANL status-outcome has represented-success false rejects")


def audit_queueing() -> None:
    rows = read_csv(RESULTS / "queueing_by_method.csv")
    required_columns = {
        "backend_invocations_per_second",
        "worker_utilization",
        "p50_wait_ms",
        "p95_wait_ms",
        "p99_wait_ms",
        "fraction_wait_gt_1s",
        "valid_drop_rate_for_rate_limit_baselines",
    }
    require(required_columns <= set(rows[0]), f"queueing_by_method missing columns: {sorted(required_columns - set(rows[0]))}")
    expected_methods = {"Global Bloom", "TRAPS learned", "TRAPS+ReplayCache", "Peppered tag", "Adaptive tag", "RateLimit+Tag"}
    require(expected_methods <= {r["method"] for r in rows}, "queueing_by_method method set incomplete")


def audit_manuscript_framing() -> None:
    main = (ROOT / "main.tex").read_text(encoding="utf-8", errors="ignore")
    intro = (ROOT / "context" / "Introduction.tex").read_text(encoding="utf-8", errors="ignore")
    blocks = (ROOT / "context" / "Building Blocks.tex").read_text(encoding="utf-8", errors="ignore")
    framework = (ROOT / "context" / "System Framework and Threat Model.tex").read_text(encoding="utf-8", errors="ignore")
    construction = (ROOT / "context" / "Firewall Construction.tex").read_text(encoding="utf-8", errors="ignore")
    exps = (ROOT / "context" / "Experiments.tex").read_text(encoding="utf-8", errors="ignore")
    combined = "\n".join([main, intro, blocks, framework, construction, exps])
    combined_lower = combined.lower()

    required_phrases = [
        "admission indexing for high-cost point queries",
        "Password verification is our case study",
        "Admission-index design problem",
        "physical-design objective is to minimize invalid forwards",
        "learning affects resource allocation rather than authentication correctness",
        "A \\(b\\)-bit keyed tag has residual collision probability",
        "This gap is a substrate property",
        "Bloom filters are the measured AMQ substrate in this paper",
        "\\(\\Gamma_e(u,pw,c)\\)",
        "fail-to-backend",
        "The oracle differs from learned Bloom filters that predict set membership",
        "Our evaluation addresses five questions",
        "We evaluate a disclosed workload matrix",
        "The full grid carries the main evidence",
        "exact-fingerprint operating point",
        "A production 64-bit tag bound is reported separately",
        "Shift/fallback accounting at 16 bits/key",
        "LANL status-outcome replay",
        "uses the trace's real Success/Fail status as the event label",
        "invalid-heavy stress replay",
        "without using future labels",
        "rate-limit hybrids are plotted with their measured valid-drop caveat",
    ]
    missing = [phrase for phrase in required_phrases if phrase.lower() not in combined_lower]
    require(not missing, f"manuscript framing missing required language: {missing}")

    forbidden_claims = [
        "LANL validates TRAPS on real authentication",
        "LANL password verification outcomes",
        "LANL validates password failures",
        "TRAPS has 0 invalid forwards",
        "0% invalid forwards",
        "We do not claim",
        "should be read as",
        "not as deployment evidence",
        "not the main evidence",
        "negative-control stress test",
        "fail closed",
    ]
    offenders = [phrase for phrase in forbidden_claims if phrase.lower() in combined_lower]
    require(not offenders, f"manuscript still contains overclaiming phrase(s): {offenders}")


def audit_latex_and_packages() -> None:
    log = (ROOT / "main.log").read_text(encoding="utf-8", errors="ignore")
    require("Fatal error" not in log, "main.log contains fatal error")
    require("undefined references" not in log.lower(), "main.log contains undefined references")
    require("Citation `" not in log and "Reference `" not in log, "main.log contains undefined citation/reference warnings")
    require("Overfull \\hbox" not in log, "main.log contains overfull hbox")
    match = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    require(match is not None and int(match.group(1)) == 13, "main.pdf must be 13 pages for the current submission build")
    aux = (ROOT / "main.aux").read_text(encoding="utf-8", errors="ignore")
    require("newlabel{sec:related-work}{{VI}{12}" in aux, "bibliography/related-work boundary no longer proves 12-page body")
    main_text = (ROOT / "main.tex").read_text(encoding="utf-8", errors="ignore")
    require("AI-Generated Content Acknowledgement" in main_text, "main.tex missing AI-generated content acknowledgement")
    require("context/Appendix" not in main_text and "\\appendix" not in main_text, "submission source must not include an appendix")
    tex = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "context").glob("*.tex"))
    win_user_marker = "\\" + "Users" + "\\"
    mac_user_marker = "/" + "Users" + "/"
    home_marker = "/" + "home" + "/"
    local_path_pattern = (
        r"(?:[A-Za-z]:\\[^\\\s]+\\|"
        + re.escape(win_user_marker)
        + "|"
        + re.escape(mac_user_marker)
        + "|"
        + re.escape(home_marker)
        + ")"
    )
    email_pattern = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"

    main_source = (ROOT / "main.tex").read_text(encoding="utf-8", errors="ignore")
    anonymous_mode = False
    source_for_anonymity = "\n".join(
        [
            main_source,
            tex,
            (ROOT / "README_CODE_PACKAGE.md").read_text(encoding="utf-8", errors="ignore"),
            (ROOT / "ARTIFACT_EVALUATION.md").read_text(encoding="utf-8", errors="ignore"),
            (ROOT / "PACKAGE_MANIFEST.md").read_text(encoding="utf-8", errors="ignore"),
            (ROOT / "SUBMISSION_ARTIFACT_UPLOAD.md").read_text(encoding="utf-8", errors="ignore"),
        ]
    )
    required_author_terms = [
        "Yadi Wen",
        "Yaxuan Wang",
        "Yujie Xu",
        "Dan Yu",
        "Rong Du",
        "Yue Fu",
        "Yongle Chen",
        "Taiyuan University of Technology",
        "link.tyut.edu.cn",
        "fuyue",
    ]
    missing_author_terms = [term for term in required_author_terms if term not in main_source]
    require(not missing_author_terms, f"single-blind submission missing author term(s): {missing_author_terms}")
    email_leaks = re.findall(email_pattern, source_for_anonymity)
    local_path_leaks = re.findall(local_path_pattern, source_for_anonymity)
    if anonymous_mode:
        require(not email_leaks, f"anonymous source contains email address(es): {email_leaks}")
    require(not local_path_leaks, f"anonymous source contains local absolute path marker(s): {local_path_leaks}")
    pdf_text = (ROOT / "main.pdf").read_bytes().decode("latin-1", errors="ignore")
    if anonymous_mode:
        require(not re.findall(email_pattern, pdf_text), "main.pdf metadata/text contains email address")
    require(not re.findall(local_path_pattern, pdf_text), "main.pdf metadata/text contains local absolute path marker")
    forbidden = [
        "dataset_a_dashboard",
        "dataset_b_dashboard",
        "dataset_c_dashboard",
        "dataset_c_big_dashboard",
        "lbf_layers_tradeoff",
        "e2e-credential",
        "e2e-occupancy",
        "e2e-drifted",
        "e2e-benign",
    ]
    require(not any(s in tex for s in forbidden), "old dashboard/layer-sweep references remain in main text")
    for rel in [
        "figure/icde_experiments/icde_frontier_production.pdf",
        "figure/icde_experiments/icde_shift_grid_heatmap.pdf",
        "figure/icde_experiments/icde_scale_update_sweep.pdf",
        "figure/icde_experiments/icde_lanl_trace_suite.pdf",
        "figure/icde_experiments/icde_queueing_by_method.pdf",
        "figure/icde_experiments/icde_replay_cache_ci.pdf",
        "packages/ICDE_TRAPS_github_code.zip",
        "packages/ICDE_TRAPS_overleaf_minimal.zip",
        "packages/SHA256SUMS.txt",
    ]:
        require((ROOT / rel).exists(), f"missing deliverable: {rel}")
    for rel in [
        "main.pdf",
        "figure/icde_experiments/icde_frontier_production.pdf",
        "figure/icde_experiments/icde_shift_grid_heatmap.pdf",
        "figure/icde_experiments/icde_scale_update_sweep.pdf",
        "figure/icde_experiments/icde_lanl_trace_suite.pdf",
        "figure/icde_experiments/icde_queueing_by_method.pdf",
        "figure/icde_experiments/icde_replay_cache_ci.pdf",
    ]:
        require_no_type3_fonts(ROOT / rel)
    audit_checksums()
    with zipfile.ZipFile(ROOT / "packages" / "ICDE_TRAPS_github_code.zip") as zf:
        names = set(zf.namelist())
    require(not any(re.findall(local_path_pattern, n) for n in names), "GitHub package entry names contain local absolute path markers")
    require(not any(re.findall(email_pattern, n) for n in names), "GitHub package entry names contain email-like strings")
    require("README_CODE_PACKAGE.md" in names, "README missing from GitHub package")
    require("ARTIFACT_EVALUATION.md" in names, "artifact guide missing from GitHub package")
    require("PACKAGE_MANIFEST.md" in names, "package manifest missing from GitHub package")
    require("SUBMISSION_ARTIFACT_UPLOAD.md" in names, "artifact upload checklist missing from GitHub package")
    require("experiments/audit_submission_requirements.py" in names, "audit script missing from GitHub package")
    require("experiments/package_submission.py" in names, "packaging script missing from GitHub package")
    with zipfile.ZipFile(ROOT / "packages" / "ICDE_TRAPS_github_code.zip") as zf:
        for rel in [
            "main.tex",
            "README_CODE_PACKAGE.md",
            "ARTIFACT_EVALUATION.md",
            "PACKAGE_MANIFEST.md",
            "SUBMISSION_ARTIFACT_UPLOAD.md",
            "experiments/run_icde_experiments.py",
            "experiments/run_lanl_trace_experiment.py",
            "experiments/audit_submission_requirements.py",
            "experiments/package_submission.py",
        ]:
            require(rel in names, f"GitHub package missing critical file: {rel}")
            local_bytes = (ROOT / rel).read_bytes()
            package_bytes = zf.read(rel)
            require(package_bytes == local_bytes, f"GitHub package has stale critical file: {rel}")
            if rel.endswith((".tex", ".md", ".py")):
                package_text = package_bytes.decode("utf-8", errors="ignore")
                package_emails = re.findall(email_pattern, package_text)
                package_paths = re.findall(local_path_pattern, package_text)
                if anonymous_mode:
                    require(not package_emails, f"GitHub package {rel} contains email address(es): {package_emails}")
                require(not package_paths, f"GitHub package {rel} contains local absolute path marker(s): {package_paths}")
        for rel in [
            "experiments/results/frontier_production.csv",
            "experiments/results/frontier_production_summary.csv",
            "experiments/results/frontier_production_memory.csv",
            "experiments/results/shift_grid.csv",
            "experiments/results/shift_grid_sensitivity.csv",
            "experiments/results/shift_grid_worstcase.csv",
            "experiments/results/shift_dynamic_capacity.csv",
            "experiments/results/shift_dynamic_capacity_summary.csv",
            "experiments/results/traps_ablation.csv",
            "experiments/results/lanl_trace_conditioned.csv",
            "experiments/results/lanl_trace_denominators.csv",
            "experiments/results/lanl_trace_negative_controls.csv",
            "experiments/results/lanl_status_outcome.csv",
            "experiments/results/lanl_status_outcome_denominators.csv",
            "experiments/results/queueing_by_method.csv",
            "figure/icde_experiments/icde_frontier_production.pdf",
            "figure/icde_experiments/icde_shift_grid_heatmap.pdf",
            "figure/icde_experiments/icde_lanl_trace_suite.pdf",
            "figure/icde_experiments/icde_queueing_by_method.pdf",
        ]:
            require(rel in names, f"GitHub package missing Pro-required artifact: {rel}")
    require(not any("__pycache__" in n for n in names), "GitHub package contains Python bytecode")
    require(not any("_rendered_dashboards" in n for n in names), "GitHub package contains rendered dashboard scratch files")
    require(not any("dataset_a_dashboard" in n or "dataset_b_dashboard" in n or "dataset_c_dashboard" in n for n in names), "GitHub package contains old dashboard PDFs")
    require("context/Appendix.tex" not in names, "GitHub package contains appendix source")
    with zipfile.ZipFile(ROOT / "packages" / "ICDE_TRAPS_overleaf_minimal.zip") as zf:
        overleaf_names = set(zf.namelist())
        require(not any(re.findall(local_path_pattern, n) for n in overleaf_names), "Overleaf package entry names contain local absolute path markers")
        require(not any(re.findall(email_pattern, n) for n in overleaf_names), "Overleaf package entry names contain email-like strings")
        require(not any(n.endswith((".aux", ".log", ".fls", ".fdb_latexmk")) for n in overleaf_names), "Overleaf package contains build byproducts")
        require("context/Appendix.tex" not in overleaf_names, "Overleaf package contains appendix source")
        for rel in [
            "main.tex",
            "references.bib",
            "main.bbl",
            "context/Introduction.tex",
            "context/Experiments.tex",
            "context/Related Work.tex",
            "figure/icde_experiments/icde_frontier_production.pdf",
            "figure/icde_experiments/icde_shift_grid_heatmap.pdf",
            "figure/icde_experiments/icde_lanl_trace_suite.pdf",
            "figure/icde_experiments/icde_queueing_by_method.pdf",
        ]:
            require(rel in overleaf_names, f"Overleaf package missing critical file: {rel}")
            require(zf.read(rel) == (ROOT / rel).read_bytes(), f"Overleaf package has stale critical file: {rel}")


def main() -> int:
    checks = [
        audit_frontier,
        audit_shift_grid,
        audit_ablation,
        audit_lanl,
        audit_queueing,
        audit_manuscript_framing,
        audit_latex_and_packages,
    ]
    failures: list[str] = []
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as exc:
            failures.append(f"FAIL {check.__name__}: {exc}")
            print(failures[-1])
    if failures:
        return 1
    print("All submission requirement audits passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
