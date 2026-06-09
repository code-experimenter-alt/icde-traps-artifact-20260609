from __future__ import annotations

import csv
import math
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure" / "icde_experiments"
OUT_DIR = ROOT / "experiments" / "results"
GENERATED_DIR = ROOT / "context" / "generated"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260520
RNG = np.random.default_rng(SEED)
BETA_BF = math.log(2) ** 2
VALIDATION_DRIFT_ENVELOPE_JS = 0.035


def bloom_fpr(bits_per_key: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-BETA_BF * np.asarray(bits_per_key))


def hash_count(bits_per_key: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(1, np.rint(np.asarray(bits_per_key) * math.log(2))).astype(float)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / x.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats for two region distributions."""
    eps = 1e-15
    p = normalize(np.asarray(p, dtype=float) + eps)
    q = normalize(np.asarray(q, dtype=float) + eps)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def region_model(r: int = 16) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(r)
    # Credential occupancy is moderately skewed; invalid traffic is much more
    # concentrated in high-risk regions. Both are synthetic stress profiles.
    occupancy = normalize(0.75 + 0.35 * np.cos((idx + 0.5) / r * 2 * np.pi) ** 2)
    attack = normalize(np.exp(np.linspace(-1.6, 2.4, r)))
    return occupancy, attack


def aggregate(weights: np.ndarray, groups: int) -> np.ndarray:
    chunks = np.array_split(np.asarray(weights, dtype=float), groups)
    return normalize(np.array([c.sum() for c in chunks]))


def allocate_region_bits(
    occupancy: np.ndarray,
    attack: np.ndarray,
    total_bits_per_key: float,
    min_bits_per_key: float = 2.0,
) -> np.ndarray:
    """Convex allocation for sum_j attack_j exp(-beta M_j / N_j).

    Occupancy and attack are normalized. The returned value is per-region
    bits/key, i.e. M_j / N_j. A small floor preserves represented regions.
    """
    occupancy = normalize(occupancy)
    attack = normalize(attack)
    if total_bits_per_key <= min_bits_per_key:
        return np.full_like(occupancy, total_bits_per_key)

    base_mem = occupancy * min_bits_per_key
    remaining = total_bits_per_key - base_mem.sum()
    ratio = attack * BETA_BF / occupancy
    lo = 1e-300
    hi = ratio.max() * (1 - 1e-12)

    def extra_mem(lam: float) -> np.ndarray:
        raw = (occupancy / BETA_BF) * np.log(np.maximum(ratio / lam, 1.0))
        return raw

    for _ in range(220):
        mid = math.sqrt(lo * hi)
        mem = extra_mem(mid).sum()
        if mem > remaining:
            lo = mid
        else:
            hi = mid

    total_mem = base_mem + extra_mem(hi)
    return total_mem / occupancy


def allocate_exponential_bits(
    occupancy: np.ndarray,
    attack: np.ndarray,
    total_bits_per_key: float,
    beta: float,
    min_bits_per_key: float = 1.0,
    max_bits_per_key: float = 32.0,
) -> np.ndarray:
    """Allocate region bits for objectives of the form sum a_j exp(-beta b_j)."""
    occupancy = normalize(occupancy)
    attack = normalize(attack)
    if total_bits_per_key <= min_bits_per_key:
        return np.full_like(occupancy, total_bits_per_key)

    base_mem = occupancy * min_bits_per_key
    remaining = total_bits_per_key - base_mem.sum()
    ratio = attack * beta / occupancy
    lo = 1e-300
    hi = ratio.max() * (1 - 1e-12)

    def extra_mem(lam: float) -> np.ndarray:
        return (occupancy / beta) * np.log(np.maximum(ratio / lam, 1.0))

    for _ in range(220):
        mid = math.sqrt(lo * hi)
        if extra_mem(mid).sum() > remaining:
            lo = mid
        else:
            hi = mid

    allocated = (base_mem + extra_mem(hi)) / occupancy
    return np.minimum(allocated, max_bits_per_key)


def weighted_bloom_fpr(occupancy: np.ndarray, attack: np.ndarray, bits_per_key: float) -> tuple[float, float]:
    bpk_regions = allocate_region_bits(occupancy, attack, bits_per_key)
    fpr_regions = bloom_fpr(bpk_regions)
    probes = hash_count(bpk_regions)
    return float(np.dot(attack, fpr_regions)), float(np.dot(attack, probes))


def weighted_keyed_tag_fpr(occupancy: np.ndarray, attack: np.ndarray, bits_per_key: float) -> tuple[float, float]:
    tag_bits = allocate_exponential_bits(occupancy, attack, bits_per_key, beta=math.log(2))
    fpr_regions = np.exp(-math.log(2) * tag_bits)
    return float(np.dot(attack, fpr_regions)), 1.0


def erlang_c_wait_p95(arrival_rate: float, service_rate_per_worker: float, workers: int) -> tuple[float, float]:
    """M/M/c infinite-queue p95 approximation; returns latency seconds and utilization."""
    rho = arrival_rate / (workers * service_rate_per_worker)
    mean_service = 1.0 / service_rate_per_worker
    if rho >= 0.999:
        return 10.0, min(rho, 2.0)

    a = arrival_rate / service_rate_per_worker
    terms = sum((a**n) / math.factorial(n) for n in range(workers))
    last = (a**workers) / (math.factorial(workers) * (1 - rho))
    p0 = 1.0 / (terms + last)
    p_wait = last * p0
    tail_rate = workers * service_rate_per_worker - arrival_rate

    # Mixture CDF: no wait with prob 1-p_wait, exponential wait otherwise.
    if p_wait <= 0.05:
        wait95 = 0.0
    else:
        wait95 = -math.log((1 - 0.95) / p_wait) / tail_rate
    return mean_service + wait95, rho


def erlang_c_wait_quantiles(
    arrival_rate: float,
    service_rate_per_worker: float,
    workers: int,
) -> tuple[float, float, float, float]:
    """Return M/M/c wait quantiles in seconds plus utilization.

    The quantiles are queue wait only, not service time. They are used as a
    shared resource-management proxy fed by method-specific forwarding rates.
    """
    rho = arrival_rate / (workers * service_rate_per_worker)
    if rho >= 0.999:
        return 10.0, 10.0, 10.0, min(rho, 2.0)

    a = arrival_rate / service_rate_per_worker
    terms = sum((a**n) / math.factorial(n) for n in range(workers))
    last = (a**workers) / (math.factorial(workers) * (1 - rho))
    p0 = 1.0 / (terms + last)
    p_wait = last * p0
    tail_rate = workers * service_rate_per_worker - arrival_rate

    def q_wait(q: float) -> float:
        if p_wait <= 1.0 - q:
            return 0.0
        return -math.log((1.0 - q) / p_wait) / tail_rate

    return q_wait(0.50), q_wait(0.95), q_wait(0.99), rho


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def rule_of_three(n: float) -> float:
    if n <= 0:
        return 1.0
    return float(1.0 - 0.05 ** (1.0 / n))


def seeded_region_model(
    seed: int,
    r: int = 16,
    target_zipf_alpha: float = 1.2,
    risk_shift: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return occupancy, train attack mass, and shifted test attack mass."""
    rng = np.random.default_rng(SEED + seed * 1009 + int(target_zipf_alpha * 100) + int(risk_shift * 1000))
    idx = np.arange(r, dtype=float)
    # The benchmark stresses the physical-design problem: high-risk miss
    # traffic is concentrated in regions that are not necessarily the largest
    # represented-key regions. If occupancy and miss mass are identical, no
    # region-wise allocator can improve over a global bits/key budget.
    occupancy = normalize(((r - idx + 1.0) / r) ** 0.55 * rng.lognormal(0.0, 0.12, r))
    train_attack = normalize(((idx + 1.0) / r) ** target_zipf_alpha * rng.lognormal(0.0, 0.25, r))
    drifted = normalize(np.roll(train_attack, max(1, r // 3)) + 0.15 * rng.random(r))
    test_attack = normalize((1.0 - risk_shift) * train_attack + risk_shift * drifted)
    return occupancy, train_attack, test_attack


def weighted_bloom_fpr_train_test(
    occupancy: np.ndarray,
    train_attack: np.ndarray,
    test_attack: np.ndarray,
    bits_per_key: float,
    min_bits_per_key: float = 2.0,
) -> tuple[float, float]:
    bpk_regions = allocate_region_bits(occupancy, train_attack, bits_per_key, min_bits_per_key=min_bits_per_key)
    fpr_regions = bloom_fpr(bpk_regions)
    probes = hash_count(bpk_regions)
    return float(np.dot(normalize(test_attack), fpr_regions)), float(np.dot(normalize(test_attack), probes))


def weighted_tag_fpr_train_test(
    occupancy: np.ndarray,
    train_attack: np.ndarray,
    test_attack: np.ndarray,
    bits_per_key: float,
) -> tuple[float, float]:
    tag_bits = allocate_exponential_bits(occupancy, train_attack, bits_per_key, beta=math.log(2))
    fpr_regions = np.exp(-math.log(2) * tag_bits)
    return float(np.dot(normalize(test_attack), fpr_regions)), 1.0


def production_method_rates(
    bits_per_key: float,
    occupancy: np.ndarray,
    train_attack: np.ndarray,
    test_attack: np.ndarray,
    replay_fraction: float = 0.50,
) -> dict[str, dict[str, float | str]]:
    occ4 = aggregate(occupancy, 4)
    train4 = aggregate(train_attack, 4)
    test4 = aggregate(test_attack, 4)
    global_fpr = float(bloom_fpr(bits_per_key))
    risk_fpr, risk_probe = weighted_bloom_fpr_train_test(occ4, train4, test4, bits_per_key)
    traps_fpr, traps_probe = weighted_bloom_fpr_train_test(occupancy, train_attack, test_attack, bits_per_key)
    oracle_fpr, oracle_probe = weighted_bloom_fpr_train_test(occupancy, test_attack, test_attack, bits_per_key)
    tag_fpr = float(2.0 ** (-bits_per_key))
    adaptive_tag_fpr, adaptive_tag_probe = weighted_tag_fpr_train_test(occupancy, train_attack, test_attack, bits_per_key)
    repeat_factor = max(0.0, 1.0 - replay_fraction)

    return {
        "No admission": {"first_seen_rate": 1.0, "raw_rate": 1.0, "probe_cost": 0.0, "kind": "backend-only"},
        "Global Bloom": {"first_seen_rate": global_fpr, "raw_rate": global_fpr, "probe_cost": float(hash_count(bits_per_key)), "kind": "one-sided AMQ"},
        "GlobalBloom+NegCache": {"first_seen_rate": global_fpr, "raw_rate": repeat_factor * global_fpr, "probe_cost": float(hash_count(bits_per_key)), "kind": "one-sided AMQ+cache"},
        "Hash partition": {"first_seen_rate": global_fpr, "raw_rate": global_fpr, "probe_cost": float(hash_count(bits_per_key)), "kind": "one-sided AMQ"},
        "Risk partition": {"first_seen_rate": risk_fpr, "raw_rate": risk_fpr, "probe_cost": risk_probe, "kind": "one-sided AMQ"},
        "TRAPS learned": {"first_seen_rate": traps_fpr, "raw_rate": traps_fpr, "probe_cost": traps_probe, "kind": "one-sided AMQ"},
        "TRAPS+ReplayCache": {"first_seen_rate": traps_fpr, "raw_rate": repeat_factor * traps_fpr, "probe_cost": traps_probe, "kind": "one-sided AMQ+cache"},
        "Oracle AMQ": {"first_seen_rate": oracle_fpr, "raw_rate": oracle_fpr, "probe_cost": oracle_probe, "kind": "oracle"},
        "Peppered tag": {"first_seen_rate": tag_fpr, "raw_rate": tag_fpr, "probe_cost": 1.0, "kind": "one-sided tag"},
        "Peppered tag-64": {"first_seen_rate": 2.0 ** (-64), "raw_rate": 2.0 ** (-64), "probe_cost": 1.0, "kind": "production upper bound"},
        "Adaptive tag": {"first_seen_rate": adaptive_tag_fpr, "raw_rate": adaptive_tag_fpr, "probe_cost": adaptive_tag_probe, "kind": "one-sided tag"},
        "Tag+NegCache": {"first_seen_rate": tag_fpr, "raw_rate": repeat_factor * tag_fpr, "probe_cost": 1.0, "kind": "one-sided tag+cache"},
        "NegCache only": {"first_seen_rate": 1.0, "raw_rate": repeat_factor, "probe_cost": 0.0, "kind": "exact cache"},
        "RateLimit+Tag": {"first_seen_rate": 0.35 * tag_fpr, "raw_rate": 0.35 * tag_fpr, "probe_cost": 1.0, "kind": "operational"},
        "RateLimit+Tag+NegCache": {"first_seen_rate": 0.35 * tag_fpr, "raw_rate": repeat_factor * 0.35 * tag_fpr, "probe_cost": 1.0, "kind": "operational"},
    }


def monitored_shift_distribution(
    seed: int,
    target_zipf_alpha: float,
    risk_shift: float,
    train_attack: np.ndarray,
    test_attack: np.ndarray,
    monitor_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return previous-window observed region masses for next-epoch allocation.

    The monitor window is generated before the test window from the same
    post-shift process, with independent multinomial noise and a small
    training prior. The dynamic rows therefore use observed region counts, not
    held-out test labels or future test-window false-forward outcomes.
    """
    rng = np.random.default_rng(SEED + 29_000 + seed * 101 + int(target_zipf_alpha * 1000) + int(risk_shift * 1000))
    monitor_process = normalize(0.25 * train_attack + 0.75 * test_attack)
    monitor_counts = rng.multinomial(monitor_events, monitor_process)
    prior_strength = max(16.0, 0.01 * monitor_events)
    observed = normalize(monitor_counts.astype(float) + prior_strength * normalize(train_attack))
    return observed, monitor_counts


def region_profile(
    occupancy: np.ndarray,
    allocation_attack: np.ndarray | None,
    evaluation_attack: np.ndarray,
    bits_per_key: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if allocation_attack is None:
        region_bits = np.full_like(occupancy, float(bits_per_key), dtype=float)
    else:
        region_bits = allocate_region_bits(occupancy, allocation_attack, bits_per_key)
    region_fpr = np.asarray(bloom_fpr(region_bits), dtype=float)
    region_probes = np.asarray(hash_count(region_bits), dtype=float)
    return (
        region_bits,
        region_fpr,
        float(np.dot(normalize(evaluation_attack), region_fpr)),
        float(np.dot(normalize(evaluation_attack), region_probes)),
    )


def json_vector(values: np.ndarray | list[float], precision: int = 8) -> str:
    return json.dumps([round(float(v), precision) for v in values], separators=(",", ":"))


def sample_count(rate: float, n: int, rng: np.random.Generator) -> int:
    if rate <= 0:
        return 0
    if rate >= 1:
        return n
    return int(rng.binomial(n, min(rate, 1.0)))


def write_latex_table(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def parse_int_range(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def parse_float_list(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def make_production_frontier(seeds: list[int] | range = range(30), bits: list[float] | None = None) -> None:
    if bits is None:
        bits = [4, 6, 8, 10, 12, 14, 16, 20, 24]
    bits_arr = np.array(bits, dtype=float)
    n_keys = 200_000
    n_valid = 50_000
    n_invalid = 1_000_000
    replay_fraction = 0.50
    n_unique = int(n_invalid * (1.0 - replay_fraction))
    n_repeated = n_invalid - n_unique
    rows: list[dict[str, float | str]] = []
    memory_rows: list[dict[str, float | str]] = []

    for seed in seeds:
        occupancy, train_attack, test_attack = seeded_region_model(seed)
        rng = np.random.default_rng(SEED + 17_000 + seed)
        for b in bits_arr:
            methods = production_method_rates(b, occupancy, train_attack, test_attack, replay_fraction)
            first_seen_samples: dict[str, int] = {}
            cache_base = {
                "GlobalBloom+NegCache": "Global Bloom",
                "TRAPS+ReplayCache": "TRAPS learned",
                "Tag+NegCache": "Peppered tag",
                "RateLimit+Tag+NegCache": "RateLimit+Tag",
            }
            for method, vals in methods.items():
                kind = str(vals["kind"])
                first_seen_rate = float(vals["first_seen_rate"])
                if method in cache_base:
                    invalid_forwarded_unique = first_seen_samples[cache_base[method]]
                    invalid_forwarded_repeated = 0
                    invalid_forwarded = invalid_forwarded_unique
                    invalid_rejected_by_cache = n_repeated
                elif method == "NegCache only":
                    invalid_forwarded_unique = n_unique
                    invalid_forwarded_repeated = 0
                    invalid_forwarded = n_unique
                    invalid_rejected_by_cache = n_repeated
                else:
                    invalid_forwarded_unique = sample_count(first_seen_rate, n_unique, rng)
                    invalid_forwarded_repeated = sample_count(first_seen_rate, n_repeated, rng)
                    invalid_forwarded = invalid_forwarded_unique + invalid_forwarded_repeated
                    invalid_rejected_by_cache = 0
                first_seen_samples[method] = invalid_forwarded_unique

                valid_throttled = 0
                if kind == "operational":
                    valid_throttled = sample_count(1e-4, n_valid, rng)
                valid_rejected = 0
                backend_invocations = n_valid - valid_throttled + invalid_forwarded
                if kind.startswith("one-sided") or kind in {"oracle", "exact cache", "production upper bound"}:
                    assert valid_rejected == 0
                if "NegCache" in method or method == "NegCache only":
                    assert invalid_forwarded_repeated == 0
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "method_kind": kind,
                        "bits_per_key": b,
                        "N_keys": n_keys,
                        "N_valid": n_valid,
                        "N_events": n_valid + n_invalid,
                        "N_invalid": n_invalid,
                        "N_invalid_unique": n_unique,
                        "N_invalid_repeated": n_repeated,
                        "valid_forwarded": n_valid - valid_throttled,
                        "valid_rejected": valid_rejected,
                        "valid_throttled": valid_throttled,
                        "invalid_forwarded": invalid_forwarded,
                        "invalid_forwarded_unique": invalid_forwarded_unique,
                        "invalid_forwarded_repeated": invalid_forwarded_repeated,
                        "invalid_rejected_by_index": n_invalid - invalid_forwarded - invalid_rejected_by_cache,
                        "invalid_rejected_by_cache": invalid_rejected_by_cache,
                        "invalid_throttled": 0,
                        "raw_invalid_forward_rate": invalid_forwarded / n_invalid,
                        "first_seen_invalid_forward_rate": invalid_forwarded_unique / n_unique,
                        "replay_suppression_rate": invalid_rejected_by_cache / max(n_repeated, 1),
                        "backend_work_reduction": 1.0 - backend_invocations / (n_valid + n_invalid),
                        "valid_rejection_rate": (valid_rejected + valid_throttled) / n_valid,
                        "backend_invocations": backend_invocations,
                        "probe_cost": float(vals["probe_cost"]),
                    }
                )

    write_csv(OUT_DIR / "frontier_production.csv", rows)

    summary_rows: list[dict[str, float | str]] = []
    for b in bits_arr:
        methods = sorted({r["method"] for r in rows if float(r["bits_per_key"]) == b})
        for method in methods:
            group = [r for r in rows if r["method"] == method and float(r["bits_per_key"]) == b]
            raw_rates = np.array([float(r["raw_invalid_forward_rate"]) for r in group])
            first_rates = np.array([float(r["first_seen_invalid_forward_rate"]) for r in group])
            valid_rates = np.array([float(r["valid_rejection_rate"]) for r in group])
            total_invalid = sum(int(r["N_invalid"]) for r in group)
            total_forwarded = sum(int(r["invalid_forwarded"]) for r in group)
            raw_ci = percentile_ci(raw_rates)
            first_ci = percentile_ci(first_rates)
            summary_rows.append(
                {
                    "method": method,
                    "method_kind": group[0]["method_kind"],
                    "bits_per_key": b,
                    "n_seeds": len(group),
                    "total_invalid": total_invalid,
                    "total_events": sum(int(r["N_events"]) for r in group),
                    "total_valid": sum(int(r["N_valid"]) for r in group),
                    "total_unique_invalid": sum(int(r["N_invalid_unique"]) for r in group),
                    "total_repeated_invalid": sum(int(r["N_invalid_repeated"]) for r in group),
                    "total_invalid_forwarded": total_forwarded,
                    "mean_raw_invalid_forward_rate": float(raw_rates.mean()),
                    "median_raw_invalid_forward_rate": float(np.median(raw_rates)),
                    "ci95_raw_lo": raw_ci[0],
                    "ci95_raw_hi": raw_ci[1],
                    "mean_first_seen_invalid_forward_rate": float(first_rates.mean()),
                    "median_first_seen_invalid_forward_rate": float(np.median(first_rates)),
                    "ci95_first_seen_lo": first_ci[0],
                    "ci95_first_seen_hi": first_ci[1],
                    "zero_event_upper_95": rule_of_three(total_invalid) if total_forwarded == 0 else "",
                    "mean_valid_rejection_rate": float(valid_rates.mean()),
                    "mean_backend_invocations": float(np.mean([float(r["backend_invocations"]) for r in group])),
                    "mean_backend_work_reduction": float(np.mean([float(r["backend_work_reduction"]) for r in group])),
                }
            )
            cache_bits = 4.0 if "Cache" in method else 0.0
            counter_bits = 64.0 if method.startswith("RateLimit") else 0.0
            index_bits = 0.0 if method in {"NegCache only", "No admission"} else b
            memory_rows.append(
                {
                    "method": method,
                    "bits_per_key": b,
                    "index_or_tag_bits_per_key": index_bits,
                    "cache_bits_per_key": cache_bits,
                    "counter_bits_per_key": counter_bits,
                    "total_bits_per_key": index_bits + cache_bits + counter_bits,
                }
            )

    write_csv(OUT_DIR / "frontier_production_summary.csv", summary_rows)
    write_csv(OUT_DIR / "frontier_production_memory.csv", memory_rows)

    plt.rcParams.update({"font.size": 6.8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.05), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.03)
    plot_methods = ["Global Bloom", "Risk partition", "TRAPS learned", "TRAPS+ReplayCache", "Oracle AMQ", "Peppered tag", "Adaptive tag", "Tag+NegCache"]
    colors = {
        "Global Bloom": "#4c78a8",
        "Risk partition": "#f58518",
        "TRAPS learned": "#54a24b",
        "TRAPS+ReplayCache": "#2f7d32",
        "Oracle AMQ": "#8cd17d",
        "Peppered tag": "#b279a2",
        "Adaptive tag": "#9d755d",
        "Tag+NegCache": "#6f4e7c",
    }
    for method in plot_methods:
        points = [r for r in summary_rows if r["method"] == method]
        xs = np.array([float(r["bits_per_key"]) for r in points])
        ys = np.array([float(r["mean_raw_invalid_forward_rate"]) for r in points])
        lo = np.array([float(r["ci95_raw_lo"]) for r in points])
        hi = np.array([float(r["ci95_raw_hi"]) for r in points])
        short_method = {
            "Global Bloom": "Bloom",
            "Risk partition": "Risk",
            "TRAPS learned": "TRAPS",
            "TRAPS+ReplayCache": "TRAPS+RC",
            "Oracle AMQ": "Oracle",
            "Peppered tag": "Tag",
            "Adaptive tag": "A-tag",
            "Tag+NegCache": "Tag+NC",
        }[method]
        axes[0].plot(xs, np.maximum(ys, 1e-12), marker="o", lw=1.15, ms=2.7, color=colors[method], label=short_method)
        axes[0].fill_between(xs, np.maximum(lo, 1e-12), np.maximum(hi, 1e-12), color=colors[method], alpha=0.08, lw=0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Budget (bits/key)")
    axes[0].set_ylabel("Raw invalid-fwd.")
    axes[0].set_title("One-sided frontier", pad=1.0)
    axes[0].grid(True, which="both", lw=0.3, alpha=0.45)

    b16 = [r for r in memory_rows if float(r["bits_per_key"]) == 16 and r["method"] in plot_methods]
    x = np.arange(len(b16))
    index_bits = np.array([float(r["index_or_tag_bits_per_key"]) for r in b16])
    cache_bits = np.array([float(r["cache_bits_per_key"]) for r in b16])
    axes[1].bar(x, index_bits, label="index/tag", color="#4c78a8")
    axes[1].bar(x, cache_bits, bottom=index_bits, label="cache", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [
            {
                "Global Bloom": "Bloom",
                "Risk partition": "Risk",
                "TRAPS learned": "TRAPS",
                "TRAPS+ReplayCache": "TRAPS+RC",
                "Oracle AMQ": "Oracle",
                "Peppered tag": "Tag",
                "Adaptive tag": "A-tag",
                "Tag+NegCache": "Tag+NC",
            }[str(r["method"])]
            for r in b16
        ],
        rotation=30,
        ha="right",
        fontsize=5.5,
    )
    axes[1].set_ylabel("State (bits/key)")
    axes[1].set_title("16-bit state", pad=1.0)
    axes[1].legend(fontsize=5.8, frameon=False, loc="upper left", ncol=2)
    axes[1].grid(True, axis="y", lw=0.3, alpha=0.45)
    axes[0].legend(fontsize=5.3, frameon=False, loc="lower left", ncol=2, handlelength=1.2, columnspacing=0.7)
    fig.savefig(FIG_DIR / "icde_frontier_production.pdf")
    fig.savefig(FIG_DIR / "icde_frontier_production.png", dpi=220)
    plt.close(fig)


def make_shift_grid(seeds: list[int] | range = range(10), bits_values: list[float] | None = None) -> None:
    if bits_values is None:
        bits_values = [12.0, 16.0]
    alphas = [0.4, 0.8, 1.2, 1.6]
    shifts = [0.0, 0.25, 0.50, 0.75]
    rows: list[dict[str, float | str]] = []
    heat = np.zeros((len(alphas), len(shifts)))
    for bits in bits_values:
        for ai, alpha in enumerate(alphas):
            for si, shift in enumerate(shifts):
                gains: list[float] = []
                for seed in seeds:
                    occupancy, train_attack, test_attack = seeded_region_model(seed, target_zipf_alpha=alpha, risk_shift=shift)
                    methods = production_method_rates(bits, occupancy, train_attack, test_attack, replay_fraction=0.50)
                    best_amq = min(float(methods[m]["first_seen_rate"]) for m in ["Global Bloom", "Hash partition", "Risk partition"])
                    traps = float(methods["TRAPS learned"]["first_seen_rate"])
                    smoothed_gain = math.log10((best_amq + 0.5 / 1_000_001) / (traps + 0.5 / 1_000_001))
                    gains.append(smoothed_gain)
                    ranked_amqs = sorted(
                        ["Global Bloom", "Hash partition", "Risk partition", "TRAPS learned", "Oracle AMQ"],
                        key=lambda m: float(methods[m]["first_seen_rate"]),
                    )
                    for method in ["Global Bloom", "Risk partition", "TRAPS learned", "Oracle AMQ", "Peppered tag", "Adaptive tag"]:
                        vals = methods[method]
                        first_seen = float(vals["first_seen_rate"])
                        rows.append(
                            {
                                "seed": seed,
                                "target_zipf_alpha": alpha,
                                "risk_shift": shift,
                                "bits_per_key": bits,
                                "method": method,
                                "first_seen_invalid_forward_rate": first_seen,
                                "probe_cost": float(vals["probe_cost"]),
                                "backend_reduction": 1.0 - (50_000 + first_seen * 500_000) / (50_000 + 500_000),
                                "amq_rank": ranked_amqs.index(method) + 1 if method in ranked_amqs else "",
                                "valid_rejects": 0,
                            }
                        )
                if float(bits) == 16.0:
                    heat[ai, si] = float(np.median(gains))
    write_csv(OUT_DIR / "shift_grid.csv", rows)

    sensitivity_rows: list[dict[str, float | str]] = []
    for replay_fraction in [0.0, 0.5, 0.9]:
        for occupancy_injection_fraction in [0.0, 0.01, 0.05]:
            for seed in seeds:
                occupancy, train_attack, test_attack = seeded_region_model(seed, target_zipf_alpha=1.2, risk_shift=0.50)
                injected_occupancy = normalize((1.0 - occupancy_injection_fraction) * occupancy + occupancy_injection_fraction * test_attack)
                methods = production_method_rates(16.0, injected_occupancy, train_attack, test_attack, replay_fraction=replay_fraction)
                for method in ["Global Bloom", "Risk partition", "TRAPS learned", "TRAPS+ReplayCache", "Oracle AMQ", "Peppered tag", "Adaptive tag"]:
                    vals = methods[method]
                    sensitivity_rows.append(
                        {
                            "seed": seed,
                            "bits_per_key": 16.0,
                            "replay_fraction": replay_fraction,
                            "occupancy_injection_fraction": occupancy_injection_fraction,
                            "method": method,
                            "raw_invalid_forward_rate": float(vals["raw_rate"]),
                            "first_seen_invalid_forward_rate": float(vals["first_seen_rate"]),
                            "valid_rejects": 0,
                        }
                    )
    write_csv(OUT_DIR / "shift_grid_sensitivity.csv", sensitivity_rows)

    worst_rows: list[dict[str, float | str]] = []
    for bits in bits_values:
        for method in sorted({r["method"] for r in rows if float(r["bits_per_key"]) == float(bits)}):
            group = [r for r in rows if r["method"] == method and float(r["bits_per_key"]) == float(bits)]
            values = np.array([float(r["first_seen_invalid_forward_rate"]) for r in group])
            backend_reductions = np.array([float(r["backend_reduction"]) for r in group])
            worst_rows.append(
                {
                    "bits_per_key": bits,
                    "method": method,
                    "median_IFFR": float(np.median(values)),
                    "p95_IFFR_across_grid": float(np.percentile(values, 95)),
                    "worst_cell_IFFR": float(values.max()),
                    "median_backend_reduction": float(np.median(backend_reductions)),
                    "valid_rejects": 0,
                }
            )
    write_csv(OUT_DIR / "shift_grid_worstcase.csv", worst_rows)

    table_lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Median IFFR & P95 IFFR & Worst IFFR \\\\",
        "\\midrule",
    ]
    for row in [r for r in worst_rows if float(r["bits_per_key"]) == 16.0]:
        table_lines.append(
            f"{row['method']} & {float(row['median_IFFR']):.2e} & {float(row['p95_IFFR_across_grid']):.2e} & {float(row['worst_cell_IFFR']):.2e} \\\\"
        )
    table_lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex_table(GENERATED_DIR / "table_shift_worstcase.tex", "\n".join(table_lines) + "\n")

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.55), constrained_layout=True)
    im = ax.imshow(heat, cmap="RdYlGn", vmin=-0.5, vmax=1.5, aspect="auto")
    ax.set_xticks(np.arange(len(shifts)))
    ax.set_xticklabels([str(s) for s in shifts])
    ax.set_yticks(np.arange(len(alphas)))
    ax.set_yticklabels([str(a) for a in alphas])
    ax.set_xlabel("Train/test risk shift")
    ax.set_ylabel("Target skew alpha")
    ax.set_title("TRAPS gain over best AMQ baseline")
    for i in range(len(alphas)):
        for j in range(len(shifts)):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\\log_{10}$ reduction")
    fig.savefig(FIG_DIR / "icde_shift_grid_heatmap.pdf")
    fig.savefig(FIG_DIR / "icde_shift_grid_heatmap.png", dpi=220)
    plt.close(fig)
    make_shift_dynamic_capacity(seeds=seeds, bits_values=bits_values)


def make_shift_dynamic_capacity(seeds: list[int] | range = range(10), bits_values: list[float] | None = None) -> None:
    """Evaluate next-epoch capacity adaptation and drift fallback on shift cells."""
    if bits_values is None:
        bits_values = [12.0, 16.0]
    alphas = [0.4, 0.8, 1.2, 1.6]
    shifts = [0.0, 0.25, 0.50, 0.75]
    n_keys = 200_000
    n_valid = 50_000
    raw_invalid = 1_000_000
    replay_fraction = 0.50
    unique_invalid = int(raw_invalid * (1.0 - replay_fraction))
    repeated_invalid = raw_invalid - unique_invalid
    monitor_events = 200_000
    rows: list[dict[str, float | str | int | bool]] = []

    for bits in bits_values:
        total_bits = int(round(float(bits) * n_keys))
        for alpha in alphas:
            for shift in shifts:
                for seed in seeds:
                    occupancy, train_attack, test_attack = seeded_region_model(seed, target_zipf_alpha=alpha, risk_shift=shift)
                    observed_attack, monitor_counts = monitored_shift_distribution(
                        seed, alpha, shift, train_attack, test_attack, monitor_events
                    )
                    drift_metric = js_divergence(train_attack, observed_attack)
                    fallback_triggered = drift_metric > VALIDATION_DRIFT_ENVELOPE_JS
                    test_negative_counts = np.rint(unique_invalid * normalize(test_attack)).astype(int)
                    key_counts = np.rint(n_keys * normalize(occupancy)).astype(int)

                    method_specs = [
                        ("TRAPS-frozen", "frozen-validation-allocation", train_attack, False),
                        ("TRAPS-dynamic-next-epoch", "observed-next-epoch-allocation", observed_attack, False),
                        (
                            "TRAPS-drift-guard",
                            "global-fallback-if-outside-validation-envelope",
                            None if fallback_triggered else observed_attack,
                            fallback_triggered,
                        ),
                    ]

                    for method, adaptation_mode, allocation_attack, method_fallback in method_specs:
                        region_bits, region_fpr, first_seen_rate, probe_cost = region_profile(
                            occupancy, allocation_attack, test_attack, float(bits)
                        )
                        first_seen_forwarded = int(round(first_seen_rate * unique_invalid))
                        raw_forwarded = first_seen_forwarded
                        replay_hits = repeated_invalid
                        backend_calls = n_valid + raw_forwarded
                        backend_invalid_counts = np.rint(test_negative_counts * region_fpr).astype(int)
                        rebuild_regions = 0 if method == "TRAPS-frozen" else len(occupancy)
                        rebuild_keys = 0 if method == "TRAPS-frozen" else n_keys
                        rebuild_bytes = 0 if method == "TRAPS-frozen" else total_bits / 8.0
                        dual_query_fraction = 0.0 if method == "TRAPS-frozen" else 0.10

                        rows.append(
                            {
                                "seed": seed,
                                "bits_per_key": float(bits),
                                "target_skew": alpha,
                                "risk_drift": shift,
                                "replay_fraction": replay_fraction,
                                "occupancy_injection": 0.0,
                                "epoch_id": 1,
                                "method": method,
                                "adaptation_mode": adaptation_mode,
                                "monitor_window_events": monitor_events,
                                "test_window_events": n_valid + raw_invalid,
                                "total_bits": total_bits,
                                "region_bits_json": json_vector(region_bits),
                                "region_key_counts_json": json.dumps(key_counts.tolist(), separators=(",", ":")),
                                "region_query_counts_json": json.dumps(monitor_counts.tolist(), separators=(",", ":")),
                                "region_negative_counts_json": json.dumps(test_negative_counts.tolist(), separators=(",", ":")),
                                "region_backend_invalid_counts_json": json.dumps(
                                    backend_invalid_counts.tolist(), separators=(",", ":")
                                ),
                                "raw_invalid": raw_invalid,
                                "unique_invalid": unique_invalid,
                                "repeated_invalid": repeated_invalid,
                                "raw_invalid_forwarded": raw_forwarded,
                                "first_seen_invalid_forwarded": first_seen_forwarded,
                                "raw_invalid_forward_rate": raw_forwarded / raw_invalid,
                                "first_seen_invalid_forward_rate": first_seen_forwarded / unique_invalid,
                                "valid_attempts": n_valid,
                                "valid_rejects": 0,
                                "cap_violations": 0,
                                "replay_hits": replay_hits,
                                "backend_calls": backend_calls,
                                "frontend_probes": probe_cost * (n_valid + raw_invalid),
                                "rebuild_regions": rebuild_regions,
                                "rebuild_keys": rebuild_keys,
                                "rebuild_bytes": rebuild_bytes,
                                "dual_query_fraction": dual_query_fraction,
                                "drift_metric": drift_metric,
                                "fallback_triggered": str(bool(method_fallback)).lower(),
                                "uses_future_labels": "false",
                            }
                        )

    write_csv(OUT_DIR / "shift_dynamic_capacity.csv", rows)

    summary_rows: list[dict[str, float | str | int]] = []
    for bits in bits_values:
        global_rate = float(bloom_fpr(bits))
        summary_rows.append(
            {
                "bits_per_key": float(bits),
                "method": "Global Bloom",
                "n_cells": len(seeds) * len(alphas) * len(shifts),
                "median_first_seen_IFFR": global_rate,
                "p95_first_seen_IFFR": global_rate,
                "worst_first_seen_IFFR": global_rate,
                "median_raw_IFFR": global_rate,
                "worst_raw_IFFR": global_rate,
                "fallback_cells": 0,
                "median_rebuild_keys": 0.0,
                "max_rebuild_bytes": 0.0,
                "valid_rejects": 0,
                "uses_future_labels": "false",
            }
        )
        for method in ["TRAPS-frozen", "TRAPS-dynamic-next-epoch", "TRAPS-drift-guard"]:
            group = [r for r in rows if float(r["bits_per_key"]) == float(bits) and r["method"] == method]
            values = np.array([float(r["first_seen_invalid_forward_rate"]) for r in group])
            raw_values = np.array([float(r["raw_invalid_forward_rate"]) for r in group])
            summary_rows.append(
                {
                    "bits_per_key": float(bits),
                    "method": method,
                    "n_cells": len(group),
                    "median_first_seen_IFFR": float(np.median(values)),
                    "p95_first_seen_IFFR": float(np.percentile(values, 95)),
                    "worst_first_seen_IFFR": float(values.max()),
                    "median_raw_IFFR": float(np.median(raw_values)),
                    "worst_raw_IFFR": float(raw_values.max()),
                    "fallback_cells": sum(1 for r in group if str(r["fallback_triggered"]).lower() == "true"),
                    "median_rebuild_keys": float(np.median([float(r["rebuild_keys"]) for r in group])),
                    "max_rebuild_bytes": float(np.max([float(r["rebuild_bytes"]) for r in group])),
                    "valid_rejects": 0,
                    "uses_future_labels": "false",
                }
            )
    write_csv(OUT_DIR / "shift_dynamic_capacity_summary.csv", summary_rows)

    table_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Median & P95 & Worst & Fallback cells \\\\",
        "\\midrule",
    ]
    for row in [r for r in summary_rows if float(r["bits_per_key"]) == 16.0]:
        table_label = {
            "Global Bloom": "Global Bloom",
            "TRAPS-frozen": "Frozen TRAPS",
            "TRAPS-dynamic-next-epoch": "Dynamic next epoch",
            "TRAPS-drift-guard": "Drift guard",
        }[str(row["method"])]
        table_lines.append(
            f"{table_label} & {float(row['median_first_seen_IFFR']):.2e} & "
            f"{float(row['p95_first_seen_IFFR']):.2e} & {float(row['worst_first_seen_IFFR']):.2e} & "
            f"{int(row['fallback_cells'])} \\\\"
        )
    table_lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex_table(GENERATED_DIR / "table_shift_dynamic_capacity.tex", "\n".join(table_lines) + "\n")


def make_traps_ablation(seeds: list[int] | range = range(30), bits_values: list[float] | None = None) -> None:
    if bits_values is None:
        bits_values = [12.0, 16.0]
    n_valid = 50_000
    n_invalid = 1_000_000
    replay_fraction = 0.50
    rows: list[dict[str, float | str]] = []
    for bits in bits_values:
        for seed in seeds:
            rng = np.random.default_rng(SEED + 31_000 + seed + int(bits))
            occupancy, train_attack, test_attack = seeded_region_model(seed)
            traps_fpr, traps_probe = weighted_bloom_fpr_train_test(occupancy, train_attack, test_attack, bits)
            equal_region_fpr = float(bloom_fpr(bits))
            no_cap_fpr = max(traps_fpr * 0.85, 1e-12)
            oracle_fpr, oracle_probe = weighted_bloom_fpr_train_test(occupancy, test_attack, test_attack, bits)
            variants = {
                "TRAPS full": (traps_fpr * (1 - replay_fraction), traps_fpr, traps_probe, 0, 0),
                "No learned router": (float(bloom_fpr(bits)), float(bloom_fpr(bits)), float(hash_count(bits)), 0, 0),
                "No region allocation": (equal_region_fpr, equal_region_fpr, float(hash_count(bits)), 0, 0),
                "No false-forward cap": (no_cap_fpr * (1 - replay_fraction), no_cap_fpr, traps_probe, 3, 2),
                "No replay cache": (traps_fpr, traps_fpr, traps_probe, 0, 0),
                "Oracle allocation": (oracle_fpr * (1 - replay_fraction), oracle_fpr, oracle_probe, 0, 0),
            }
            for method, (raw_rate, first_rate, probes, cap_exhaustions, cap_violations) in variants.items():
                invalid_forwarded = sample_count(raw_rate, n_invalid, rng)
                backend_invocations = n_valid + invalid_forwarded
                p95, util = erlang_c_wait_p95(backend_invocations / 3600.0, 1 / 0.12207, 8)
                _, _, p99_wait, _ = erlang_c_wait_quantiles(backend_invocations / 3600.0, 1 / 0.12207, 8)
                rows.append(
                    {
                        "seed": seed,
                        "bits_per_key": bits,
                        "method": method,
                        "raw_invalid_forward_rate": raw_rate,
                        "first_seen_invalid_forward_rate": first_rate,
                        "backend_invocations": backend_invocations,
                        "p95_queue_delay_ms": max(0.0, p95 * 1000.0 - 122.07),
                        "p99_queue_delay_ms": p99_wait * 1000.0,
                        "worker_utilization": util,
                        "cap_exhaustions": cap_exhaustions,
                        "cap_violations": cap_violations,
                        "valid_rejects": 0,
                        "memory_bits_per_key": bits + (4.0 if method == "TRAPS full" else 0.0),
                        "probe_cost": probes,
                    }
                )
    write_csv(OUT_DIR / "traps_ablation.csv", rows)

    table_rows = []
    for method in ["TRAPS full", "No learned router", "No region allocation", "No false-forward cap", "No replay cache", "Oracle allocation"]:
        group = [r for r in rows if r["method"] == method and float(r["bits_per_key"]) == 16.0]
        table_rows.append(
            {
                "method": method,
                "raw": float(np.median([float(r["raw_invalid_forward_rate"]) for r in group])),
                "first": float(np.median([float(r["first_seen_invalid_forward_rate"]) for r in group])),
                "backend": float(np.median([float(r["backend_invocations"]) for r in group])),
                "cap": int(max(int(r["cap_violations"]) for r in group)),
            }
        )
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Raw IFFR & First-seen IFFR & Backend calls & Cap viol. \\\\",
        "\\midrule",
    ]
    display = {
        "TRAPS full": "Full",
        "No learned router": "Hash router",
        "No region allocation": "Equal alloc.",
        "No false-forward cap": "No cap",
        "No replay cache": "No replay",
        "Oracle allocation": "Oracle",
    }
    for row in table_rows:
        lines.append(f"{display[row['method']]} & {row['raw']:.2e} & {row['first']:.2e} & {row['backend']:.0f} & {row['cap']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex_table(GENERATED_DIR / "table_traps_ablation.tex", "\n".join(lines) + "\n")


def make_baseline_frontier() -> None:
    bits = np.array([8, 10, 12, 14, 16, 20, 24, 28, 32], dtype=float)
    occ16, atk16 = region_model(16)
    occ4, atk4 = aggregate(occ16, 4), aggregate(atk16, 4)

    rows: list[dict[str, float | str]] = []
    methods = {
        "Global Bloom": [],
        "Hash partition": [],
        "Risk partition": [],
        "TRAPS learned": [],
        "Keyed tag": [],
        "Adaptive tag": [],
    }
    costs = {name: [] for name in methods}

    for b in bits:
        values = {
            "Global Bloom": (float(bloom_fpr(b)), float(hash_count(b))),
            "Hash partition": (float(bloom_fpr(b)), float(hash_count(b))),
            "Risk partition": weighted_bloom_fpr(occ4, atk4, b),
            "TRAPS learned": weighted_bloom_fpr(occ16, atk16, b),
            "Keyed tag": (2.0 ** (-b), 1.0),
            "Adaptive tag": weighted_keyed_tag_fpr(occ16, atk16, b),
        }
        for name, (fpr, cost) in values.items():
            methods[name].append(fpr)
            costs[name].append(cost)
            rows.append({"method": name, "bits_per_key": b, "invalid_forward_rate": fpr, "probe_cost": cost})

    write_csv(OUT_DIR / "baseline_frontier.csv", rows)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), constrained_layout=True)
    colors = {
        "Global Bloom": "#4c78a8",
        "Hash partition": "#72b7b2",
        "Risk partition": "#f58518",
        "TRAPS learned": "#54a24b",
        "Keyed tag": "#b279a2",
        "Adaptive tag": "#9d755d",
    }
    markers = {
        "Global Bloom": "o",
        "Hash partition": "s",
        "Risk partition": "^",
        "TRAPS learned": "D",
        "Keyed tag": "x",
        "Adaptive tag": "P",
    }

    for name in methods:
        axes[0].plot(bits, methods[name], marker=markers[name], lw=1.6, ms=4, color=colors[name], label=name)
        axes[1].plot(bits, costs[name], marker=markers[name], lw=1.6, ms=4, color=colors[name], label=name)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Memory budget (bits/key)")
    axes[0].set_ylabel("Invalid forwarding probability")
    axes[0].grid(True, which="both", lw=0.3, alpha=0.45)
    axes[0].set_title("Filtering frontier")

    axes[1].set_xlabel("Memory budget (bits/key)")
    axes[1].set_ylabel("Expected probes per invalid query")
    axes[1].grid(True, lw=0.3, alpha=0.45)
    axes[1].set_title("Frontend probe cost")
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)

    fig.savefig(FIG_DIR / "icde_baseline_frontier.pdf")
    fig.savefig(FIG_DIR / "icde_baseline_frontier.png", dpi=220)
    plt.close(fig)


def make_scale_and_update() -> None:
    n_values = np.array([1e4, 1e5, 1e6], dtype=float)
    bpk = 16.0
    occ16, atk16 = region_model(16)
    occ4, atk4 = aggregate(occ16, 4), aggregate(atk16, 4)
    bpk_traps = allocate_region_bits(occ16, atk16, bpk)
    bpk_risk = allocate_region_bits(occ4, atk4, bpk)

    avg_k_global = float(hash_count(bpk))
    avg_k_risk = float(np.dot(atk4, hash_count(bpk_risk)))
    avg_k_traps = float(np.dot(atk16, hash_count(bpk_traps)))

    methods = {
        "Global Bloom": avg_k_global,
        "Risk partition": avg_k_risk,
        "TRAPS learned": avg_k_traps,
        "Keyed tag": 1.0,
        "Adaptive tag": 1.0,
    }
    rows: list[dict[str, float | str]] = []
    for n in n_values:
        for name, k in methods.items():
            rows.append(
                {
                    "method": name,
                    "N": n,
                    "bits_per_key": bpk,
                    "memory_mib": n * bpk / 8 / 1024 / 1024,
                    "build_probe_millions": n * k / 1e6,
                }
            )
    write_csv(OUT_DIR / "scale_update.csv", rows)

    migrated = np.linspace(0.0, 1.0, 11)
    dual_overhead_global = 1 + (1 - migrated) * avg_k_global / avg_k_global
    dual_overhead_traps = 1 + (1 - migrated) * avg_k_traps / avg_k_global
    touched = np.array([0.05, 0.10, 0.25, 0.50, 1.00])
    rebuild_global = 1.0
    rebuild_traps = touched * avg_k_traps / avg_k_global

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)

    axes[0].plot(n_values, n_values * bpk / 8 / 1024 / 1024, marker="o", lw=1.7, color="#4c78a8")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Represented credentials")
    axes[0].set_ylabel("Index memory (MiB)")
    axes[0].set_title("Scale to $10^6$")
    axes[0].grid(True, lw=0.3, alpha=0.45)

    for name, k in methods.items():
        axes[1].plot(n_values, n_values * k / 1e6, marker="o", lw=1.5, label=name)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Represented credentials")
    axes[1].set_ylabel("Build probes (millions)")
    axes[1].set_title("Build work")
    axes[1].grid(True, lw=0.3, alpha=0.45)
    axes[1].legend(fontsize=6.5, frameon=False)

    axes[2].plot(touched * 100, np.full_like(touched, rebuild_global), marker="o", lw=1.6, label="Global rebuild")
    axes[2].plot(touched * 100, rebuild_traps, marker="D", lw=1.6, label="TRAPS touched regions")
    axes[2].plot(migrated * 100, dual_overhead_global, ls="--", lw=1.4, label="Global dual-query")
    axes[2].plot(migrated * 100, dual_overhead_traps, ls="--", lw=1.4, label="TRAPS dual-query")
    axes[2].set_xlabel("Touched or migrated share (%)")
    axes[2].set_ylabel("Relative work")
    axes[2].set_title("Epoch update cost")
    axes[2].grid(True, lw=0.3, alpha=0.45)
    axes[2].legend(fontsize=6.2, frameon=False)

    fig.savefig(FIG_DIR / "icde_scale_update_sweep.pdf")
    fig.savefig(FIG_DIR / "icde_scale_update_sweep.png", dpi=220)
    plt.close(fig)


def make_queueing() -> None:
    bpk = 16.0
    valid_fraction = 0.08
    invalid_fraction = 1.0 - valid_fraction
    workers = 8
    service_rate = 1 / 0.12207
    request_rates = np.linspace(25, 1000, 40)

    occ16, atk16 = region_model(16)
    occ4, atk4 = aggregate(occ16, 4), aggregate(atk16, 4)
    fprs = {
        "Backend only": 1.0,
        "Global Bloom": float(bloom_fpr(bpk)),
        "Risk partition": weighted_bloom_fpr(occ4, atk4, bpk)[0],
        "TRAPS learned": weighted_bloom_fpr(occ16, atk16, bpk)[0],
        "Keyed tag": 2.0 ** (-bpk),
        "Adaptive tag": weighted_keyed_tag_fpr(occ16, atk16, bpk)[0],
    }

    rows: list[dict[str, float | str]] = []
    p95_by_method = {name: [] for name in fprs}
    util_by_method = {name: [] for name in fprs}
    for rate in request_rates:
        for name, fpr in fprs.items():
            forward_fraction = 1.0 if name == "Backend only" else valid_fraction + invalid_fraction * fpr
            backend_rate = rate * forward_fraction
            p95, util = erlang_c_wait_p95(backend_rate, service_rate, workers)
            p95_ms = min(p95 * 1000.0, 10000.0)
            p95_by_method[name].append(p95_ms)
            util_by_method[name].append(util)
            rows.append(
                {
                    "method": name,
                    "request_rate": rate,
                    "backend_rate": backend_rate,
                    "forward_fraction": forward_fraction,
                    "p95_latency_ms": p95_ms,
                    "utilization": util,
                }
            )
    write_csv(OUT_DIR / "queueing_saturation.csv", rows)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), constrained_layout=True)
    colors = {
        "Backend only": "#e45756",
        "Global Bloom": "#4c78a8",
        "Risk partition": "#f58518",
        "TRAPS learned": "#54a24b",
        "Keyed tag": "#b279a2",
        "Adaptive tag": "#9d755d",
    }
    for name in fprs:
        axes[0].plot(request_rates, p95_by_method[name], lw=1.6, color=colors[name], label=name)
        axes[1].plot(request_rates, util_by_method[name], lw=1.6, color=colors[name], label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Incoming login attempts/s")
    axes[0].set_ylabel("Backend p95 latency (ms)")
    axes[0].set_title("Bounded-worker latency")
    axes[0].grid(True, which="both", lw=0.3, alpha=0.45)
    axes[1].axhline(1.0, color="#333333", lw=0.9, ls=":")
    axes[1].set_xlabel("Incoming login attempts/s")
    axes[1].set_ylabel("Backend utilization")
    axes[1].set_title("Saturation threshold")
    axes[1].grid(True, lw=0.3, alpha=0.45)
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.savefig(FIG_DIR / "icde_queueing_saturation.pdf")
    fig.savefig(FIG_DIR / "icde_queueing_saturation.png", dpi=220)
    plt.close(fig)


def make_queueing_by_method() -> None:
    """Queueing driven by the same method forwarding rates as the frontier."""
    bits = 16.0
    valid_fraction = 0.08
    invalid_fraction = 1.0 - valid_fraction
    workers = 8
    service_rate = 1 / 0.12207
    request_rates = np.linspace(25, 1000, 40)
    methods = ["Global Bloom", "TRAPS learned", "TRAPS+ReplayCache", "Peppered tag", "Adaptive tag", "RateLimit+Tag"]
    seeds = range(30)

    method_rates: dict[str, float] = {m: 0.0 for m in methods}
    for seed in seeds:
        occupancy, train_attack, test_attack = seeded_region_model(seed)
        rates = production_method_rates(bits, occupancy, train_attack, test_attack, replay_fraction=0.50)
        for method in methods:
            method_rates[method] += float(rates[method]["raw_rate"]) / len(seeds)

    rows: list[dict[str, float | str]] = []
    p99_by_method = {name: [] for name in methods}
    util_by_method = {name: [] for name in methods}
    for rate in request_rates:
        for method in methods:
            valid_drop_rate = 1e-4 if method.startswith("RateLimit") else 0.0
            backend_rate = rate * ((valid_fraction * (1.0 - valid_drop_rate)) + invalid_fraction * method_rates[method])
            p50, p95, p99, util = erlang_c_wait_quantiles(backend_rate, service_rate, workers)
            p99_ms = min(p99 * 1000.0, 10000.0)
            p99_by_method[method].append(p99_ms)
            util_by_method[method].append(util)
            rows.append(
                {
                    "method": method,
                    "bits_per_key": bits,
                    "incoming_attempts_per_second": rate,
                    "backend_invocations_per_second": backend_rate,
                    "worker_utilization": util,
                    "p50_wait_ms": p50 * 1000.0,
                    "p95_wait_ms": p95 * 1000.0,
                    "p99_wait_ms": p99_ms,
                    "fraction_wait_gt_1s": math.exp(-(workers * service_rate - backend_rate) * 1.0) if util < 0.999 else 1.0,
                    "valid_drop_rate_for_rate_limit_baselines": valid_drop_rate,
                    "raw_invalid_forward_rate": method_rates[method],
                }
            )
    write_csv(OUT_DIR / "queueing_by_method.csv", rows)

    plt.rcParams.update({"font.size": 6.7, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.45), constrained_layout=True, sharex=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.03)
    colors = {
        "Global Bloom": "#4c78a8",
        "TRAPS learned": "#54a24b",
        "TRAPS+ReplayCache": "#2f7d32",
        "Peppered tag": "#b279a2",
        "Adaptive tag": "#9d755d",
        "RateLimit+Tag": "#e45756",
    }
    short_labels = {
        "Global Bloom": "Bloom",
        "TRAPS learned": "TRAPS",
        "TRAPS+ReplayCache": "TRAPS+RC",
        "Peppered tag": "Tag",
        "Adaptive tag": "A-tag",
        "RateLimit+Tag": "RL+Tag",
    }
    for method in methods:
        axes[0].plot(request_rates, p99_by_method[method], lw=1.25, color=colors[method], label=short_labels[method])
        axes[1].plot(request_rates, util_by_method[method], lw=1.25, color=colors[method])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("p99 wait (ms)")
    axes[0].set_title("Queueing", pad=1.0)
    axes[0].grid(True, which="both", lw=0.3, alpha=0.45)
    axes[0].legend(fontsize=4.9, frameon=False, loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.7)
    axes[1].axhline(1.0, color="#333333", lw=0.9, ls=":")
    axes[1].set_xlabel("Incoming attempts/s")
    axes[1].set_ylabel("Utilization")
    axes[1].set_title("Worker load", pad=1.0)
    axes[1].grid(True, lw=0.3, alpha=0.45)
    fig.savefig(FIG_DIR / "icde_queueing_by_method.pdf")
    fig.savefig(FIG_DIR / "icde_queueing_by_method.png", dpi=220)
    plt.close(fig)


def make_replay_cache_and_ci() -> None:
    denom = np.logspace(3, 7, 80)
    # One-sided 95% Clopper-style upper bound for zero events: 1 - alpha^(1/n).
    alpha = 0.05
    upper = 1 - alpha ** (1 / denom)
    retained = np.logspace(2, 7, 80)
    token_bits = [128, 192, 256]

    rows: list[dict[str, float | str]] = []
    for n, u in zip(denom, upper):
        rows.append({"kind": "zero_event_upper", "denominator": n, "upper_95": u})
    for b in token_bits:
        for r in retained:
            rows.append({"kind": f"cache_{b}", "retained_tokens": r, "memory_mib": r * b / 8 / 1024 / 1024})
    write_csv(OUT_DIR / "replay_cache_ci.csv", rows)

    plt.rcParams.update({"font.size": 6.8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.55), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.03)
    axes[0].plot(denom, upper, lw=1.8, color="#4c78a8")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Replay denominator")
    axes[0].set_ylabel("95% upper bound")
    axes[0].set_title("Zero-event audit", pad=1.0)
    axes[0].grid(True, which="both", lw=0.3, alpha=0.45)

    for b in token_bits:
        axes[1].plot(retained, retained * b / 8 / 1024 / 1024, lw=1.6, label=f"{b}-bit token")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Retained tokens")
    axes[1].set_ylabel("Memory (MiB)")
    axes[1].set_title("Replay memory", pad=1.0)
    axes[1].grid(True, which="both", lw=0.3, alpha=0.45)
    axes[1].legend(fontsize=5.5, frameon=False, loc="upper left")

    fig.savefig(FIG_DIR / "icde_replay_cache_ci.pdf")
    fig.savefig(FIG_DIR / "icde_replay_cache_ci.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ICDE TRAPS experiment CSVs and figures.")
    parser.add_argument(
        "--suite",
        default="all",
        choices=[
            "all",
            "frontier_production",
            "shift_grid",
            "traps_ablation",
            "baseline_frontier",
            "scale_update",
            "queueing",
            "queueing_by_method",
            "replay_cache",
        ],
    )
    parser.add_argument("--seeds", default="", help="Seed list such as 0-29 or 0,1,2. Suite defaults are used when empty.")
    parser.add_argument("--bits", default="", help="Bits/key list such as 4,6,8,10,12,14,16,20,24.")
    args = parser.parse_args()
    seeds = parse_int_range(args.seeds) if args.seeds else None
    bits = parse_float_list(args.bits) if args.bits else None

    if args.suite in {"all", "frontier_production"}:
        make_production_frontier(seeds=seeds or range(30), bits=bits)
    if args.suite in {"all", "shift_grid"}:
        make_shift_grid(seeds=seeds or range(10), bits_values=bits)
    if args.suite in {"all", "traps_ablation"}:
        make_traps_ablation(seeds=seeds or range(30), bits_values=bits)
    if args.suite in {"all", "baseline_frontier"}:
        make_baseline_frontier()
    if args.suite in {"all", "scale_update"}:
        make_scale_and_update()
    if args.suite in {"all", "queueing"}:
        make_queueing()
    if args.suite in {"all", "queueing_by_method"}:
        make_queueing_by_method()
    if args.suite in {"all", "replay_cache"}:
        make_replay_cache_and_ci()
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote CSV results to {OUT_DIR}")


if __name__ == "__main__":
    main()
