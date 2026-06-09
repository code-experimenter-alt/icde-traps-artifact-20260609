from __future__ import annotations

import csv
import gzip
import hashlib
import math
import os
import ssl
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure" / "icde_experiments"
OUT_DIR = ROOT / "experiments" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANL_AUTH_URL = "https://lanl.ma.ic.ac.uk/data/cyber1/auth.txt.gz"
MAX_LINES = int(os.environ.get("LANL_AUTH_MAX_LINES", "750000"))
BITS_PER_KEY = 16.0
MAX_REGION_BITS_PER_KEY = 32.0
FLOOD_VALID_SHARE = 0.08
BETA_BF = math.log(2) ** 2
WORKERS = 8
ARGON2_MEAN_MS = 122.07
TOKEN_SCHEMAS = {
    "dst_user": ("dst_user",),
    "dst_user_comp": ("dst_user", "dst_comp"),
    "full_stable": ("dst_user", "dst_comp", "auth_type", "logon_type", "orientation"),
}


def bloom_fpr(bits_per_key: np.ndarray | float) -> np.ndarray | float:
    return np.exp(-BETA_BF * np.asarray(bits_per_key))


def hash_count(bits_per_key: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(1, np.rint(np.asarray(bits_per_key) * math.log(2))).astype(float)


def stable_hash(value: str, modulo: int) -> int:
    h = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % modulo


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    total = x.sum()
    if total <= 0:
        return np.ones_like(x) / len(x)
    return x / total


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lanl_ssl_context() -> ssl.SSLContext | None:
    if os.environ.get("LANL_AUTH_INSECURE", "") == "1":
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def load_lanl_auth(max_lines: int) -> list[dict[str, str]]:
    request = urllib.request.Request(
        LANL_AUTH_URL,
        headers={"User-Agent": "TRAPS-ICDE-experiment/1.0"},
    )
    rows: list[dict[str, str]] = []
    context = lanl_ssl_context()
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for i, raw in enumerate(gz):
                if i >= max_lines:
                    break
                parts = raw.decode("utf-8", errors="replace").strip().split(",")
                if len(parts) != 9:
                    continue
                rows.append(
                    {
                        "time": parts[0],
                        "src_user": parts[1],
                        "dst_user": parts[2],
                        "src_comp": parts[3],
                        "dst_comp": parts[4],
                        "auth_type": parts[5],
                        "logon_type": parts[6],
                        "orientation": parts[7],
                        "status": parts[8],
                    }
                )
    return rows


def token(row: dict[str, str], schema: str = "full_stable") -> str:
    # Stable routing token: no source host/user or online outcome field.
    return "|".join(row[field] for field in TOKEN_SCHEMAS[schema])


def token_features(row: dict[str, str], schema: str = "full_stable") -> dict[str, str]:
    aliases = {
        "dst_user": "du",
        "dst_comp": "dc",
        "auth_type": "auth",
        "logon_type": "logon",
        "orientation": "orient",
    }
    return {aliases[field]: row[field] for field in TOKEN_SCHEMAS[schema]}


def add_trace_conditioned_labels(
    rows: list[dict[str, str]],
    seed: int = 0,
    control: str = "LANL-real",
) -> list[dict[str, str]]:
    """Attach local password-validity labels while keeping LANL as metadata.

    LANL Success/Fail is retained only as a risk/timing covariate. It is not
    treated as password-verifier ground truth.
    """
    rng = np.random.default_rng(20260523 + seed)
    out = [dict(r) for r in rows]

    if control == "LANL-shuffled-accounts":
        pairs = [(r["dst_user"], r["dst_comp"]) for r in out]
        perm = rng.permutation(len(pairs))
        for row, j in zip(out, perm):
            row["dst_user"], row["dst_comp"] = pairs[int(j)]
    elif control == "LANL-shuffled-times":
        times = [r["time"] for r in out]
        perm = rng.permutation(len(times))
        for row, j in zip(out, perm):
            row["time"] = times[int(j)]
        out.sort(key=lambda r: int(r["time"]) if str(r["time"]).isdigit() else 0)

    token_counts = Counter(token(r, "full_stable") for r in out)
    max_count = max(token_counts.values()) if token_counts else 1
    for row in out:
        lanl_fail_covariate = row["status"] == "Fail"
        pop = token_counts[token(row, "full_stable")] / max_count
        # The generated password-valid event rate is intentionally low, as in
        # the controlled flood workloads. LANL status influences risk but is
        # not copied into the validity label.
        valid_prob = min(0.18, max(0.02, 0.055 + 0.07 * math.sqrt(pop) - (0.025 if lanl_fail_covariate else 0.0)))
        row["lanl_status_covariate"] = row["status"]
        row["synthetic_password_label"] = "valid" if rng.random() < valid_prob else "invalid"
        if control == "LANL-uniform-risk":
            row["risk_label"] = "uniform"
        else:
            risk_prob = min(0.95, 0.08 + 0.70 * float(lanl_fail_covariate) + 0.20 * (1.0 - pop))
            row["risk_label"] = "risk" if rng.random() < risk_prob else "benign"
    return out


def event_dict(row: dict[str, str], prefix: str = "", schema: str = "full_stable") -> dict[str, int]:
    return {f"{prefix}{k}={v}": 1 for k, v in token_features(row, schema).items()}


def allocate_region_bits(
    occupancy: np.ndarray,
    attack: np.ndarray,
    total_bits_per_key: float,
    max_bits_per_key: float = MAX_REGION_BITS_PER_KEY,
) -> np.ndarray:
    occupancy = normalize(occupancy)
    attack = normalize(attack)
    min_bits = 2.0
    if total_bits_per_key <= min_bits:
        return np.full_like(occupancy, total_bits_per_key)

    base_mem = occupancy * min_bits
    remaining = total_bits_per_key - base_mem.sum()
    ratio = np.maximum(attack * BETA_BF / np.maximum(occupancy, 1e-12), 1e-300)
    lo = 1e-300
    hi = ratio.max() * (1 - 1e-12)

    def extra_mem(lam: float) -> np.ndarray:
        return (occupancy / BETA_BF) * np.log(np.maximum(ratio / lam, 1.0))

    for _ in range(220):
        mid = math.sqrt(lo * hi)
        if extra_mem(mid).sum() > remaining:
            lo = mid
        else:
            hi = mid
    return np.minimum((base_mem + extra_mem(hi)) / np.maximum(occupancy, 1e-12), max_bits_per_key)


def allocate_exponential_bits(
    occupancy: np.ndarray,
    attack: np.ndarray,
    total_bits_per_key: float,
    beta: float,
    min_bits_per_key: float = 1.0,
    max_bits_per_key: float = 32.0,
) -> np.ndarray:
    occupancy = normalize(occupancy)
    attack = normalize(attack)
    if total_bits_per_key <= min_bits_per_key:
        return np.full_like(occupancy, total_bits_per_key)

    base_mem = occupancy * min_bits_per_key
    remaining = total_bits_per_key - base_mem.sum()
    ratio = np.maximum(attack * beta / np.maximum(occupancy, 1e-12), 1e-300)
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
    allocated = (base_mem + extra_mem(hi)) / np.maximum(occupancy, 1e-12)
    return np.minimum(allocated, max_bits_per_key)


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    n = len(rows)
    a = int(n * 0.60)
    b = int(n * 0.80)
    return rows[:a], rows[a:b], rows[b:]


def build_learned_router(
    train: list[dict[str, str]],
    val: list[dict[str, str]],
    all_rows: list[dict[str, str]],
    schema: str = "full_stable",
):
    hasher = FeatureHasher(n_features=2**15, input_type="dict", alternate_sign=False)
    x_train = hasher.transform(event_dict(r, schema=schema) for r in train)
    y_train = np.array(
        [
            1
            if (r.get("risk_label") == "risk" or (r.get("risk_label") is None and r["status"] == "Fail"))
            else 0
            for r in train
        ],
        dtype=int,
    )
    if len(set(y_train.tolist())) < 2:
        scores = np.array([stable_hash(token(r, schema), 1_000_003) / 1_000_003 for r in all_rows], dtype=float)
        val_scores = scores[len(train) : len(train) + len(val)]
        cutpoints = np.quantile(val_scores, np.linspace(0.0, 1.0, 17)[1:-1])
        return scores, cutpoints
    model = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")
    model.fit(x_train, y_train)

    x_all = hasher.transform(event_dict(r, schema=schema) for r in all_rows)
    scores = model.predict_proba(x_all)[:, 1]
    val_scores = scores[len(train) : len(train) + len(val)]
    cutpoints = np.quantile(val_scores, np.linspace(0.0, 1.0, 17)[1:-1])
    return scores, cutpoints


def handcrafted_scores(train: list[dict[str, str]], rows: list[dict[str, str]], schema: str = "full_stable") -> np.ndarray:
    fail_user: Counter[str] = Counter()
    total_user: Counter[str] = Counter()
    fail_comp: Counter[str] = Counter()
    total_comp: Counter[str] = Counter()
    token_total: Counter[str] = Counter()
    for r in train:
        is_fail = r.get("risk_label") == "risk" or (r.get("risk_label") is None and r["status"] == "Fail")
        total_user[r["dst_user"]] += 1
        total_comp[r["dst_comp"]] += 1
        token_total[token(r, schema)] += 1
        if is_fail:
            fail_user[r["dst_user"]] += 1
            fail_comp[r["dst_comp"]] += 1

    scores: list[float] = []
    for r in rows:
        u = r["dst_user"]
        c = r["dst_comp"]
        t = token(r, schema)
        u_rate = (fail_user[u] + 1) / (total_user[u] + 2)
        c_rate = (fail_comp[c] + 1) / (total_comp[c] + 2)
        rarity = 1.0 / math.sqrt(token_total[t] + 1)
        scores.append(0.45 * u_rate + 0.45 * c_rate + 0.10 * rarity)
    return np.array(scores)


def region_from_scores(scores: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    return np.searchsorted(cutpoints, scores, side="right")


def evaluate_partition(
    name: str,
    represented_tokens: set[str],
    test: list[dict[str, str]],
    train_regions: dict[str, int],
    test_regions: np.ndarray,
    region_count: int,
    keyed_tag: str | None = None,
    schema: str = "full_stable",
) -> dict[str, object]:
    valid = 0
    invalid = 0
    valid_drops = 0
    occ = np.zeros(region_count, dtype=float)
    atk = np.zeros(region_count, dtype=float)

    for tok, region in train_regions.items():
        if tok in represented_tokens:
            occ[region] += 1

    for i, row in enumerate(test):
        tok = token(row, schema)
        if "synthetic_password_label" in row:
            is_valid = row["synthetic_password_label"] == "valid"
            is_invalid = row["synthetic_password_label"] == "invalid"
        else:
            is_valid = row["status"] == "Success"
            is_invalid = row["status"] == "Fail"
        if is_valid and tok in represented_tokens:
            valid += 1
            # Stable token-based routing gives one-sided forwarding for represented credentials.
        elif is_invalid:
            invalid += 1
            atk[int(test_regions[i])] += 1
        elif is_valid and tok not in represented_tokens:
            # New successful credential path is excluded from the represented-credential denominator.
            continue

    if keyed_tag == "uniform":
        invalid_fwd = invalid * (2.0 ** (-BITS_PER_KEY))
        probes = 1.0
    elif keyed_tag == "adaptive":
        if atk.sum() == 0:
            atk += 1
        nonempty = occ > 0
        fprs = np.zeros(region_count, dtype=float)
        if nonempty.any():
            tag_bits = allocate_exponential_bits(
                occ[nonempty],
                np.maximum(atk[nonempty], 1e-12),
                BITS_PER_KEY,
                beta=math.log(2),
            )
            fprs[nonempty] = np.exp(-math.log(2) * tag_bits)
        invalid_fwd = float(np.dot(atk, fprs))
        probes = 1.0
    else:
        if atk.sum() == 0:
            atk += 1
        nonempty = occ > 0
        fprs = np.zeros(region_count, dtype=float)
        probes_by_region = np.ones(region_count, dtype=float)
        if nonempty.any():
            # Empty regions contain no represented credentials, so invalid queries
            # routed there are deterministic negatives. Allocate AMQ memory only
            # across regions that actually store represented tokens.
            bpk = allocate_region_bits(occ[nonempty], np.maximum(atk[nonempty], 1e-12), BITS_PER_KEY)
            fprs[nonempty] = bloom_fpr(bpk)
            probes_by_region[nonempty] = hash_count(bpk)
        invalid_fwd = float(np.dot(atk, fprs))
        probes = float(np.dot(normalize(atk), probes_by_region))

    backend_calls = valid + invalid_fwd
    raw_backend_share = backend_calls / max(valid + invalid, 1)
    flood_backend_share = FLOOD_VALID_SHARE + (1.0 - FLOOD_VALID_SHARE) * (invalid_fwd / max(invalid, 1))
    service_per_worker = 1000.0 / ARGON2_MEAN_MS
    capacity = WORKERS * service_per_worker
    saturation_attempts_per_sec = capacity / max(flood_backend_share, 1e-12)

    return {
        "schema": schema,
        "method": name,
        "represented_valid_test": valid,
        "invalid_test": invalid,
        "valid_fnr": valid_drops / max(valid, 1),
        "expected_invalid_forwards": invalid_fwd,
        "invalid_forward_rate": invalid_fwd / max(invalid, 1),
        "backend_calls": backend_calls,
        "backend_call_rate": raw_backend_share,
        "flood_valid_share": FLOOD_VALID_SHARE,
        "flood_backend_call_rate": flood_backend_share,
        "frontend_probes": probes,
        "saturation_attempts_per_sec": saturation_attempts_per_sec,
    }


def evaluate_schema(rows: list[dict[str, str]], schema: str, control: str = "LANL-real") -> list[dict[str, object]]:
    train, val, test = split_rows(rows)
    represented = {
        token(r, schema)
        for r in train + val
        if r.get("synthetic_password_label", "valid" if r["status"] == "Success" else "invalid") == "valid"
    }

    all_rows = train + val + test
    if control == "LANL-uniform-risk":
        learned_scores = np.array([stable_hash(token(r, schema), 1_000_003) / 1_000_003 for r in all_rows], dtype=float)
        learned_cuts = np.quantile(learned_scores[len(train) : len(train) + len(val)], np.linspace(0.0, 1.0, 17)[1:-1])
        risk_scores = np.ones(len(all_rows), dtype=float)
    else:
        learned_scores, learned_cuts = build_learned_router(train, val, all_rows, schema)
        risk_scores = handcrafted_scores(train, all_rows, schema)
    risk_cuts = np.quantile(risk_scores[len(train) : len(train) + len(val)], np.linspace(0.0, 1.0, 5)[1:-1])

    train_tokens = {tok: r for r in train + val if (tok := token(r, schema)) in represented}
    learned_train_regions = {
        tok: int(region_from_scores(np.array([learned_scores[i]]), learned_cuts)[0])
        for i, r in enumerate(all_rows)
        if (tok := token(r, schema)) in train_tokens
    }
    risk_train_regions = {
        tok: int(region_from_scores(np.array([risk_scores[i]]), risk_cuts)[0])
        for i, r in enumerate(all_rows)
        if (tok := token(r, schema)) in train_tokens
    }
    hash_train_regions = {tok: stable_hash(tok, 16) for tok in train_tokens}
    global_train_regions = {tok: 0 for tok in train_tokens}

    test_offset = len(train) + len(val)
    learned_test_regions = region_from_scores(learned_scores[test_offset:], learned_cuts)
    risk_test_regions = region_from_scores(risk_scores[test_offset:], risk_cuts)
    hash_test_regions = np.array([stable_hash(token(r, schema), 16) for r in test], dtype=int)
    global_test_regions = np.zeros(len(test), dtype=int)

    evaluated = [
        evaluate_partition("Global Bloom", represented, test, global_train_regions, global_test_regions, 1, schema=schema),
        evaluate_partition("Hash partition", represented, test, hash_train_regions, hash_test_regions, 16, schema=schema),
        evaluate_partition("Risk partition", represented, test, risk_train_regions, risk_test_regions, 4, schema=schema),
        evaluate_partition("TRAPS learned", represented, test, learned_train_regions, learned_test_regions, 16, schema=schema),
        evaluate_partition(
            "Keyed tag",
            represented,
            test,
            global_train_regions,
            global_test_regions,
            1,
            keyed_tag="uniform",
            schema=schema,
        ),
        evaluate_partition(
            "Adaptive tag",
            represented,
            test,
            learned_train_regions,
            learned_test_regions,
            16,
            keyed_tag="adaptive",
            schema=schema,
        ),
    ]
    for row in evaluated:
        row["control"] = control
    return evaluated


def gini_from_counts(counts: list[int]) -> float:
    if not counts:
        return 0.0
    arr = np.sort(np.asarray(counts, dtype=float))
    total = arr.sum()
    if total <= 0:
        return 0.0
    n = arr.size
    return float((2.0 * np.dot(np.arange(1, n + 1), arr) / (n * total)) - (n + 1) / n)


def js_divergence(train_counts: Counter[str], test_counts: Counter[str]) -> float:
    keys = sorted(set(train_counts) | set(test_counts))
    if not keys:
        return 0.0
    p = np.array([train_counts[k] for k in keys], dtype=float)
    q = np.array([test_counts[k] for k in keys], dtype=float)
    p = normalize(p)
    q = normalize(q)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main() -> None:
    raw_rows = load_lanl_auth(MAX_LINES)
    if len(raw_rows) < 10000:
        raise RuntimeError(f"LANL sample too small: {len(raw_rows)} rows")
    rows = add_trace_conditioned_labels(raw_rows, control="LANL-real")
    train, val, test = split_rows(rows)
    all_status = Counter(r["status"] for r in raw_rows)
    synth_status = Counter(r["synthetic_password_label"] for r in rows)

    schema_summaries = []
    for schema in TOKEN_SCHEMAS:
        represented = {token(r, schema) for r in train + val if r["synthetic_password_label"] == "valid"}
        schema_summaries.append(
            {
                "schema": schema,
                "represented_tokens": len(represented),
                "synthetic_test_invalid": sum(1 for r in test if r["synthetic_password_label"] == "invalid"),
            }
        )
    write_csv(OUT_DIR / "lanl_token_schema_summary.csv", schema_summaries)
    write_csv(
        OUT_DIR / "lanl_trace_summary.csv",
        [
            {
                "source": LANL_AUTH_URL,
                "sampled_events": len(rows),
                "train_events": len(train),
                "validation_events": len(val),
                "test_events": len(test),
                "lanl_success_covariate_events": all_status.get("Success", 0),
                "lanl_failure_covariate_events": all_status.get("Fail", 0),
                "synthetic_valid_events": synth_status.get("valid", 0),
                "synthetic_invalid_events": synth_status.get("invalid", 0),
                "represented_tokens": schema_summaries[-1]["represented_tokens"],
            }
        ],
    )
    train_tokens = Counter(token(r, "full_stable") for r in train)
    test_tokens = Counter(token(r, "full_stable") for r in test)
    user_counts = Counter(r["dst_user"] for r in rows)
    top_1pct = max(1, math.ceil(0.01 * len(user_counts)))
    top_mass = sum(v for _, v in user_counts.most_common(top_1pct)) / max(len(rows), 1)
    write_csv(
        OUT_DIR / "lanl_trace_denominators.csv",
        [
            {
                "events_used": len(rows),
                "unique_users": len(user_counts),
                "unique_sources": len({r["src_comp"] for r in rows}),
                "top_1pct_user_event_mass": top_mass,
                "gini_user_events": gini_from_counts(list(user_counts.values())),
                "train_test_JS_divergence": js_divergence(train_tokens, test_tokens),
                "synthetic_valid_events": synth_status.get("valid", 0),
                "synthetic_invalid_events": synth_status.get("invalid", 0),
                "synthetic_unique_invalid_pairs": len({token(r, "full_stable") for r in rows if r["synthetic_password_label"] == "invalid"}),
                "synthetic_replay_fraction": 1.0
                - len({token(r, "full_stable") for r in rows if r["synthetic_password_label"] == "invalid"})
                / max(synth_status.get("invalid", 1), 1),
            }
        ],
    )

    schema_results = {schema: evaluate_schema(rows, schema, control="LANL-real") for schema in TOKEN_SCHEMAS}
    results = schema_results["full_stable"]
    write_csv(OUT_DIR / "lanl_trace_conditioned.csv", results)
    write_csv(OUT_DIR / "lanl_trace_end_to_end.csv", results)

    outcome_results = evaluate_schema(raw_rows, "full_stable", control="LANL-status-outcome")
    write_csv(OUT_DIR / "lanl_status_outcome.csv", outcome_results)
    out_train, out_val, out_test = split_rows(raw_rows)
    represented_outcome = {token(r, "full_stable") for r in out_train + out_val if r["status"] == "Success"}
    outcome_success_test = sum(1 for r in out_test if r["status"] == "Success" and token(r, "full_stable") in represented_outcome)
    outcome_fail_test = sum(1 for r in out_test if r["status"] == "Fail")
    write_csv(
        OUT_DIR / "lanl_status_outcome_denominators.csv",
        [
            {
                "events_used": len(raw_rows),
                "train_events": len(out_train),
                "validation_events": len(out_val),
                "test_events": len(out_test),
                "lanl_success_events": all_status.get("Success", 0),
                "lanl_failure_events": all_status.get("Fail", 0),
                "represented_success_test_events": outcome_success_test,
                "failure_test_events": outcome_fail_test,
                "represented_tokens": len(represented_outcome),
                "unique_failure_tokens": len({token(r, "full_stable") for r in raw_rows if r["status"] == "Fail"}),
                "uses_lanl_status_as_outcome": "true",
                "uses_plaintext_passwords": "false",
            }
        ],
    )

    negative_control_results: list[dict[str, object]] = []
    for control in ["LANL-real", "LANL-shuffled-accounts", "LANL-shuffled-times", "LANL-uniform-risk"]:
        control_rows = rows if control == "LANL-real" else add_trace_conditioned_labels(raw_rows, control=control)
        negative_control_results.extend(evaluate_schema(control_rows, "full_stable", control=control))
    write_csv(OUT_DIR / "lanl_trace_negative_controls.csv", negative_control_results)

    ablation_rows = []
    for r in results:
        ablation_rows.append(
            {
                "schema": r["schema"],
                "router": r["method"],
                "invalid_forward_rate": r["invalid_forward_rate"],
                "frontend_probes": r["frontend_probes"],
                "saturation_attempts_per_sec": r["saturation_attempts_per_sec"],
            }
        )
    write_csv(OUT_DIR / "lanl_router_ablation.csv", ablation_rows)
    sensitivity_rows = []
    for schema, rows_for_schema in schema_results.items():
        for row in rows_for_schema:
            if row["method"] in {"Global Bloom", "Risk partition", "TRAPS learned", "Keyed tag", "Adaptive tag"}:
                sensitivity_rows.append(
                    {
                        "schema": schema,
                        "method": row["method"],
                        "invalid_forward_rate": row["invalid_forward_rate"],
                        "represented_valid_test": row["represented_valid_test"],
                        "invalid_test": row["invalid_test"],
                    }
                )
    write_csv(OUT_DIR / "lanl_token_sensitivity.csv", sensitivity_rows)

    methods = [r["method"] for r in results]
    color_map = {
        "Global Bloom": "#4c78a8",
        "Hash partition": "#72b7b2",
        "Risk partition": "#f58518",
        "TRAPS learned": "#54a24b",
        "Keyed tag": "#b279a2",
        "Adaptive tag": "#9d755d",
    }
    colors = [color_map[m] for m in methods]
    plt.rcParams.update({"font.size": 7.4, "font.family": "DejaVu Sans"})

    short_method_labels = {
        "Global Bloom": "Glob.",
        "Hash partition": "Hash",
        "Risk partition": "Risk",
        "TRAPS learned": "TRAPS",
        "Keyed tag": "Tag",
        "Adaptive tag": "A-tag",
    }
    method_tick_labels = [short_method_labels[m] for m in methods]

    fig, axes = plt.subplots(1, 5, figsize=(7.1, 2.05), constrained_layout=True)
    x = np.arange(len(methods))
    invalid_values = [r["invalid_forward_rate"] for r in results]
    flood_values = [r["flood_backend_call_rate"] for r in results]
    probe_values = [r["frontend_probes"] for r in results]
    saturation_values = [r["saturation_attempts_per_sec"] for r in results]

    def tighten_linear_axis(ax: plt.Axes, values: list[float], pad_fraction: float = 0.28) -> None:
        lo = min(values)
        hi = max(values)
        span = max(hi - lo, abs(hi) * 1e-4, 1e-12)
        lower = max(0.0, lo - span * pad_fraction)
        upper = hi + span * pad_fraction
        ax.set_ylim(lower, upper)

    axes[0].bar(x, invalid_values, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylim(max(min(invalid_values) * 0.55, 1e-12), max(invalid_values) * 1.45)
    axes[0].set_ylabel("Invalid fwd.")
    axes[0].set_title("Real trace")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_tick_labels, rotation=35, ha="right", fontsize=5.7)
    axes[0].tick_params(axis="x", pad=1.5)
    axes[0].grid(True, axis="y", which="both", lw=0.3, alpha=0.4)

    axes[1].bar(x, flood_values, color=colors)
    tighten_linear_axis(axes[1], flood_values)
    axes[1].set_ylabel("Forwarded share")
    axes[1].set_title("Flood load")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_tick_labels, rotation=35, ha="right", fontsize=5.7)
    axes[1].tick_params(axis="x", pad=1.5)
    axes[1].grid(True, axis="y", lw=0.3, alpha=0.4)

    axes[2].bar(x, probe_values, color=colors)
    axes[2].set_ylabel("Expected probes")
    axes[2].set_title("Frontend")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(method_tick_labels, rotation=35, ha="right", fontsize=5.7)
    axes[2].tick_params(axis="x", pad=1.5)
    axes[2].grid(True, axis="y", lw=0.3, alpha=0.4)

    axes[3].bar(x, saturation_values, color=colors)
    tighten_linear_axis(axes[3], saturation_values)
    axes[3].set_ylabel("Attempts/s")
    axes[3].set_title("8-worker cap")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(method_tick_labels, rotation=35, ha="right", fontsize=5.7)
    axes[3].tick_params(axis="x", pad=1.5)
    axes[3].grid(True, axis="y", lw=0.3, alpha=0.4)

    schema_order = list(TOKEN_SCHEMAS)
    schema_labels = ["User", "User+host", "Full"]
    sens_methods = ["Global Bloom", "TRAPS learned", "Keyed tag", "Adaptive tag"]
    controls = ["LANL-real", "LANL-shuffled-accounts", "LANL-shuffled-times", "LANL-uniform-risk"]
    control_labels = ["Real", "Acct", "Time", "Unif."]
    for method in ["Global Bloom", "TRAPS learned", "Adaptive tag"]:
        values = [
            max(
                float(next(row for row in negative_control_results if row["control"] == control and row["method"] == method)["invalid_forward_rate"]),
                1e-12,
            )
            for control in controls
        ]
        axes[4].plot(control_labels, values, marker="o", lw=1.3, ms=3.5, color=color_map[method], label=method)
    axes[4].set_yscale("log")
    axes[4].set_ylabel("Invalid fwd.")
    axes[4].set_title("Neg. controls")
    axes[4].tick_params(axis="x", rotation=30, labelsize=5.7, pad=1.5)
    axes[4].grid(True, axis="y", which="both", lw=0.3, alpha=0.4)
    fig.savefig(FIG_DIR / "icde_lanl_trace_end_to_end.pdf")
    fig.savefig(FIG_DIR / "icde_lanl_trace_end_to_end.png", dpi=220)
    fig.savefig(FIG_DIR / "icde_lanl_trace_suite.pdf")
    fig.savefig(FIG_DIR / "icde_lanl_trace_suite.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25), constrained_layout=True)
    router_results = [r for r in results if r["method"] in {"Global Bloom", "Hash partition", "Risk partition", "TRAPS learned"}]
    x_router = np.arange(len(router_results))
    axes[0].bar(x_router, [r["invalid_forward_rate"] for r in router_results], color=[color_map[r["method"]] for r in router_results])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Invalid forwarding rate")
    axes[0].set_title("Router ablation")
    axes[0].set_xticks(x_router)
    axes[0].set_xticklabels([short_method_labels[r["method"]] for r in router_results], rotation=25, ha="right")
    axes[0].grid(True, axis="y", which="both", lw=0.3, alpha=0.4)

    axes[1].bar(x, [r["frontend_probes"] for r in results], color=colors)
    axes[1].set_ylabel("Expected probes")
    axes[1].set_title("Frontend cost")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_tick_labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", lw=0.3, alpha=0.4)
    fig.savefig(FIG_DIR / "icde_lanl_router_ablation.pdf")
    fig.savefig(FIG_DIR / "icde_lanl_router_ablation.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.25), constrained_layout=True)
    x_schema = np.arange(len(schema_order))
    width = 0.18
    for offset, method in enumerate(sens_methods):
        values = [
            max(float(next(row for row in schema_results[schema] if row["method"] == method)["invalid_forward_rate"]), 1e-12)
            for schema in schema_order
        ]
        ax.bar(x_schema + (offset - 1.5) * width, values, width=width, color=color_map[method], label=method)
    ax.set_yscale("log")
    ax.set_ylabel("Invalid forwarding rate")
    ax.set_title("LANL token-granularity sensitivity")
    ax.set_xticks(x_schema)
    ax.set_xticklabels(schema_labels, rotation=20, ha="right")
    ax.grid(True, axis="y", which="both", lw=0.3, alpha=0.4)
    ax.legend(fontsize=6.5, frameon=False)
    fig.savefig(FIG_DIR / "icde_lanl_token_sensitivity.pdf")
    fig.savefig(FIG_DIR / "icde_lanl_token_sensitivity.png", dpi=220)
    plt.close(fig)

    print(f"sampled={len(rows)} success={all_status.get('Success', 0)} fail={all_status.get('Fail', 0)}")
    for row in results:
        print(
            f"{row['method']}: invalid_fwd={row['invalid_forward_rate']:.6g}, "
            f"flood_backend_share={row['flood_backend_call_rate']:.6g}, probes={row['frontend_probes']:.3f}"
        )


if __name__ == "__main__":
    main()
