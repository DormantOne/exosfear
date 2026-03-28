#!/usr/bin/env python3
"""
Graph Law Benchmark - End-to-End Harness

What this does:
- Generates a synthetic graph-law benchmark with multiple graph families
- Writes train/val/test prompts, gold files, shard files, and templates
- Surfaces benchmark snapshots at Stage 0, Midstage, and Completed
- Runs an internal baseline solver from start to finish
- Evaluates either internal baseline predictions or external prediction files

Dependencies:
    pip install networkx numpy
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import sys
import textwrap
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

FAMILY_ORDER = [
    "erdos_renyi",
    "barabasi_albert",
    "watts_strogatz",
    "stochastic_block",
    "random_geometric",
]

FAMILY_LABELS = {
    "erdos_renyi": "Erdős–Rényi",
    "barabasi_albert": "Barabási–Albert",
    "watts_strogatz": "Watts–Strogatz",
    "stochastic_block": "Stochastic Block",
    "random_geometric": "Random Geometric",
}

PARAM_SPECS = {
    "erdos_renyi": [("p", "float")],
    "barabasi_albert": [("m", "int")],
    "watts_strogatz": [("k", "int"), ("beta", "float")],
    "stochastic_block": [("blocks", "int"), ("p_in", "float"), ("p_out", "float")],
    "random_geometric": [("radius", "float")],
}

FEATURE_NAMES = [
    "n_nodes",
    "n_edges",
    "density",
    "avg_degree",
    "degree_std",
    "max_degree_ratio",
    "clustering",
    "transitivity",
    "largest_component_frac",
    "avg_path_lcc",
    "assortativity",
    "triangle_density",
    "deg_p10",
    "deg_p25",
    "deg_p50",
    "deg_p75",
    "deg_p90",
    "spectrum_1",
    "spectrum_2",
    "spectrum_3",
    "spectrum_4",
    "spectrum_5",
    "spectrum_6",
]


def section(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def short_line(title: str) -> None:
    print(f"--- {title} {'-' * max(1, 70 - len(title))}")


def prompt_int(label: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return value


def prompt_float(label: str, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return value


def prompt_bool(label: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{label} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def prompt_path(label: str, default: str) -> Path:
    raw = input(f"{label} [{default}]: ").strip()
    return Path(raw or default)


def prompt_families(default: Sequence[str]) -> List[str]:
    print("\nGraph family menu:")
    for idx, fam in enumerate(FAMILY_ORDER, start=1):
        print(f"  {idx}) {fam}")
    raw = input("Choose families by number, comma-separated [all]: ").strip()
    if not raw:
        return list(default)
    selected: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
            if 1 <= idx <= len(FAMILY_ORDER):
                selected.append(FAMILY_ORDER[idx - 1])
            else:
                raise ValueError
        except ValueError:
            print(f"Ignoring invalid entry: {part}")
    return selected or list(default)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dump(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def safe_percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, q))


def safe_assortativity(g: nx.Graph) -> float:
    try:
        val = nx.degree_pearson_correlation_coefficient(g)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def largest_component_subgraph(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g.copy()
    comp = max(nx.connected_components(g), key=len)
    return g.subgraph(comp).copy()


def even_k(rng: random.Random, n: int) -> int:
    max_k = max(4, min(12, n - 1))
    values = [k for k in range(4, max_k + 1) if k % 2 == 0 and k < n]
    return rng.choice(values) if values else 2


def sample_block_sizes(rng: random.Random, n: int, blocks: int) -> List[int]:
    # Guaranteed construction; no infinite loop.
    min_each = 3
    if n < blocks * min_each:
        min_each = max(1, n // blocks)
    sizes = [min_each] * blocks
    remaining = n - sum(sizes)
    for _ in range(remaining):
        sizes[rng.randrange(blocks)] += 1
    rng.shuffle(sizes)
    # Merge empty/zero blocks if any edge case snuck in.
    sizes = [s for s in sizes if s > 0]
    if not sizes:
        return [n]
    return sizes


def generate_graph(rng: random.Random, family: str, min_nodes: int, max_nodes: int) -> Tuple[nx.Graph, Dict[str, Any], Dict[str, Any]]:
    n = rng.randint(min_nodes, max_nodes)

    if family == "erdos_renyi":
        p = round(rng.uniform(0.04, 0.22), 3)
        g = nx.erdos_renyi_graph(n, p, seed=rng.randrange(1_000_000))
        params = {"p": p}
        meta = {}

    elif family == "barabasi_albert":
        m = rng.randint(1, min(6, max(1, n - 1)))
        g = nx.barabasi_albert_graph(n, m, seed=rng.randrange(1_000_000))
        params = {"m": m}
        meta = {}

    elif family == "watts_strogatz":
        k = even_k(rng, n)
        beta = round(rng.uniform(0.02, 0.45), 3)
        g = nx.watts_strogatz_graph(n, k, beta, seed=rng.randrange(1_000_000))
        params = {"k": k, "beta": beta}
        meta = {}

    elif family == "stochastic_block":
        max_blocks = min(4, max(2, n // 6))
        blocks = rng.randint(2, max_blocks)
        sizes = sample_block_sizes(rng, n, blocks)
        p_in = round(rng.uniform(0.35, 0.85), 3)
        max_p_out = min(0.25, max(0.03, p_in * 0.55))
        p_out = round(rng.uniform(0.01, max_p_out), 3)
        probs = [[p_out for _ in sizes] for _ in sizes]
        for i in range(len(sizes)):
            probs[i][i] = p_in
        g = nx.stochastic_block_model(sizes, probs, seed=rng.randrange(1_000_000))
        g = nx.Graph(g)
        params = {"blocks": len(sizes), "p_in": p_in, "p_out": p_out}
        meta = {"sizes": sizes}

    elif family == "random_geometric":
        radius = round(rng.uniform(0.12, 0.32), 3)
        g = nx.random_geometric_graph(n, radius, seed=rng.randrange(1_000_000))
        g = nx.Graph(g)
        params = {"radius": radius}
        meta = {}

    else:
        raise ValueError(f"Unknown family: {family}")

    # Relabel nodes to integers for consistency.
    mapping = {node: idx for idx, node in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    return g, params, meta


def spectral_signature(g: nx.Graph, topk: int) -> List[float]:
    n = g.number_of_nodes()
    if n == 0:
        return [0.0] * topk
    a = nx.to_numpy_array(g, dtype=float)
    eigs = np.linalg.eigvalsh(a)
    eigs = sorted((abs(float(x)) for x in eigs), reverse=True)
    sig = [round(v, 4) for v in eigs[:topk]]
    while len(sig) < topk:
        sig.append(0.0)
    return sig


def degree_stats(g: nx.Graph) -> Dict[str, float]:
    degs = [float(d) for _, d in g.degree()]
    n = max(1, g.number_of_nodes())
    return {
        "avg_degree": safe_mean(degs),
        "degree_std": float(np.std(degs)) if degs else 0.0,
        "max_degree_ratio": (max(degs) / max(1, n - 1)) if degs else 0.0,
        "deg_p10": safe_percentile(degs, 10),
        "deg_p25": safe_percentile(degs, 25),
        "deg_p50": safe_percentile(degs, 50),
        "deg_p75": safe_percentile(degs, 75),
        "deg_p90": safe_percentile(degs, 90),
    }


def graph_features(g: nx.Graph, topk: int) -> Dict[str, float]:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    dens = nx.density(g) if n > 1 else 0.0
    lcc = largest_component_subgraph(g) if n else g.copy()
    lcc_frac = (lcc.number_of_nodes() / n) if n else 0.0
    avg_path = 0.0
    if lcc.number_of_nodes() > 1:
        try:
            avg_path = float(nx.average_shortest_path_length(lcc))
        except Exception:
            avg_path = 0.0
    triangles = sum(nx.triangles(g).values()) / 3 if n else 0.0
    tri_density = triangles / max(1.0, n)

    feats: Dict[str, float] = {
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": float(dens),
        "clustering": float(nx.average_clustering(g)) if n else 0.0,
        "transitivity": float(nx.transitivity(g)) if n else 0.0,
        "largest_component_frac": float(lcc_frac),
        "avg_path_lcc": float(avg_path),
        "assortativity": safe_assortativity(g),
        "triangle_density": float(tri_density),
    }
    feats.update(degree_stats(g))
    spec = spectral_signature(g, topk)
    for idx in range(6):
        feats[f"spectrum_{idx+1}"] = float(spec[idx]) if idx < len(spec) else 0.0
    return feats


def format_degree_bins(g: nx.Graph) -> str:
    degs = sorted(int(d) for _, d in g.degree())
    if not degs:
        return "0:0"
    bins = Counter(min(d, 12) for d in degs)
    ordered = []
    for k in sorted(bins):
        label = f"{k}" if k < 12 else "12+"
        ordered.append(f"{label}:{bins[k]}")
    return ", ".join(ordered)


def sample_edges_text(g: nx.Graph, sample_max: int, rng: random.Random) -> str:
    edges = list(g.edges())
    if not edges:
        return "none"
    rng2 = random.Random(rng.randrange(1_000_000))
    rng2.shuffle(edges)
    picked = edges[:sample_max]
    return ", ".join(f"({u},{v})" for u, v in picked)


def describe_example(example_id: str, family: str, params: Dict[str, Any], feats: Dict[str, float], edge_text: str, spectrum_topk: int) -> Dict[str, str]:
    local = textwrap.dedent(
        f"""
        EXAMPLE {example_id}
        LOCAL_SHAPE
        degree_bins={format_numeric_bins(feats)}
        avg_degree={feats['avg_degree']:.3f}
        degree_std={feats['degree_std']:.3f}
        max_degree_ratio={feats['max_degree_ratio']:.3f}
        triangle_density={feats['triangle_density']:.3f}
        clustering={feats['clustering']:.3f}
        """
    ).strip()

    global_txt = textwrap.dedent(
        f"""
        EXAMPLE {example_id}
        GLOBAL_SHAPE
        n_nodes={int(feats['n_nodes'])}
        n_edges={int(feats['n_edges'])}
        density={feats['density']:.4f}
        transitivity={feats['transitivity']:.4f}
        largest_component_frac={feats['largest_component_frac']:.4f}
        avg_path_lcc={feats['avg_path_lcc']:.4f}
        assortativity={feats['assortativity']:.4f}
        """
    ).strip()

    spec_values = ", ".join(f"{feats[f'spectrum_{i+1}']:.4f}" for i in range(spectrum_topk))
    spectral = textwrap.dedent(
        f"""
        EXAMPLE {example_id}
        SPECTRAL_SHAPE
        top_abs_adjacency_eigenvalues=[{spec_values}]
        """
    ).strip()

    edge = textwrap.dedent(
        f"""
        EXAMPLE {example_id}
        EDGE_GLIMPSE
        sampled_edges={edge_text}
        """
    ).strip()

    prompt = textwrap.dedent(
        f"""
        Infer the latent graph-generating law from the evidence below.

        Reply in exactly two lines.
        Line 1 must be:
        LAW family=<family>; <param1>=<value>; <param2>=<value>
        Line 2 must be:
        SELF confidence=<0_to_1>; alt_family=<family>; why=<brief_reason>

        Use only these families:
        erdos_renyi, barabasi_albert, watts_strogatz, stochastic_block, random_geometric

        {local}

        {global_txt}

        {spectral}

        {edge}
        """
    ).strip()

    gold_law = format_law(family, params)
    gold_self = f"SELF confidence=1.00; alt_family=none; why=gold label"
    answer = gold_law + "\n" + gold_self

    return {
        "local": local,
        "global": global_txt,
        "spectral": spectral,
        "edge": edge,
        "prompt": prompt,
        "answer": answer,
    }


def format_numeric_bins(feats: Dict[str, float]) -> str:
    return ", ".join(
        [
            f"p10={feats['deg_p10']:.2f}",
            f"p25={feats['deg_p25']:.2f}",
            f"p50={feats['deg_p50']:.2f}",
            f"p75={feats['deg_p75']:.2f}",
            f"p90={feats['deg_p90']:.2f}",
        ]
    )


def format_law(family: str, params: Dict[str, Any]) -> str:
    pieces = [f"family={family}"]
    for name, kind in PARAM_SPECS[family]:
        value = params[name]
        if kind == "int":
            pieces.append(f"{name}={int(value)}")
        else:
            pieces.append(f"{name}={float(value):.3f}")
    return "LAW " + "; ".join(pieces)


@dataclass
class Example:
    example_id: str
    split: str
    family: str
    params: Dict[str, Any]
    features: Dict[str, float]
    lens_text: Dict[str, str]

    def to_gold(self) -> Dict[str, Any]:
        return {
            "id": self.example_id,
            "family": self.family,
            "params": self.params,
            "gold_law": format_law(self.family, self.params),
        }

    def to_prompt(self) -> Dict[str, Any]:
        return {
            "id": self.example_id,
            "prompt": self.lens_text["prompt"],
        }

    def to_train_record(self) -> Dict[str, Any]:
        return {
            "id": self.example_id,
            "prompt": self.lens_text["prompt"],
            "target": self.lens_text["answer"],
            "family": self.family,
            "params": self.params,
        }


def feature_vector(example: Example) -> np.ndarray:
    return np.array([float(example.features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)


def make_example(example_id: str, split: str, family: str, rng: random.Random, min_nodes: int, max_nodes: int, spectrum_topk: int, edge_sample_max: int) -> Example:
    g, params, _ = generate_graph(rng, family, min_nodes, max_nodes)
    feats = graph_features(g, topk=max(6, spectrum_topk))
    edge_text = sample_edges_text(g, edge_sample_max, rng)
    lenses = describe_example(example_id, family, params, feats, edge_text, spectrum_topk)
    return Example(example_id, split, family, params, feats, lenses)


def generate_examples(count: int, split: str, families: Sequence[str], rng: random.Random, min_nodes: int, max_nodes: int, spectrum_topk: int, edge_sample_max: int, start_index: int) -> List[Example]:
    rows: List[Example] = []
    for i in range(count):
        family = rng.choice(list(families))
        ex_id = f"{split[:2].upper()}{start_index + i:07d}"
        rows.append(make_example(ex_id, split, family, rng, min_nodes, max_nodes, spectrum_topk, edge_sample_max))
        if (i + 1) % 200 == 0 or i + 1 == count:
            print(f"  generated {split}: {i + 1}/{count}")
    return rows


def write_split_files(examples: Sequence[Example], out_dir: Path, split: str) -> None:
    split_dir = out_dir / split
    ensure_dir(split_dir)

    write_jsonl((ex.to_train_record() for ex in examples), split_dir / f"{split}_records.jsonl")
    write_jsonl((ex.to_prompt() for ex in examples), split_dir / f"{split}_prompts.jsonl")
    write_jsonl((ex.to_gold() for ex in examples), split_dir / f"{split}_gold.jsonl")

    blank_rows = [{"id": ex.example_id, "prediction": "LAW family=<family>; <param>=<value>\nSELF confidence=<0_to_1>; alt_family=<family>; why=<brief_reason>"} for ex in examples]
    write_jsonl(blank_rows, split_dir / "DO_NOT_SCORE_blank_response_template.jsonl")

    for lens_name, file_name in [
        ("local", f"{split}_node0_local.txt"),
        ("global", f"{split}_node1_global.txt"),
        ("spectral", f"{split}_node2_spectral.txt"),
        ("edge", f"{split}_node3_edge.txt"),
    ]:
        with (split_dir / file_name).open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(ex.lens_text[lens_name])
                f.write("\n\n")


def dataset_summary(examples: Sequence[Example]) -> Dict[str, Any]:
    fam_counts = Counter(ex.family for ex in examples)
    avg_nodes = safe_mean([ex.features["n_nodes"] for ex in examples])
    avg_density = safe_mean([ex.features["density"] for ex in examples])
    avg_clustering = safe_mean([ex.features["clustering"] for ex in examples])
    per_family = {}
    for fam in FAMILY_ORDER:
        fam_rows = [ex for ex in examples if ex.family == fam]
        if not fam_rows:
            continue
        per_family[fam] = {
            "count": len(fam_rows),
            "avg_nodes": round(safe_mean([ex.features["n_nodes"] for ex in fam_rows]), 3),
            "avg_density": round(safe_mean([ex.features["density"] for ex in fam_rows]), 5),
            "avg_clustering": round(safe_mean([ex.features["clustering"] for ex in fam_rows]), 5),
        }
    return {
        "count": len(examples),
        "family_counts": dict(fam_counts),
        "avg_nodes": round(avg_nodes, 3),
        "avg_density": round(avg_density, 5),
        "avg_clustering": round(avg_clustering, 5),
        "per_family": per_family,
    }


def print_dataset_summary(name: str, summary: Dict[str, Any]) -> None:
    short_line(name)
    print(json.dumps(summary, indent=2))


def parameter_defaults_by_family(train_rows: Sequence[Example]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fam in FAMILY_ORDER:
        fam_rows = [r for r in train_rows if r.family == fam]
        if not fam_rows:
            continue
        params_out: Dict[str, Any] = {}
        for param, kind in PARAM_SPECS[fam]:
            vals = [r.params[param] for r in fam_rows]
            if kind == "int":
                params_out[param] = int(round(safe_mean([int(v) for v in vals])))
            else:
                params_out[param] = round(safe_mean([float(v) for v in vals]), 3)
        out[fam] = params_out
    return out


def parse_prediction_text(text: str) -> Dict[str, Any]:
    law_line = ""
    self_line = ""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("LAW "):
            law_line = line
        elif line.upper().startswith("SELF "):
            self_line = line
    if not law_line:
        law_line = text.strip().splitlines()[0].strip() if text.strip() else ""

    family_match = re.search(r"family\s*=\s*([a-zA-Z_]+)", law_line)
    family = family_match.group(1).strip() if family_match else None
    params: Dict[str, Any] = {}
    if family in PARAM_SPECS:
        for param, kind in PARAM_SPECS[family]:
            m = re.search(rf"{param}\s*=\s*([-+]?\d*\.?\d+)", law_line)
            if m:
                val = float(m.group(1))
                params[param] = int(round(val)) if kind == "int" else round(val, 3)
    confidence = None
    alt_family = None
    why = None
    if self_line:
        m_conf = re.search(r"confidence\s*=\s*([-+]?\d*\.?\d+)", self_line)
        m_alt = re.search(r"alt_family\s*=\s*([a-zA-Z_]+)", self_line)
        m_why = re.search(r"why\s*=\s*(.*)$", self_line)
        if m_conf:
            confidence = float(m_conf.group(1))
        if m_alt:
            alt_family = m_alt.group(1)
        if m_why:
            why = m_why.group(1).strip()
    return {
        "law_line": law_line,
        "self_line": self_line,
        "family": family,
        "params": params,
        "confidence": confidence,
        "alt_family": alt_family,
        "why": why,
    }


def parameter_match_score(gold_family: str, gold_params: Dict[str, Any], pred_family: Optional[str], pred_params: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
    if pred_family != gold_family:
        return 0.0, False, {"status": "wrong_family"}

    specs = PARAM_SPECS[gold_family]
    per_param = {}
    good = True
    total = 0.0
    count = 0
    for name, kind in specs:
        g = gold_params[name]
        p = pred_params.get(name)
        if p is None:
            per_param[name] = {"present": False, "score": 0.0}
            good = False
            count += 1
            continue
        if kind == "int":
            diff = abs(int(p) - int(g))
            score = 1.0 if diff == 0 else max(0.0, 1.0 - diff / max(1.0, abs(g)))
            ok = diff == 0
        else:
            denom = max(0.05, abs(float(g)))
            rel = abs(float(p) - float(g)) / denom
            score = max(0.0, 1.0 - rel)
            ok = rel <= 0.20
        per_param[name] = {"present": True, "gold": g, "pred": p, "score": round(score, 4), "ok": ok}
        total += score
        count += 1
        good = good and ok
    return (total / max(1, count)), good, per_param


def evaluate_prediction_rows(gold_rows: Sequence[Dict[str, Any]], pred_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pred_map = {row["id"]: row for row in pred_rows}
    details = []
    family_correct = 0
    param_exact = 0
    total_param_score = 0.0
    missing = 0
    parse_fail = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for gold in gold_rows:
        gid = gold["id"]
        prow = pred_map.get(gid)
        if prow is None:
            missing += 1
            details.append({"id": gid, "status": "missing"})
            continue
        parsed = parse_prediction_text(prow.get("prediction", ""))
        pred_family = parsed["family"]
        if pred_family is None:
            parse_fail += 1
        confusion[gold["family"]][pred_family or "<none>"] += 1
        fam_ok = pred_family == gold["family"]
        if fam_ok:
            family_correct += 1
        param_score, param_ok, param_detail = parameter_match_score(gold["family"], gold["params"], pred_family, parsed["params"])
        total_param_score += param_score
        if param_ok:
            param_exact += 1
        details.append({
            "id": gid,
            "gold_family": gold["family"],
            "pred_family": pred_family,
            "family_correct": fam_ok,
            "param_exact": param_ok,
            "param_score": round(param_score, 4),
            "param_detail": param_detail,
            "confidence": parsed["confidence"],
            "alt_family": parsed["alt_family"],
            "why": parsed["why"],
        })

    n = max(1, len(gold_rows))
    report = {
        "num_examples": len(gold_rows),
        "family_accuracy": round(family_correct / n, 4),
        "param_exact_accuracy": round(param_exact / n, 4),
        "mean_param_score": round(total_param_score / n, 4),
        "overall_score": round((family_correct / n) * 0.6 + (total_param_score / n) * 0.4, 4),
        "missing_predictions": missing,
        "parse_failures": parse_fail,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "details": details,
    }
    return report


class PrototypeBaseline:
    def __init__(self, k: int = 5):
        self.k = k
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.train_rows: List[Example] = []
        self.train_vecs: Optional[np.ndarray] = None
        self.family_defaults: Dict[str, Dict[str, Any]] = {}
        self.family_centroids: Dict[str, np.ndarray] = {}

    def fit(self, train_rows: Sequence[Example]) -> None:
        self.train_rows = list(train_rows)
        x = np.stack([feature_vector(r) for r in train_rows])
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.train_vecs = (x - self.mean_) / self.std_
        self.family_defaults = parameter_defaults_by_family(train_rows)
        for fam in FAMILY_ORDER:
            idxs = [i for i, r in enumerate(train_rows) if r.family == fam]
            if idxs:
                self.family_centroids[fam] = self.train_vecs[idxs].mean(axis=0)

    def _standardize(self, row: Example) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (feature_vector(row) - self.mean_) / self.std_

    def _neighbor_votes(self, vec: np.ndarray) -> Tuple[List[int], List[float], Dict[str, float]]:
        assert self.train_vecs is not None
        dists = np.linalg.norm(self.train_vecs - vec[None, :], axis=1)
        k = min(self.k, len(self.train_rows))
        idxs = np.argsort(dists)[:k]
        weights: Dict[str, float] = defaultdict(float)
        dist_list: List[float] = []
        for idx in idxs:
            w = 1.0 / (1e-6 + float(dists[idx]))
            fam = self.train_rows[int(idx)].family
            weights[fam] += w
            dist_list.append(float(dists[idx]))
        return idxs.tolist(), dist_list, dict(weights)

    def _predict_params(self, family: str, vec: np.ndarray) -> Dict[str, Any]:
        assert self.train_vecs is not None
        fam_idxs = [i for i, r in enumerate(self.train_rows) if r.family == family]
        if not fam_idxs:
            return dict(self.family_defaults.get(family, {}))
        dists = np.linalg.norm(self.train_vecs[fam_idxs] - vec[None, :], axis=1)
        order = np.argsort(dists)[: min(self.k, len(fam_idxs))]
        neighbor_indices = [fam_idxs[int(i)] for i in order]
        weights = np.array([1.0 / (1e-6 + float(dists[int(i)])) for i in order], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        out: Dict[str, Any] = {}
        for param, kind in PARAM_SPECS[family]:
            vals = np.array([float(self.train_rows[idx].params[param]) for idx in neighbor_indices], dtype=float)
            if vals.size == 0:
                out[param] = self.family_defaults.get(family, {}).get(param, 0)
            else:
                val = float(np.dot(weights, vals))
                out[param] = int(round(val)) if kind == "int" else round(val, 3)
        return out

    def predict_structured(self, row: Example) -> Dict[str, Any]:
        vec = self._standardize(row)
        _, _, votes = self._neighbor_votes(vec)
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        pred_family = ranked[0][0] if ranked else FAMILY_ORDER[0]
        alt_family = ranked[1][0] if len(ranked) > 1 else "none"
        top = ranked[0][1] if ranked else 1.0
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = clamp((top / max(1e-6, top + runner)), 0.05, 0.99)
        pred_params = self._predict_params(pred_family, vec)
        why = explain_prediction(row.features, pred_family, alt_family)
        return {
            "family": pred_family,
            "params": pred_params,
            "confidence": round(confidence, 3),
            "alt_family": alt_family,
            "why": why,
        }

    def predict_text(self, row: Example) -> str:
        pred = self.predict_structured(row)
        line1 = format_law(pred["family"], pred["params"])
        line2 = f"SELF confidence={pred['confidence']:.2f}; alt_family={pred['alt_family']}; why={pred['why']}"
        return line1 + "\n" + line2


def explain_prediction(feats: Dict[str, float], family: str, alt_family: str) -> str:
    cl = feats["clustering"]
    deg_std = feats["degree_std"]
    dens = feats["density"]
    max_ratio = feats["max_degree_ratio"]
    lcc = feats["largest_component_frac"]
    assort = feats["assortativity"]

    if family == "barabasi_albert":
        return f"hub dominance and broad degree spread; max_degree_ratio={max_ratio:.2f}, degree_std={deg_std:.2f}"
    if family == "random_geometric":
        return f"strong locality signature with high clustering={cl:.2f} and fragmented/nearby connections; alt={alt_family}"
    if family == "watts_strogatz":
        return f"small-world mix of high clustering={cl:.2f} with mostly connected structure and moderate degree spread"
    if family == "stochastic_block":
        return f"community-like structure, assortativity={assort:.2f}, component_frac={lcc:.2f}, clustering={cl:.2f}"
    return f"roughly homogeneous randomness with density={dens:.3f}, clustering={cl:.2f}, and limited hub dominance"


def majority_baseline_predictions(rows: Sequence[Example], majority_family: str, family_default_params: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    preds = []
    params = family_default_params.get(majority_family, {})
    text = format_law(majority_family, params) + "\nSELF confidence=0.20; alt_family=none; why=majority baseline"
    for row in rows:
        preds.append({"id": row.example_id, "prediction": text})
    return preds


def run_baseline_predictions(model: PrototypeBaseline, rows: Sequence[Example]) -> List[Dict[str, Any]]:
    return [{"id": row.example_id, "prediction": model.predict_text(row)} for row in rows]


def select_best_k(train_rows: Sequence[Example], val_rows: Sequence[Example], candidate_ks: Sequence[int]) -> Tuple[int, List[Dict[str, Any]]]:
    scored = []
    best_k = candidate_ks[0]
    best_score = -1.0
    for k in candidate_ks:
        model = PrototypeBaseline(k=k)
        model.fit(train_rows)
        preds = run_baseline_predictions(model, val_rows)
        report = evaluate_prediction_rows([r.to_gold() for r in val_rows], preds)
        scored.append({"k": k, **{m: report[m] for m in ["family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score"]}})
        if report["overall_score"] > best_score:
            best_score = report["overall_score"]
            best_k = k
    return best_k, scored


def report_stage_zero(run_dir: Path, train_rows: Sequence[Example], val_rows: Sequence[Example], test_rows: Sequence[Example]) -> Dict[str, Any]:
    train_summary = dataset_summary(train_rows)
    val_summary = dataset_summary(val_rows)
    test_summary = dataset_summary(test_rows)
    majority_family = max(train_summary["family_counts"].items(), key=lambda kv: kv[1])[0]
    chance = round(1.0 / max(1, len(train_summary["family_counts"])), 4)
    majority_baseline = round(train_summary["family_counts"][majority_family] / max(1, len(train_rows)), 4)
    report = {
        "stage": "stage_0",
        "train_summary": train_summary,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "majority_family": majority_family,
        "chance_family_accuracy": chance,
        "majority_family_accuracy_estimate": majority_baseline,
    }
    json_dump(report, run_dir / "reports" / "stage0_benchmark.json")

    section("STAGE 0 BENCHMARK")
    print_dataset_summary("TRAIN", train_summary)
    print_dataset_summary("VAL", val_summary)
    print_dataset_summary("TEST", test_summary)
    print(json.dumps({
        "majority_family": majority_family,
        "chance_family_accuracy": chance,
        "majority_family_accuracy_estimate": majority_baseline,
    }, indent=2))
    return report


def report_midstage(run_dir: Path, train_rows: Sequence[Example], val_rows: Sequence[Example]) -> Tuple[PrototypeBaseline, Dict[str, Any]]:
    majority_family = Counter(r.family for r in train_rows).most_common(1)[0][0]
    family_defaults = parameter_defaults_by_family(train_rows)
    majority_preds = majority_baseline_predictions(val_rows, majority_family, family_defaults)
    majority_report = evaluate_prediction_rows([r.to_gold() for r in val_rows], majority_preds)

    best_k, search_table = select_best_k(train_rows, val_rows, [1, 3, 5, 7, 9])
    model = PrototypeBaseline(k=best_k)
    model.fit(train_rows)
    val_preds = run_baseline_predictions(model, val_rows)
    val_report = evaluate_prediction_rows([r.to_gold() for r in val_rows], val_preds)

    report = {
        "stage": "midstage",
        "k_search": search_table,
        "selected_k": best_k,
        "majority_baseline": {
            k: majority_report[k] for k in ["family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score"]
        },
        "validation_baseline": {
            k: val_report[k] for k in ["family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score", "missing_predictions", "parse_failures"]
        },
        "validation_confusion": val_report["confusion"],
        "sample_predictions": val_preds[:5],
    }
    json_dump(report, run_dir / "reports" / "midstage_benchmark.json")

    section("MIDSTAGE BENCHMARK")
    print(json.dumps(report, indent=2))
    return model, report


def report_completed(run_dir: Path, model: PrototypeBaseline, test_rows: Sequence[Example]) -> Dict[str, Any]:
    preds = run_baseline_predictions(model, test_rows)
    report = evaluate_prediction_rows([r.to_gold() for r in test_rows], preds)
    json_dump(report, run_dir / "reports" / "completed_benchmark.json")
    write_jsonl(preds, run_dir / "baseline_predictions_test.jsonl")

    section("COMPLETED BENCHMARK")
    print(json.dumps({
        k: report[k] for k in ["num_examples", "family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score", "missing_predictions", "parse_failures"]
    }, indent=2))
    short_line("Sample baseline predictions")
    for row in preds[:5]:
        print(json.dumps(row, ensure_ascii=False))
    return report


def write_readme(run_dir: Path, config: Dict[str, Any], stage0: Dict[str, Any], mid: Dict[str, Any], final: Dict[str, Any]) -> None:
    txt = textwrap.dedent(
        f"""
        # Graph Law Benchmark Run

        ## Config
        ```json
        {json.dumps(config, indent=2)}
        ```

        ## Stage 0
        ```json
        {json.dumps(stage0, indent=2)}
        ```

        ## Midstage
        ```json
        {json.dumps(mid, indent=2)}
        ```

        ## Completed
        ```json
        {json.dumps({k: final[k] for k in ['num_examples','family_accuracy','param_exact_accuracy','mean_param_score','overall_score']}, indent=2)}
        ```

        Key files:
        - `train/train_node0_local.txt`
        - `train/train_node1_global.txt`
        - `train/train_node2_spectral.txt`
        - `train/train_node3_edge.txt`
        - `test/test_prompts.jsonl`
        - `test/test_gold.jsonl`
        - `baseline_predictions_test.jsonl`
        - `reports/stage0_benchmark.json`
        - `reports/midstage_benchmark.json`
        - `reports/completed_benchmark.json`
        """
    ).strip() + "\n"
    (run_dir / "RUN_SUMMARY.md").write_text(txt, encoding="utf-8")


def full_pipeline(
    out_dir: Path,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    min_nodes: int,
    max_nodes: int,
    spectrum_topk: int,
    edge_sample_max: int,
    families: Sequence[str],
) -> Dict[str, Any]:
    rng = random.Random(seed)
    ensure_dir(out_dir)

    config = {
        "train_examples": train_n,
        "val_examples": val_n,
        "test_examples": test_n,
        "output_dir": str(out_dir),
        "seed": seed,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "spectrum_topk": spectrum_topk,
        "edge_sample_max": edge_sample_max,
        "families": list(families),
    }
    json_dump(config, out_dir / "run_config.json")

    section("GENERATING DATA")
    train_rows = generate_examples(train_n, "train", families, rng, min_nodes, max_nodes, spectrum_topk, edge_sample_max, 0)
    val_rows = generate_examples(val_n, "val", families, rng, min_nodes, max_nodes, spectrum_topk, edge_sample_max, 1_000_000)
    test_rows = generate_examples(test_n, "test", families, rng, min_nodes, max_nodes, spectrum_topk, edge_sample_max, 2_000_000)

    section("WRITING FILES")
    write_split_files(train_rows, out_dir, "train")
    write_split_files(val_rows, out_dir, "val")
    write_split_files(test_rows, out_dir, "test")
    print(f"Wrote benchmark files to: {out_dir}")

    stage0 = report_stage_zero(out_dir, train_rows, val_rows, test_rows)
    model, mid = report_midstage(out_dir, train_rows, val_rows)
    final = report_completed(out_dir, model, test_rows)
    write_readme(out_dir, config, stage0, mid, final)

    return {
        "config": config,
        "stage0": stage0,
        "midstage": mid,
        "completed": final,
    }


def evaluate_external_predictions(gold_path: Path, pred_path: Path, out_path: Optional[Path]) -> Dict[str, Any]:
    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)
    report = evaluate_prediction_rows(gold_rows, pred_rows)
    if out_path is not None:
        json_dump(report, out_path)
    section("EXTERNAL EVALUATION")
    print(json.dumps({
        k: report[k] for k in ["num_examples", "family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score", "missing_predictions", "parse_failures"]
    }, indent=2))
    return report


def inspect_existing_benchmark(out_dir: Path) -> None:
    section("INSPECT EXISTING BENCHMARK")
    for rel in [
        "run_config.json",
        "reports/stage0_benchmark.json",
        "reports/midstage_benchmark.json",
        "reports/completed_benchmark.json",
    ]:
        p = out_dir / rel
        print(f"\n[{rel}]")
        if p.exists():
            print(p.read_text(encoding="utf-8")[:5000])
        else:
            print("missing")


def main() -> None:
    section("Graph Law Benchmark - End to End")
    print("Choose mode:")
    print("  1) Full pipeline: generate + benchmark + baseline + reports")
    print("  2) Generate benchmark only")
    print("  3) Evaluate external predictions")
    print("  4) Inspect an existing benchmark folder")
    mode = input("Select [1]: ").strip() or "1"

    if mode == "1":
        print("\nFull pipeline")
        out_dir = prompt_path("Output directory", "bench_graph_law_e2e")
        train_n = prompt_int("Train examples", 1200, 10)
        val_n = prompt_int("Validation examples", 200, 5)
        test_n = prompt_int("Test examples", 200, 5)
        seed = prompt_int("Random seed", 42)
        min_nodes = prompt_int("Minimum nodes per graph", 28, 8)
        max_nodes = prompt_int("Maximum nodes per graph", 72, min_nodes)
        spectrum_topk = prompt_int("How many top adjacency eigenvalues to keep", 6, 2, 6)
        edge_sample_max = prompt_int("Maximum sampled edges shown in EDGE_GLIMPSE", 40, 5)
        families = prompt_families(FAMILY_ORDER)
        results = full_pipeline(out_dir, train_n, val_n, test_n, seed, min_nodes, max_nodes, spectrum_topk, edge_sample_max, families)
        section("DONE")
        print(json.dumps({
            "output_dir": str(out_dir),
            "overall_score": results["completed"]["overall_score"],
            "family_accuracy": results["completed"]["family_accuracy"],
            "param_exact_accuracy": results["completed"]["param_exact_accuracy"],
            "summary_file": str(out_dir / "RUN_SUMMARY.md"),
        }, indent=2))

    elif mode == "2":
        print("\nGenerate benchmark only")
        out_dir = prompt_path("Output directory", "bench_graph_law_e2e")
        train_n = prompt_int("Train examples", 1200, 10)
        val_n = prompt_int("Validation examples", 200, 5)
        test_n = prompt_int("Test examples", 200, 5)
        seed = prompt_int("Random seed", 42)
        min_nodes = prompt_int("Minimum nodes per graph", 28, 8)
        max_nodes = prompt_int("Maximum nodes per graph", 72, min_nodes)
        spectrum_topk = prompt_int("How many top adjacency eigenvalues to keep", 6, 2, 6)
        edge_sample_max = prompt_int("Maximum sampled edges shown in EDGE_GLIMPSE", 40, 5)
        families = prompt_families(FAMILY_ORDER)
        rng = random.Random(seed)
        section("GENERATING DATA")
        train_rows = generate_examples(train_n, "train", families, rng, min_nodes, max_nodes, spectrum_topk, edge_sample_max, 0)
        val_rows = generate_examples(val_n, "val", families, rng, min_nodes, max_nodes, spectrum_topk, edge_sample_max, 1_000_000)
        test_rows = generate_examples(test_n, "test", families, rng, min_nodes, max_nodes, spectrum_topk, edge_sample_max, 2_000_000)
        write_split_files(train_rows, out_dir, "train")
        write_split_files(val_rows, out_dir, "val")
        write_split_files(test_rows, out_dir, "test")
        report_stage_zero(out_dir, train_rows, val_rows, test_rows)
        print(f"\nGenerated benchmark only at: {out_dir}")

    elif mode == "3":
        print("\nEvaluate external predictions")
        gold_path = prompt_path("Gold JSONL path", "bench_graph_law_e2e/test/test_gold.jsonl")
        pred_path = prompt_path("Prediction JSONL path", "my_predictions.jsonl")
        write_report = prompt_bool("Write detailed report JSON", True)
        out_path = prompt_path("Report output path", "eval_report.json") if write_report else None
        if not pred_path.exists():
            print(f"Prediction file not found: {pred_path}")
            print("Tip: copy the blank template and fill it in first.")
            sys.exit(1)
        evaluate_external_predictions(gold_path, pred_path, out_path)

    elif mode == "4":
        out_dir = prompt_path("Benchmark directory", "bench_graph_law_e2e")
        inspect_existing_benchmark(out_dir)

    else:
        print("Unknown selection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
