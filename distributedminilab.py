
#!/usr/bin/env python3
"""
EXOSFEAR MiniLab Distributed
A distributed graph-law benchmark harness for a small LAN cluster
(e.g. one Mac + one Dell) with coordinator/worker modes.

What this does:
- Generates a stronger graph-law benchmark with:
  - in-distribution train / val / test_standard
  - noisy test split
  - OOD-large test split
- Distributes graph generation + feature extraction across workers on your LAN
- Trains a central KNN-style baseline
- Distributes prediction back out to workers
- Surfaces Stage 0, Midstage, and Completed benchmark reports
- Writes prompts, gold files, blank templates, baseline predictions, and summaries

Dependencies:
    pip install networkx numpy

Security:
- Intended for trusted local networks only.
- Optional shared token for worker auth.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import secrets
import json
import math
import os
import random
import re
import socket
import statistics
import sys
import textwrap
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

# ------------------ basic io / prompts ------------------


def section(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def short_line(title: str) -> None:
    print(f"--- {title} {'-' * max(1, 70 - len(title))}")


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
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_int(label: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            v = default
        else:
            try:
                v = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if min_value is not None and v < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and v > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return v


def prompt_float(label: str, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            v = default
        else:
            try:
                v = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if min_value is not None and v < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and v > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return v


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
        except ValueError:
            pass
    return selected or list(default)


def prompt_worker_urls(default: str = "") -> List[str]:
    raw = input(f"Worker URLs, comma-separated [{default}]: ").strip()
    text = raw or default
    if not text:
        return []
    out = []
    for part in text.split(","):
        u = part.strip().rstrip("/")
        if u:
            out.append(u)
    return out


# ------------------ utilities ------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_mean(vals: Sequence[float], default: float = 0.0) -> float:
    return float(sum(vals) / len(vals)) if vals else float(default)


def safe_percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def safe_assortativity(g: nx.Graph) -> float:
    try:
        val = nx.degree_pearson_correlation_coefficient(g)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def safe_avg_path_lcc(g: nx.Graph) -> float:
    if g.number_of_nodes() <= 1 or g.number_of_edges() == 0:
        return 0.0
    comp_nodes = max(nx.connected_components(g), key=len)
    sub = g.subgraph(comp_nodes).copy()
    if sub.number_of_nodes() <= 1:
        return 0.0
    try:
        return float(nx.average_shortest_path_length(sub))
    except Exception:
        return 0.0


def top_adj_eigs(g: nx.Graph, topk: int = 6) -> List[float]:
    n = g.number_of_nodes()
    if n == 0:
        return [0.0] * topk
    arr = nx.to_numpy_array(g, dtype=float)
    try:
        vals = np.linalg.eigvalsh(arr)
        vals = np.sort(np.abs(vals))[::-1]
        out = [float(v) for v in vals[:topk]]
    except Exception:
        out = [0.0] * topk
    if len(out) < topk:
        out += [0.0] * (topk - len(out))
    return out


# ------------------ specs / rows ------------------


@dataclass
class ExampleSpec:
    example_id: str
    split: str
    family: str
    params: Dict[str, Any]
    n_nodes: int
    seed: int
    noise_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleRow:
    example_id: str
    split: str
    family: str
    params: Dict[str, Any]
    n_nodes: int
    seed: int
    noise_level: float
    features: Dict[str, float]
    prompt: str

    def to_gold(self) -> Dict[str, Any]:
        return {"id": self.example_id, "family": self.family, "params": self.params}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------ graph generation ------------------


def random_connected_geometric(n: int, radius: float, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    for _ in range(8):
        g = nx.random_geometric_graph(n, radius, seed=rng.randint(0, 10_000_000))
        if g.number_of_nodes() > 0 and nx.number_connected_components(g) <= max(2, n // 25):
            return nx.Graph(g)
        radius *= 1.05
    return nx.Graph(g)


def sample_family_params(family: str, n: int, rng: random.Random) -> Dict[str, Any]:
    if family == "erdos_renyi":
        # keep away from trivial extremes
        p = round(rng.uniform(0.04, 0.22), 3)
        return {"p": p}
    if family == "barabasi_albert":
        max_m = max(2, min(8, n // 6))
        return {"m": rng.randint(2, max_m)}
    if family == "watts_strogatz":
        kmax = max(4, min(14, (n // 2) - ((n // 2) % 2)))
        candidates = [k for k in range(4, kmax + 1, 2)]
        k = rng.choice(candidates or [4])
        beta = round(rng.uniform(0.03, 0.38), 3)
        return {"k": k, "beta": beta}
    if family == "stochastic_block":
        max_blocks = min(4, max(2, n // 12))
        blocks = rng.randint(2, max_blocks)
        # safe block partition
        base = [n // blocks] * blocks
        for i in range(n % blocks):
            base[i] += 1
        rng.shuffle(base)
        p_in = round(rng.uniform(0.22, 0.55), 3)
        p_out = round(rng.uniform(0.01, min(0.18, p_in * 0.45)), 3)
        return {"blocks": blocks, "p_in": p_in, "p_out": p_out}
    if family == "random_geometric":
        radius = round(rng.uniform(0.16, 0.33), 3)
        return {"radius": radius}
    raise ValueError(f"Unsupported family: {family}")


def build_graph_from_spec(spec: ExampleSpec) -> nx.Graph:
    fam = spec.family
    p = spec.params
    seed = int(spec.seed)
    n = int(spec.n_nodes)

    if fam == "erdos_renyi":
        g = nx.erdos_renyi_graph(n, float(p["p"]), seed=seed)
    elif fam == "barabasi_albert":
        g = nx.barabasi_albert_graph(n, int(p["m"]), seed=seed)
    elif fam == "watts_strogatz":
        g = nx.watts_strogatz_graph(n, int(p["k"]), float(p["beta"]), seed=seed)
    elif fam == "stochastic_block":
        blocks = int(p["blocks"])
        sizes = [n // blocks] * blocks
        for i in range(n % blocks):
            sizes[i] += 1
        probs = []
        for i in range(blocks):
            row = []
            for j in range(blocks):
                row.append(float(p["p_in"]) if i == j else float(p["p_out"]))
            probs.append(row)
        g = nx.stochastic_block_model(sizes, probs, seed=seed)
        g = nx.Graph(g)
    elif fam == "random_geometric":
        g = random_connected_geometric(n, float(p["radius"]), seed=seed)
    else:
        raise ValueError(f"Unsupported family: {fam}")

    if spec.noise_level > 0:
        g = apply_edge_noise(g, spec.noise_level, seed + 1337)

    # normalize labels
    g = nx.convert_node_labels_to_integers(g)
    return g


def apply_edge_noise(g: nx.Graph, noise_level: float, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    h = g.copy()
    n = h.number_of_nodes()
    if n <= 1:
        return h
    edges = list(h.edges())
    num_flip = max(1, int(round(len(edges) * noise_level)))
    rng.shuffle(edges)
    remove = edges[: min(len(edges), num_flip // 2)]
    for u, v in remove:
        if h.has_edge(u, v):
            h.remove_edge(u, v)
    target_add = len(remove)
    tries = 0
    while target_add > 0 and tries < num_flip * 20 + 100:
        u = rng.randrange(n)
        v = rng.randrange(n)
        tries += 1
        if u == v or h.has_edge(u, v):
            continue
        h.add_edge(u, v)
        target_add -= 1
    return h


# ------------------ features / prompts ------------------


def graph_features(g: nx.Graph, topk: int = 6) -> Dict[str, float]:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    deg = [d for _, d in g.degree()]
    density = float(nx.density(g)) if n > 1 else 0.0
    clustering = float(nx.average_clustering(g)) if m > 0 else 0.0
    transitivity = float(nx.transitivity(g)) if m > 0 else 0.0
    largest_comp_frac = 0.0
    if n > 0:
        largest_comp_frac = len(max(nx.connected_components(g), key=len)) / n if m > 0 else (1.0 / n)
    triangles = sum(nx.triangles(g).values()) / 3 if m > 0 else 0.0
    tri_dens = float(triangles / max(1, n))
    eigs = top_adj_eigs(g, topk=topk)

    feats = {
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": density,
        "avg_degree": safe_mean([float(d) for d in deg]),
        "degree_std": float(statistics.pstdev(deg)) if len(deg) > 1 else 0.0,
        "max_degree_ratio": (max(deg) / max(1, n - 1)) if deg else 0.0,
        "clustering": clustering,
        "transitivity": transitivity,
        "largest_component_frac": float(largest_comp_frac),
        "avg_path_lcc": safe_avg_path_lcc(g),
        "assortativity": safe_assortativity(g),
        "triangle_density": tri_dens,
        "deg_p10": safe_percentile(deg, 10),
        "deg_p25": safe_percentile(deg, 25),
        "deg_p50": safe_percentile(deg, 50),
        "deg_p75": safe_percentile(deg, 75),
        "deg_p90": safe_percentile(deg, 90),
    }
    for i in range(6):
        feats[f"spectrum_{i+1}"] = float(eigs[i] if i < len(eigs) else 0.0)
    return feats


def sample_edges_text(g: nx.Graph, max_edges: int, seed: int) -> str:
    edges = list(g.edges())
    rng = random.Random(seed)
    rng.shuffle(edges)
    take = edges[: min(max_edges, len(edges))]
    if not take:
        return "none"
    return ", ".join(f"({u},{v})" for u, v in take)


def render_prompt(spec: ExampleSpec, feats: Dict[str, float], g: nx.Graph, spectrum_topk: int = 6, edge_sample_max: int = 40) -> str:
    local = textwrap.dedent(f"""\
    LOCAL_SHAPE
    n_nodes={int(feats["n_nodes"])}
    n_edges={int(feats["n_edges"])}
    density={feats["density"]:.4f}
    avg_degree={feats["avg_degree"]:.3f}
    degree_std={feats["degree_std"]:.3f}
    max_degree_ratio={feats["max_degree_ratio"]:.3f}
    deg_p10={feats["deg_p10"]:.2f}
    deg_p25={feats["deg_p25"]:.2f}
    deg_p50={feats["deg_p50"]:.2f}
    deg_p75={feats["deg_p75"]:.2f}
    deg_p90={feats["deg_p90"]:.2f}
    """).strip()

    global_txt = textwrap.dedent(f"""\
    GLOBAL_SHAPE
    clustering={feats["clustering"]:.4f}
    transitivity={feats["transitivity"]:.4f}
    largest_component_frac={feats["largest_component_frac"]:.4f}
    avg_path_lcc={feats["avg_path_lcc"]:.4f}
    assortativity={feats["assortativity"]:.4f}
    triangle_density={feats["triangle_density"]:.4f}
    noise_level={spec.noise_level:.3f}
    """).strip()

    spectral_lines = [f"spectrum_{i+1}={feats[f'spectrum_{i+1}']:.4f}" for i in range(spectrum_topk)]
    spectral_txt = "SPECTRAL_SHAPE\n" + "\n".join(spectral_lines)

    edge_txt = "EDGE_GLIMPSE\n" + sample_edges_text(g, edge_sample_max, spec.seed + 91)

    instr = textwrap.dedent("""\
    TASK
    Infer the generating graph law family and parameters.

    Reply in exactly two lines.

    Line 1:
    LAW family=<family>; <param1>=<value>; <param2>=<value>

    Line 2:
    SELF confidence=<0 to 1>; alt_family=<family>; why=<brief reason>

    Do not add any extra text.
    """).strip()

    return "\n\n".join([local, global_txt, spectral_txt, edge_txt, instr])


def extract_row_from_spec(spec: ExampleSpec, spectrum_topk: int = 6, edge_sample_max: int = 40) -> ExampleRow:
    g = build_graph_from_spec(spec)
    feats = graph_features(g, topk=max(6, spectrum_topk))
    prompt = render_prompt(spec, feats, g, spectrum_topk=spectrum_topk, edge_sample_max=edge_sample_max)
    return ExampleRow(
        example_id=spec.example_id,
        split=spec.split,
        family=spec.family,
        params=spec.params,
        n_nodes=spec.n_nodes,
        seed=spec.seed,
        noise_level=spec.noise_level,
        features=feats,
        prompt=prompt,
    )


# ------------------ dataset generation ------------------


def balanced_family_sequence(total: int, families: Sequence[str], rng: random.Random) -> List[str]:
    base = []
    while len(base) < total:
        base.extend(list(families))
    base = base[:total]
    rng.shuffle(base)
    return base


def make_specs_for_split(
    split: str,
    count: int,
    families: Sequence[str],
    n_min: int,
    n_max: int,
    seed: int,
    noise_range: Tuple[float, float] = (0.0, 0.0),
) -> List[ExampleSpec]:
    rng = random.Random(seed)
    fams = balanced_family_sequence(count, families, rng)
    specs: List[ExampleSpec] = []
    for i in range(count):
        fam = fams[i]
        n = rng.randint(n_min, n_max)
        params = sample_family_params(fam, n, rng)
        noise = round(rng.uniform(noise_range[0], noise_range[1]), 3) if noise_range[1] > 0 else 0.0
        ex_id = f"{split[:2].upper()}{seed % 1000:03d}{i:05d}"
        specs.append(ExampleSpec(
            example_id=ex_id,
            split=split,
            family=fam,
            params=params,
            n_nodes=n,
            seed=rng.randint(0, 10_000_000),
            noise_level=noise,
        ))
    return specs


# ------------------ distributed execution ------------------


WORKER_CONTEXT = {
    "worker_name": socket.gethostname(),
    "token": "",
    "version": "exosfearminilab-distributed-1",
}


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class MiniLabWorkerHandler(BaseHTTPRequestHandler):
    server_version = "ExosfearMiniLabWorker/1.0"

    def _check_token(self) -> bool:
        expected = WORKER_CONTEXT.get("token", "")
        if not expected:
            return True
        got = self.headers.get("X-Token", "")
        return got == expected

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            if not self._check_token():
                _json_response(self, 403, {"ok": False, "error": "bad token"})
                return
            _json_response(self, 200, {
                "ok": True,
                "worker_name": WORKER_CONTEXT["worker_name"],
                "version": WORKER_CONTEXT["version"],
                "pid": os.getpid(),
            })
            return
        _json_response(self, 404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if not self._check_token():
            _json_response(self, 403, {"ok": False, "error": "bad token"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            _json_response(self, 400, {"ok": False, "error": "invalid json"})
            return

        started = time.time()
        try:
            if self.path == "/extract":
                specs = [ExampleSpec(**row) for row in payload.get("specs", [])]
                spectrum_topk = int(payload.get("spectrum_topk", 6))
                edge_sample_max = int(payload.get("edge_sample_max", 40))
                rows = [extract_row_from_spec(s, spectrum_topk=spectrum_topk, edge_sample_max=edge_sample_max).to_dict() for s in specs]
                _json_response(self, 200, {
                    "ok": True,
                    "rows": rows,
                    "worker_name": WORKER_CONTEXT["worker_name"],
                    "seconds": round(time.time() - started, 4),
                })
                return

            if self.path == "/predict":
                model_pkg = payload["model"]
                rows = [ExampleRow(**row) for row in payload.get("rows", [])]
                model = PrototypeBaseline.from_package(model_pkg)
                preds = [{"id": row.example_id, "prediction": model.predict_text(row)} for row in rows]
                _json_response(self, 200, {
                    "ok": True,
                    "predictions": preds,
                    "worker_name": WORKER_CONTEXT["worker_name"],
                    "seconds": round(time.time() - started, 4),
                })
                return

            _json_response(self, 404, {"ok": False, "error": "not found"})
        except Exception as e:
            _json_response(self, 500, {"ok": False, "error": repr(e)})

    def log_message(self, fmt: str, *args) -> None:
        return


def start_worker(host: str, port: int, token: str = "") -> None:
    WORKER_CONTEXT["token"] = token or ""
    WORKER_CONTEXT["worker_name"] = socket.gethostname()
    httpd = ThreadingHTTPServer((host, port), MiniLabWorkerHandler)
    section("WORKER READY")
    print(json.dumps({
        "worker_name": WORKER_CONTEXT["worker_name"],
        "bind": f"http://{host}:{port}",
        "token_enabled": bool(token),
        "pid": os.getpid(),
    }, indent=2))
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping worker...")


def http_json(url: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None, token: str = "", timeout: int = 300) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method=method)
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if token:
        req.add_header("X-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chunked(seq: Sequence[Any], size: int) -> List[List[Any]]:
    out = []
    for i in range(0, len(seq), size):
        out.append(list(seq[i:i + size]))
    return out


def ping_workers(worker_urls: Sequence[str], token: str) -> List[Dict[str, Any]]:
    statuses = []
    for url in worker_urls:
        try:
            data = http_json(url.rstrip("/") + "/health", method="GET", token=token, timeout=10)
            statuses.append({"url": url, **data})
        except Exception as e:
            statuses.append({"url": url, "ok": False, "error": repr(e)})
    return statuses


def run_distributed_extract(
    specs: Sequence[ExampleSpec],
    worker_urls: Sequence[str],
    token: str,
    chunk_size: int,
    spectrum_topk: int,
    edge_sample_max: int,
    include_local: bool = True,
) -> Tuple[List[ExampleRow], List[Dict[str, Any]]]:
    if not specs:
        return [], []

    workers = (["__local__"] if include_local else []) + list(worker_urls)
    if not workers:
        workers = ["__local__"]

    chunks = chunked(list(specs), max(1, chunk_size))
    results: List[ExampleRow] = []
    stats: List[Dict[str, Any]] = []
    grouped: Dict[str, List[ExampleSpec]] = defaultdict(list)

    for idx, ch in enumerate(chunks):
        worker = workers[idx % len(workers)]
        grouped[worker].extend(ch)

    for worker, subset in grouped.items():
        started = time.time()
        if worker == "__local__":
            rows = [extract_row_from_spec(s, spectrum_topk=spectrum_topk, edge_sample_max=edge_sample_max) for s in subset]
            elapsed = time.time() - started
            results.extend(rows)
            stats.append({
                "worker": "local",
                "num_specs": len(subset),
                "seconds": round(elapsed, 4),
                "rows_per_sec": round(len(subset) / max(elapsed, 1e-6), 3),
            })
        else:
            payload = {
                "specs": [s.to_dict() for s in subset],
                "spectrum_topk": spectrum_topk,
                "edge_sample_max": edge_sample_max,
            }
            data = http_json(worker.rstrip("/") + "/extract", method="POST", payload=payload, token=token, timeout=1800)
            rows = [ExampleRow(**r) for r in data["rows"]]
            elapsed = float(data.get("seconds", time.time() - started))
            results.extend(rows)
            stats.append({
                "worker": data.get("worker_name", worker),
                "url": worker,
                "num_specs": len(subset),
                "seconds": round(elapsed, 4),
                "rows_per_sec": round(len(subset) / max(elapsed, 1e-6), 3),
            })

    results.sort(key=lambda r: r.example_id)
    return results, stats


def run_distributed_predict(
    model_pkg: Dict[str, Any],
    rows: Sequence[ExampleRow],
    worker_urls: Sequence[str],
    token: str,
    chunk_size: int,
    include_local: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []

    workers = (["__local__"] if include_local else []) + list(worker_urls)
    if not workers:
        workers = ["__local__"]

    chunks = chunked(list(rows), max(1, chunk_size))
    grouped: Dict[str, List[ExampleRow]] = defaultdict(list)
    for idx, ch in enumerate(chunks):
        worker = workers[idx % len(workers)]
        grouped[worker].extend(ch)

    results: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []

    for worker, subset in grouped.items():
        started = time.time()
        if worker == "__local__":
            model = PrototypeBaseline.from_package(model_pkg)
            preds = [{"id": row.example_id, "prediction": model.predict_text(row)} for row in subset]
            elapsed = time.time() - started
            results.extend(preds)
            stats.append({
                "worker": "local",
                "num_rows": len(subset),
                "seconds": round(elapsed, 4),
                "rows_per_sec": round(len(subset) / max(elapsed, 1e-6), 3),
            })
        else:
            payload = {"model": model_pkg, "rows": [r.to_dict() for r in subset]}
            data = http_json(worker.rstrip("/") + "/predict", method="POST", payload=payload, token=token, timeout=1800)
            elapsed = float(data.get("seconds", time.time() - started))
            results.extend(data["predictions"])
            stats.append({
                "worker": data.get("worker_name", worker),
                "url": worker,
                "num_rows": len(subset),
                "seconds": round(elapsed, 4),
                "rows_per_sec": round(len(subset) / max(elapsed, 1e-6), 3),
            })

    results.sort(key=lambda r: r["id"])
    return results, stats


# ------------------ model / evaluation ------------------


def feature_vector(row: ExampleRow) -> np.ndarray:
    return np.array([float(row.features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)


def parameter_defaults_by_family(train_rows: Sequence[ExampleRow]) -> Dict[str, Dict[str, Any]]:
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


def format_law(family: str, params: Dict[str, Any]) -> str:
    bits = [f"family={family}"]
    for param, kind in PARAM_SPECS.get(family, []):
        if param not in params:
            continue
        v = params[param]
        if kind == "int":
            bits.append(f"{param}={int(v)}")
        else:
            bits.append(f"{param}={float(v):.3f}")
    return "LAW " + "; ".join(bits)


def explain_prediction(feats: Dict[str, float], family: str, alt_family: str) -> str:
    cl = feats["clustering"]
    deg_std = feats["degree_std"]
    dens = feats["density"]
    max_ratio = feats["max_degree_ratio"]
    lcc = feats["largest_component_frac"]
    assort = feats["assortativity"]
    if family == "barabasi_albert":
        return f"hub-heavy degree shape; max_degree_ratio={max_ratio:.2f}, degree_std={deg_std:.2f}"
    if family == "random_geometric":
        return f"locality signature with clustering={cl:.2f} and spatial-like connectivity; alt={alt_family}"
    if family == "watts_strogatz":
        return f"small-world profile with elevated clustering={cl:.2f} and moderate degree spread"
    if family == "stochastic_block":
        return f"community-like signal with assortativity={assort:.2f}, lcc_frac={lcc:.2f}, clustering={cl:.2f}"
    return f"roughly homogeneous random pattern with density={dens:.3f} and limited hub dominance"


class PrototypeBaseline:
    def __init__(self, k: int = 5):
        self.k = k
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.train_rows: List[ExampleRow] = []
        self.train_vecs: Optional[np.ndarray] = None
        self.family_defaults: Dict[str, Dict[str, Any]] = {}

    def fit(self, train_rows: Sequence[ExampleRow]) -> None:
        self.train_rows = list(train_rows)
        x = np.stack([feature_vector(r) for r in train_rows], axis=0)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.train_vecs = (x - self.mean_) / self.std_
        self.family_defaults = parameter_defaults_by_family(train_rows)

    def package(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "train_vecs": self.train_vecs.tolist() if self.train_vecs is not None else None,
            "train_meta": [{"family": r.family, "params": r.params, "features": r.features} for r in self.train_rows],
            "family_defaults": self.family_defaults,
        }

    @classmethod
    def from_package(cls, pkg: Dict[str, Any]) -> "PrototypeBaseline":
        obj = cls(k=int(pkg["k"]))
        obj.mean_ = np.asarray(pkg["mean"], dtype=float)
        obj.std_ = np.asarray(pkg["std"], dtype=float)
        obj.train_vecs = np.asarray(pkg["train_vecs"], dtype=float)
        obj.family_defaults = pkg.get("family_defaults", {})
        obj.train_rows = [
            ExampleRow(
                example_id=f"TR{i:06d}",
                split="train",
                family=meta["family"],
                params=meta["params"],
                n_nodes=int(meta["features"].get("n_nodes", 0)),
                seed=0,
                noise_level=0.0,
                features=meta["features"],
                prompt="",
            )
            for i, meta in enumerate(pkg["train_meta"])
        ]
        return obj

    def _standardize(self, row: ExampleRow) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (feature_vector(row) - self.mean_) / self.std_

    def _neighbor_votes(self, vec: np.ndarray) -> List[Tuple[int, float]]:
        assert self.train_vecs is not None
        dists = np.linalg.norm(self.train_vecs - vec[None, :], axis=1)
        order = np.argsort(dists)[: min(self.k, len(self.train_rows))]
        out = []
        for idx in order:
            out.append((int(idx), float(dists[idx])))
        return out

    def _predict_params(self, family: str, vec: np.ndarray) -> Dict[str, Any]:
        assert self.train_vecs is not None
        fam_idxs = [i for i, r in enumerate(self.train_rows) if r.family == family]
        if not fam_idxs:
            return dict(self.family_defaults.get(family, {}))
        dists = np.linalg.norm(self.train_vecs[fam_idxs] - vec[None, :], axis=1)
        order = np.argsort(dists)[: min(self.k, len(fam_idxs))]
        idxs = [fam_idxs[int(i)] for i in order]
        weights = np.array([1.0 / (1e-6 + float(dists[int(i)])) for i in order], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        params_out: Dict[str, Any] = {}
        for param, kind in PARAM_SPECS[family]:
            vals = np.array([float(self.train_rows[idx].params[param]) for idx in idxs], dtype=float)
            val = float(np.dot(weights, vals)) if vals.size else float(self.family_defaults.get(family, {}).get(param, 0))
            params_out[param] = int(round(val)) if kind == "int" else round(val, 3)
        return params_out

    def predict_structured(self, row: ExampleRow) -> Dict[str, Any]:
        vec = self._standardize(row)
        neigh = self._neighbor_votes(vec)
        fam_votes: Dict[str, float] = defaultdict(float)
        for idx, dist in neigh:
            w = 1.0 / (1e-6 + dist)
            fam_votes[self.train_rows[idx].family] += w
        ranked = sorted(fam_votes.items(), key=lambda kv: kv[1], reverse=True)
        pred_family = ranked[0][0] if ranked else FAMILY_ORDER[0]
        alt_family = ranked[1][0] if len(ranked) > 1 else "none"
        top = ranked[0][1] if ranked else 1.0
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = clamp(top / max(1e-6, top + runner), 0.05, 0.99)
        pred_params = self._predict_params(pred_family, vec)
        why = explain_prediction(row.features, pred_family, alt_family)
        return {
            "family": pred_family,
            "params": pred_params,
            "confidence": round(confidence, 3),
            "alt_family": alt_family,
            "why": why,
        }

    def predict_text(self, row: ExampleRow) -> str:
        pred = self.predict_structured(row)
        return (
            format_law(pred["family"], pred["params"]) + "\n" +
            f"SELF confidence={pred['confidence']:.2f}; alt_family={pred['alt_family']}; why={pred['why']}"
        )


def parse_prediction_text(text: str) -> Dict[str, Any]:
    law_line = ""
    self_line = ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("LAW "):
            law_line = s
        elif s.upper().startswith("SELF "):
            self_line = s
    if not law_line:
        law_line = text.strip().splitlines()[0].strip() if text.strip() else ""

    family_match = re.search(r"family\s*=\s*([a-zA-Z_]+)", law_line)
    family = family_match.group(1).strip() if family_match else None
    params = {}
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


def parameter_match_score(gold_family: str, gold_params: Dict[str, Any], pred_family: Optional[str], pred_params: Dict[str, Any]) -> Tuple[float, bool]:
    if pred_family != gold_family:
        return 0.0, False
    total = 0.0
    count = 0
    exact = True
    for name, kind in PARAM_SPECS[gold_family]:
        g = gold_params[name]
        p = pred_params.get(name)
        count += 1
        if p is None:
            exact = False
            continue
        if kind == "int":
            diff = abs(int(p) - int(g))
            total += 1.0 if diff == 0 else max(0.0, 1.0 - diff / max(1.0, abs(g)))
            exact = exact and (diff == 0)
        else:
            denom = max(0.05, abs(float(g)))
            rel = abs(float(p) - float(g)) / denom
            total += max(0.0, 1.0 - rel)
            exact = exact and (rel <= 0.20)
    return total / max(1, count), exact


def evaluate_prediction_rows(gold_rows: Sequence[Dict[str, Any]], pred_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pred_map = {r["id"]: r for r in pred_rows}
    details = []
    family_correct = 0
    param_exact = 0
    mean_param_score = 0.0
    missing = 0
    parse_failures = 0
    confusion = defaultdict(lambda: defaultdict(int))
    confidence_vals = []

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
            parse_failures += 1
        confusion[gold["family"]][pred_family or "<none>"] += 1
        fam_ok = pred_family == gold["family"]
        if fam_ok:
            family_correct += 1
        param_score, exact_ok = parameter_match_score(gold["family"], gold["params"], pred_family, parsed["params"])
        mean_param_score += param_score
        if exact_ok:
            param_exact += 1
        conf = parsed.get("confidence")
        if conf is not None:
            confidence_vals.append(float(conf))
        details.append({
            "id": gid,
            "gold_family": gold["family"],
            "pred_family": pred_family,
            "family_correct": fam_ok,
            "param_exact": exact_ok,
            "param_score": round(param_score, 4),
            "confidence": conf,
            "alt_family": parsed.get("alt_family"),
            "why": parsed.get("why"),
        })

    n = max(1, len(gold_rows))
    return {
        "num_examples": len(gold_rows),
        "family_accuracy": round(family_correct / n, 4),
        "param_exact_accuracy": round(param_exact / n, 4),
        "mean_param_score": round(mean_param_score / n, 4),
        "overall_score": round((family_correct / n) * 0.6 + (mean_param_score / n) * 0.4, 4),
        "avg_confidence": round(sum(confidence_vals) / len(confidence_vals), 4) if confidence_vals else None,
        "missing_predictions": missing,
        "parse_failures": parse_failures,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "details": details,
    }


def select_best_k(train_rows: Sequence[ExampleRow], val_rows: Sequence[ExampleRow], ks: Sequence[int]) -> Tuple[int, List[Dict[str, Any]]]:
    table = []
    best_k = ks[0]
    best_score = -1.0
    for k in ks:
        model = PrototypeBaseline(k=k)
        model.fit(train_rows)
        preds = [{"id": r.example_id, "prediction": model.predict_text(r)} for r in val_rows]
        rep = evaluate_prediction_rows([r.to_gold() for r in val_rows], preds)
        row = {"k": k, "overall_score": rep["overall_score"], "family_accuracy": rep["family_accuracy"], "param_exact_accuracy": rep["param_exact_accuracy"]}
        table.append(row)
        if rep["overall_score"] > best_score:
            best_score = rep["overall_score"]
            best_k = k
    return best_k, table


# ------------------ reports / writing ------------------


def dataset_summary(rows: Sequence[ExampleRow]) -> Dict[str, Any]:
    fam_counts = Counter(r.family for r in rows)
    return {
        "num_examples": len(rows),
        "family_counts": dict(fam_counts),
        "avg_nodes": round(safe_mean([r.n_nodes for r in rows]), 3),
        "avg_density": round(safe_mean([r.features["density"] for r in rows]), 4),
        "avg_clustering": round(safe_mean([r.features["clustering"] for r in rows]), 4),
        "avg_noise_level": round(safe_mean([r.noise_level for r in rows]), 4),
    }


def print_dataset_summary(name: str, summary: Dict[str, Any]) -> None:
    short_line(name)
    print(json.dumps(summary, indent=2))


def write_split_files(base_dir: Path, split_name: str, rows: Sequence[ExampleRow]) -> None:
    split_dir = base_dir / split_name
    ensure_dir(split_dir)
    write_jsonl([{"id": r.example_id, "prompt": r.prompt} for r in rows], split_dir / f"{split_name}_prompts.jsonl")
    write_jsonl([r.to_gold() for r in rows], split_dir / f"{split_name}_gold.jsonl")
    write_jsonl([{"id": r.example_id, "prediction": "LAW family=<family>; <param>=<value>\nSELF confidence=<0..1>; alt_family=<family>; why=<brief reason>"} for r in rows], split_dir / "DO_NOT_SCORE_blank_response_template.jsonl")

    node0 = [{"id": r.example_id, "text": r.prompt.split("\n\n")[0]} for r in rows]
    node1 = [{"id": r.example_id, "text": r.prompt.split("\n\n")[1]} for r in rows]
    node2 = [{"id": r.example_id, "text": r.prompt.split("\n\n")[2]} for r in rows]
    node3 = [{"id": r.example_id, "text": r.prompt.split("\n\n")[3]} for r in rows]
    write_jsonl(node0, split_dir / f"{split_name}_node0_local.jsonl")
    write_jsonl(node1, split_dir / f"{split_name}_node1_global.jsonl")
    write_jsonl(node2, split_dir / f"{split_name}_node2_spectral.jsonl")
    write_jsonl(node3, split_dir / f"{split_name}_node3_edge.jsonl")


def report_stage_zero(run_dir: Path, rows_by_split: Dict[str, List[ExampleRow]], extract_stats: Dict[str, Any], worker_status: List[Dict[str, Any]]) -> Dict[str, Any]:
    report = {
        "stage": "stage_0",
        "split_summaries": {k: dataset_summary(v) for k, v in rows_by_split.items()},
        "worker_status": worker_status,
        "extract_stats": extract_stats,
        "chance_family_accuracy": round(1.0 / max(1, len(set(r.family for r in rows_by_split["train"]))), 4),
        "majority_family": Counter(r.family for r in rows_by_split["train"]).most_common(1)[0][0] if rows_by_split["train"] else None,
    }
    train_counts = Counter(r.family for r in rows_by_split["train"])
    report["majority_family_accuracy_estimate"] = round(max(train_counts.values()) / max(1, len(rows_by_split["train"])), 4) if train_counts else 0.0
    json_dump(report, run_dir / "reports" / "stage0_benchmark.json")
    section("STAGE 0 BENCHMARK")
    print(json.dumps(report, indent=2))
    return report


def report_midstage(run_dir: Path, train_rows: Sequence[ExampleRow], val_rows: Sequence[ExampleRow]) -> Tuple[PrototypeBaseline, Dict[str, Any]]:
    majority_family = Counter(r.family for r in train_rows).most_common(1)[0][0]
    family_defaults = parameter_defaults_by_family(train_rows)
    majority_preds = [
        {"id": r.example_id, "prediction": format_law(majority_family, family_defaults.get(majority_family, {})) + "\nSELF confidence=0.20; alt_family=none; why=majority baseline"}
        for r in val_rows
    ]
    majority_report = evaluate_prediction_rows([r.to_gold() for r in val_rows], majority_preds)
    best_k, k_table = select_best_k(train_rows, val_rows, [1, 3, 5, 7, 9, 11])
    model = PrototypeBaseline(k=best_k)
    model.fit(train_rows)
    val_preds = [{"id": r.example_id, "prediction": model.predict_text(r)} for r in val_rows]
    val_report = evaluate_prediction_rows([r.to_gold() for r in val_rows], val_preds)

    report = {
        "stage": "midstage",
        "majority_baseline": {k: majority_report[k] for k in ["family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score"]},
        "k_search": k_table,
        "selected_k": best_k,
        "validation": {k: val_report[k] for k in ["family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score", "avg_confidence", "missing_predictions", "parse_failures"]},
        "validation_confusion": val_report["confusion"],
        "sample_predictions": val_preds[:5],
    }
    json_dump(report, run_dir / "reports" / "midstage_benchmark.json")
    section("MIDSTAGE BENCHMARK")
    print(json.dumps(report, indent=2))
    return model, report


def report_completed(
    run_dir: Path,
    model_pkg: Dict[str, Any],
    test_rows_by_split: Dict[str, List[ExampleRow]],
    worker_urls: Sequence[str],
    token: str,
    chunk_size: int,
    include_local: bool,
) -> Dict[str, Any]:
    split_reports = {}
    predict_stats = {}
    for split_name, rows in test_rows_by_split.items():
        preds, stats = run_distributed_predict(model_pkg, rows, worker_urls, token, chunk_size, include_local=include_local)
        rep = evaluate_prediction_rows([r.to_gold() for r in rows], preds)
        split_reports[split_name] = {k: rep[k] for k in ["num_examples", "family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score", "avg_confidence", "missing_predictions", "parse_failures"]}
        predict_stats[split_name] = stats
        write_jsonl(preds, run_dir / f"baseline_predictions_{split_name}.jsonl")
        json_dump(rep, run_dir / "reports" / f"{split_name}_detailed.json")

    completed = {
        "stage": "completed",
        "split_reports": split_reports,
        "predict_stats": predict_stats,
    }
    json_dump(completed, run_dir / "reports" / "completed_benchmark.json")
    section("COMPLETED BENCHMARK")
    print(json.dumps(completed, indent=2))
    return completed


def write_run_summary(run_dir: Path, config: Dict[str, Any], stage0: Dict[str, Any], mid: Dict[str, Any], completed: Dict[str, Any]) -> None:
    txt = textwrap.dedent(f"""\
    # EXOSFEAR MiniLab Distributed Run

    ## Configuration
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
    {json.dumps(completed, indent=2)}
    ```

    ## Main outputs
    - `reports/stage0_benchmark.json`
    - `reports/midstage_benchmark.json`
    - `reports/completed_benchmark.json`
    - `train/`, `val/`, `test_standard/`, `test_noisy/`, `test_ood_large/`
    - `baseline_predictions_test_standard.jsonl`
    - `baseline_predictions_test_noisy.jsonl`
    - `baseline_predictions_test_ood_large.jsonl`
    """)
    (run_dir / "RUN_SUMMARY.md").write_text(txt, encoding="utf-8")


# ------------------ main pipeline ------------------


def full_pipeline_interactive() -> None:
    section("EXOSFEAR MINILAB DISTRIBUTED")
    out_dir = prompt_path("Output directory", "bench_graph_law_distributed")
    train_n = prompt_int("Train examples", 1200, 20)
    val_n = prompt_int("Validation examples", 200, 10)
    test_n = prompt_int("Test examples per test split", 200, 10)
    seed = prompt_int("Random seed", 42)
    n_min = prompt_int("Minimum nodes per graph (train/val/standard)", 28, 8)
    n_max = prompt_int("Maximum nodes per graph (train/val/standard)", 72, n_min)
    ood_min = prompt_int("OOD large minimum nodes", max(80, n_max + 8), n_max + 1)
    ood_max = prompt_int("OOD large maximum nodes", max(140, ood_min + 20), ood_min)
    spectrum_topk = prompt_int("How many top adjacency eigenvalues to keep", 6, 3, 12)
    edge_sample_max = prompt_int("Maximum sampled edges shown in EDGE_GLIMPSE", 40, 5, 200)
    families = prompt_families(FAMILY_ORDER)

    print("\nCluster setup")
    worker_urls = prompt_worker_urls("")
    include_local = prompt_bool("Include coordinator machine as a local worker", True)
    token = input("Shared worker token [blank for none]: ").strip()
    chunk_size = prompt_int("Chunk size per worker request", 64, 1)
    print()

    run_full_pipeline(
        out_dir=out_dir,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        seed=seed,
        n_min=n_min,
        n_max=n_max,
        ood_min=ood_min,
        ood_max=ood_max,
        spectrum_topk=spectrum_topk,
        edge_sample_max=edge_sample_max,
        families=families,
        worker_urls=worker_urls,
        include_local=include_local,
        token=token,
        chunk_size=chunk_size,
    )


def run_full_pipeline(
    out_dir: Path,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
    n_min: int,
    n_max: int,
    ood_min: int,
    ood_max: int,
    spectrum_topk: int,
    edge_sample_max: int,
    families: Sequence[str],
    worker_urls: Sequence[str],
    include_local: bool,
    token: str,
    chunk_size: int,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "reports")

    worker_status = ping_workers(worker_urls, token) if worker_urls else []
    if worker_urls:
        short_line("Worker status")
        print(json.dumps(worker_status, indent=2))

    config = {
        "out_dir": str(out_dir),
        "train_n": train_n,
        "val_n": val_n,
        "test_n_per_split": test_n,
        "seed": seed,
        "n_min": n_min,
        "n_max": n_max,
        "ood_min": ood_min,
        "ood_max": ood_max,
        "spectrum_topk": spectrum_topk,
        "edge_sample_max": edge_sample_max,
        "families": list(families),
        "worker_urls": list(worker_urls),
        "include_local": include_local,
        "chunk_size": chunk_size,
        "token_enabled": bool(token),
    }
    json_dump(config, out_dir / "config.json")

    # specs
    specs_train = make_specs_for_split("train", train_n, families, n_min, n_max, seed + 11)
    specs_val = make_specs_for_split("val", val_n, families, n_min, n_max, seed + 22)
    specs_standard = make_specs_for_split("test_standard", test_n, families, n_min, n_max, seed + 33)
    specs_noisy = make_specs_for_split("test_noisy", test_n, families, n_min, n_max, seed + 44, noise_range=(0.05, 0.18))
    specs_ood = make_specs_for_split("test_ood_large", test_n, families, ood_min, ood_max, seed + 55)

    # distributed extraction
    rows_train, stats_train = run_distributed_extract(specs_train, worker_urls, token, chunk_size, spectrum_topk, edge_sample_max, include_local)
    rows_val, stats_val = run_distributed_extract(specs_val, worker_urls, token, chunk_size, spectrum_topk, edge_sample_max, include_local)
    rows_standard, stats_std = run_distributed_extract(specs_standard, worker_urls, token, chunk_size, spectrum_topk, edge_sample_max, include_local)
    rows_noisy, stats_noisy = run_distributed_extract(specs_noisy, worker_urls, token, chunk_size, spectrum_topk, edge_sample_max, include_local)
    rows_ood, stats_ood = run_distributed_extract(specs_ood, worker_urls, token, chunk_size, spectrum_topk, edge_sample_max, include_local)

    rows_by_split = {
        "train": rows_train,
        "val": rows_val,
        "test_standard": rows_standard,
        "test_noisy": rows_noisy,
        "test_ood_large": rows_ood,
    }

    # write data files
    for split_name, rows in rows_by_split.items():
        write_split_files(out_dir, split_name, rows)
        write_jsonl([r.to_dict() for r in rows], out_dir / split_name / f"{split_name}_rows.jsonl")

    extract_stats = {
        "train": stats_train,
        "val": stats_val,
        "test_standard": stats_std,
        "test_noisy": stats_noisy,
        "test_ood_large": stats_ood,
    }

    stage0 = report_stage_zero(out_dir, rows_by_split, extract_stats, worker_status)
    model, mid = report_midstage(out_dir, rows_train, rows_val)
    completed = report_completed(
        run_dir=out_dir,
        model_pkg=model.package(),
        test_rows_by_split={"test_standard": rows_standard, "test_noisy": rows_noisy, "test_ood_large": rows_ood},
        worker_urls=worker_urls,
        token=token,
        chunk_size=chunk_size,
        include_local=include_local,
    )
    write_run_summary(out_dir, config, stage0, mid, completed)
    section("RUN COMPLETE")
    print(json.dumps({
        "run_dir": str(out_dir),
        "summary": str(out_dir / "RUN_SUMMARY.md"),
        "stage0_report": str(out_dir / "reports" / "stage0_benchmark.json"),
        "midstage_report": str(out_dir / "reports" / "midstage_benchmark.json"),
        "completed_report": str(out_dir / "reports" / "completed_benchmark.json"),
    }, indent=2))


# ------------------ external eval / inspect ------------------


def evaluate_external_predictions() -> None:
    section("EVALUATE EXTERNAL PREDICTIONS")
    run_dir = prompt_path("Run directory", "bench_graph_law_distributed")
    split = input("Split to score [test_standard]: ").strip() or "test_standard"
    gold_path = prompt_path("Gold JSONL path", str(run_dir / split / f"{split}_gold.jsonl"))
    pred_path = prompt_path("Prediction JSONL path", "my_predictions.jsonl")
    out_path = prompt_path("Write report JSON", str(run_dir / "reports" / f"external_eval_{split}.json"))
    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)
    rep = evaluate_prediction_rows(gold_rows, pred_rows)
    json_dump(rep, out_path)
    print(json.dumps({k: rep[k] for k in ["num_examples", "family_accuracy", "param_exact_accuracy", "mean_param_score", "overall_score", "avg_confidence", "missing_predictions", "parse_failures"]}, indent=2))
    print(f"\nDetailed report written to: {out_path}")


def inspect_run_dir() -> None:
    section("INSPECT EXISTING RUN")
    run_dir = prompt_path("Run directory", "bench_graph_law_distributed")
    for rel in ["config.json", "reports/stage0_benchmark.json", "reports/midstage_benchmark.json", "reports/completed_benchmark.json", "RUN_SUMMARY.md"]:
        path = run_dir / rel
        print(f"{'OK' if path.exists() else 'MISSING'}  {path}")


# ------------------ startup wizard / cli ------------------


def get_local_ipv4_candidates() -> List[str]:
    ips: List[str] = []

    def add(ip: str) -> None:
        ip = (ip or '').strip()
        if not ip or ip.startswith('127.') or ':' in ip:
            return
        if ip not in ips:
            ips.append(ip)

    try:
        hostname = socket.gethostname()
        for res in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            add(res[4][0])
    except Exception:
        pass

    for probe in ['8.8.8.8', '1.1.1.1', '192.168.1.1', '10.0.0.1']:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((probe, 80))
            add(s.getsockname()[0])
            s.close()
        except Exception:
            pass

    return ips


def port_is_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


def find_free_port(host: str = '0.0.0.0', preferred: int = 8765, span: int = 200) -> int:
    for port in range(preferred, preferred + span):
        if port_is_free(host, port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def normalize_worker_url(text: str, default_port: Optional[int] = None) -> str:
    text = text.strip().rstrip('/')
    if not text:
        return ''
    if not text.startswith('http://') and not text.startswith('https://'):
        if default_port and ':' not in text.split('/')[-1]:
            text = f'http://{text}:{default_port}'
        else:
            text = 'http://' + text
    return text.rstrip('/')


def scan_subnet_for_workers(base_ip: str, port: int, token: str = '', timeout: float = 0.35, max_hosts: int = 254) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    try:
        net = ipaddress.ip_network(base_ip + '/24', strict=False)
    except Exception:
        return found

    candidates = [str(ip) for ip in net.hosts()][:max_hosts]

    def check(ip: str) -> Optional[Dict[str, Any]]:
        url = f'http://{ip}:{port}'
        try:
            data = http_json(url + '/health', method='GET', token=token, timeout=max(1, int(timeout)))
            data['url'] = url
            return data
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futures = {ex.submit(check, ip): ip for ip in candidates if ip != base_ip}
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
            except Exception:
                res = None
            if res:
                found.append(res)
    found.sort(key=lambda x: x.get('url', ''))
    return found


def prompt_generated_token() -> str:
    generated = secrets.token_urlsafe(12)
    raw = input(f'Shared token [Enter={generated}; type none for no token]: ').strip()
    if not raw:
        return generated
    if raw.lower() in {'none', 'blank', 'no'}:
        return ''
    return raw


def worker_onboarding() -> None:
    section('START WORKER SERVER')
    ips = get_local_ipv4_candidates()
    if ips:
        print('Detected local IPv4 addresses:')
        for ip in ips:
            print(f'  - {ip}')
    host = input('Bind host [0.0.0.0]: ').strip() or '0.0.0.0'
    suggested = find_free_port(host, 8765)
    port = prompt_int('Port', suggested, 1, 65535)
    while not port_is_free(host, port):
        print(f'Port {port} is busy on {host}.')
        suggested = find_free_port(host, port + 1)
        port = prompt_int('Choose a free port', suggested, 1, 65535)
    token = prompt_generated_token()
    print()
    print('Share one of these URLs with the coordinator:')
    if ips:
        for ip in ips:
            print(f'  http://{ip}:{port}')
    else:
        print(f'  http://<this-machine-ip>:{port}')
    print(f'Token: {token or "<none>"}')
    print()
    start_worker(host, port, token=token)


def prompt_worker_urls_smart(default_port: int, token: str) -> List[str]:
    ips = get_local_ipv4_candidates()
    discovered: List[str] = []
    if ips:
        print('Local machine IPv4 addresses:')
        for ip in ips:
            print(f'  - {ip}')
        if prompt_bool(f'Scan local /24 for worker servers on port {default_port}', True):
            seen = set()
            for ip in ips:
                short_line(f'Scanning around {ip}/24')
                found = scan_subnet_for_workers(ip, default_port, token=token)
                for row in found:
                    url = row.get('url', '').rstrip('/')
                    if url and url not in seen:
                        seen.add(url)
                        discovered.append(url)
                if found:
                    print(json.dumps(found, indent=2))
                else:
                    print('No workers found on that subnet and port.')
    default_text = ', '.join(discovered)
    raw = input(f'Worker URLs, comma-separated [{default_text}]: ').strip()
    text = raw or default_text
    out: List[str] = []
    for part in text.split(','):
        u = normalize_worker_url(part, default_port=default_port)
        if u and u not in out:
            out.append(u)
    return out


def coordinator_onboarding() -> None:
    section('EXOSFEAR MINILAB DISTRIBUTED')
    out_dir = prompt_path('Output directory', 'bench_graph_law_distributed')
    train_n = prompt_int('Train examples', 1200, 20)
    val_n = prompt_int('Validation examples', 200, 10)
    test_n = prompt_int('Test examples per test split', 200, 10)
    seed = prompt_int('Random seed', 42)
    n_min = prompt_int('Minimum nodes per graph (train/val/standard)', 28, 8)
    n_max = prompt_int('Maximum nodes per graph (train/val/standard)', 72, n_min)
    ood_min = prompt_int('OOD large minimum nodes', max(80, n_max + 8), n_max + 1)
    ood_max = prompt_int('OOD large maximum nodes', max(140, ood_min + 20), ood_min)
    spectrum_topk = prompt_int('How many top adjacency eigenvalues to keep', 6, 3, 12)
    edge_sample_max = prompt_int('Maximum sampled edges shown in EDGE_GLIMPSE', 40, 5, 200)
    families = prompt_families(FAMILY_ORDER)

    print('\nCluster setup')
    print('The shared token is just a password you choose once and use on every worker plus the coordinator.')
    token = prompt_generated_token()
    worker_port = prompt_int('Worker port to scan/use', 8765, 1, 65535)
    worker_urls = prompt_worker_urls_smart(worker_port, token)
    include_local = prompt_bool('Include coordinator machine as a local worker', True)
    chunk_size = prompt_int('Chunk size per worker request', 64, 1)
    print()

    run_full_pipeline(
        out_dir=out_dir,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        seed=seed,
        n_min=n_min,
        n_max=n_max,
        ood_min=ood_min,
        ood_max=ood_max,
        spectrum_topk=spectrum_topk,
        edge_sample_max=edge_sample_max,
        families=families,
        worker_urls=worker_urls,
        include_local=include_local,
        token=token,
        chunk_size=chunk_size,
    )


def startup_wizard() -> None:
    section('EXOSFEAR MINILAB DISTRIBUTED')
    print('Choose mode:')
    print('  1) Full distributed pipeline (coordinator)')
    print('  2) Start worker server')
    print('  3) Evaluate external predictions')
    print('  4) Inspect existing run directory')
    choice = input('Select [1]: ').strip() or '1'

    if choice == '1':
        coordinator_onboarding()
    elif choice == '2':
        worker_onboarding()
    elif choice == '3':
        evaluate_external_predictions()
    elif choice == '4':
        inspect_run_dir()
    else:
        print('Unknown choice.')


def main() -> None:
    parser = argparse.ArgumentParser(description='EXOSFEAR MiniLab Distributed')
    parser.add_argument('--mode', choices=['menu', 'worker', 'coordinator', 'eval', 'inspect'], default='menu')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--token', default='')
    args = parser.parse_args()

    if args.mode == 'worker':
        start_worker(args.host, args.port, token=args.token)
        return
    if args.mode == 'coordinator':
        coordinator_onboarding()
        return
    if args.mode == 'eval':
        evaluate_external_predictions()
        return
    if args.mode == 'inspect':
        inspect_run_dir()
        return

    startup_wizard()


if __name__ == '__main__':
    main()
