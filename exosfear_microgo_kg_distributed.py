#!/usr/bin/env python3
"""
EXOSFEAR MicroGo KG Distributed

new

One-file distributed self-play training lab for 6x6 Micro Go.
Each side is a small graph-team of specialist node-nets:
  opening, tactics, territory, endgame

Architecture follows the proven MiniLab distributed pattern:
- Workers are HTTP servers that sit and wait for work
- Coordinator scans LAN to find workers, pushes jobs via HTTP POST
- Start workers first, then coordinator
- Coordinator also runs as a local worker by default

Dependencies: pip install torch numpy
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
from dataclasses import dataclass, field, asdict
import gzip
import hashlib
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import pickle
import random
import secrets
import socket
import sys
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("Requires PyTorch.  pip install torch numpy", file=sys.stderr)
    raise

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ── constants ──────────────────────────────────────────────────────────
BOARD_SIZE = 6
PASS_MOVE = BOARD_SIZE * BOARD_SIZE
ALL_MOVES = BOARD_SIZE * BOARD_SIZE + 1
KOMI = 3.5
MAX_GAME_LEN = 120
SEED = 42
EXPERT_NAMES = ["opening", "tactics", "territory", "endgame"]


# ── io helpers ─────────────────────────────────────────────────────────
def section(title: str) -> None:
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def short_line(title: str) -> None:
    print(f"--- {title} {'-' * max(1, 64 - len(title))}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dump(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def compress_obj(obj: Any) -> str:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def decompress_obj(s: str) -> Any:
    return pickle.loads(gzip.decompress(base64.b64decode(s.encode("ascii"))))


# ── prompts ────────────────────────────────────────────────────────────
def prompt_int(label: str, default: int, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    while True:
        try:
            raw = input(f"{label} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  integer required"); continue
        if min_v is not None and v < min_v:
            print(f"  must be >= {min_v}"); continue
        if max_v is not None and v > max_v:
            print(f"  must be <= {max_v}"); continue
        return v


def prompt_bool(label: str, default: bool = True) -> bool:
    tag = "Y/n" if default else "y/N"
    try:
        raw = input(f"{label} [{tag}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not raw:
        return default
    return raw in {"y", "yes", "1", "true"}


def prompt_str(label: str, default: str) -> str:
    try:
        raw = input(f"{label} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return raw or default


def prompt_path(label: str, default: str) -> Path:
    return Path(prompt_str(label, default))


def prompt_generated_token() -> str:
    generated = secrets.token_urlsafe(12)
    try:
        raw = input(f"Shared token [Enter={generated}; type 'none' for no token]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return generated
    if not raw:
        return generated
    if raw.lower() in {"none", "blank", "no"}:
        return ""
    return raw


# ── network helpers ────────────────────────────────────────────────────
def get_local_ips() -> List[str]:
    ips: List[str] = []
    def add(ip: str) -> None:
        ip = (ip or "").strip()
        if ip and not ip.startswith("127.") and ":" not in ip and ip not in ips:
            ips.append(ip)
    try:
        for res in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(res[4][0])
    except Exception:
        pass
    for probe in ["8.8.8.8", "1.1.1.1", "192.168.1.1", "10.0.0.1"]:
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


def find_free_port(host: str = "0.0.0.0", preferred: int = 8765, span: int = 200) -> int:
    for p in range(preferred, preferred + span):
        if port_is_free(host, p):
            return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def http_json(url: str, method: str = "GET", payload: Optional[Dict] = None,
              token: str = "", timeout: int = 300) -> Dict[str, Any]:
    import urllib.request
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url=url, data=data, method=method)
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if token:
        req.add_header("X-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def normalize_worker_url(text: str, default_port: Optional[int] = None) -> str:
    text = text.strip().rstrip("/")
    if not text:
        return ""
    if not text.startswith("http://") and not text.startswith("https://"):
        if default_port and ":" not in text.split("/")[-1]:
            text = f"http://{text}:{default_port}"
        else:
            text = "http://" + text
    return text.rstrip("/")


def scan_subnet_for_workers(base_ip: str, port: int, token: str = "",
                            timeout: float = 1.0) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    try:
        net = ipaddress.ip_network(base_ip + "/24", strict=False)
    except Exception:
        return found
    candidates = [str(ip) for ip in net.hosts()]

    def check(ip: str) -> Optional[Dict[str, Any]]:
        url = f"http://{ip}:{port}"
        try:
            data = http_json(url + "/health", method="GET", token=token,
                             timeout=max(1, int(timeout)))
            if data.get("ok") and data.get("role") == "microgo_worker":
                data["url"] = url
                return data
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex:
        futs = {ex.submit(check, ip): ip for ip in candidates}
        for fut in concurrent.futures.as_completed(futs):
            try:
                res = fut.result()
            except Exception:
                res = None
            if res:
                found.append(res)
    found.sort(key=lambda x: x.get("url", ""))
    return found


def ping_workers(urls: Sequence[str], token: str) -> List[Dict[str, Any]]:
    out = []
    for url in urls:
        try:
            data = http_json(url.rstrip("/") + "/health", token=token, timeout=5)
            out.append({"url": url, **data})
        except Exception as e:
            out.append({"url": url, "ok": False, "error": repr(e)})
    return out


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Go rules ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GoState:
    board: bytes
    to_play: int
    passes: int
    history: Tuple[str, ...]
    move_count: int = 0

    @staticmethod
    def new() -> "GoState":
        b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        st = GoState(b.tobytes(), 1, 0, tuple(), 0)
        h = st._pos_hash()
        return GoState(b.tobytes(), 1, 0, (h,), 0)

    def _pos_hash(self) -> str:
        return hashlib.sha1(self.board + bytes([1 if self.to_play == 1 else 2])).hexdigest()

    def board_array(self) -> np.ndarray:
        return np.frombuffer(self.board, dtype=np.int8).copy().reshape(BOARD_SIZE, BOARD_SIZE)

    def _nbrs(self, r, c):
        out = []
        if r > 0: out.append((r-1, c))
        if r+1 < BOARD_SIZE: out.append((r+1, c))
        if c > 0: out.append((r, c-1))
        if c+1 < BOARD_SIZE: out.append((r, c+1))
        return out

    def _group_libs(self, board, r, c):
        color = int(board[r, c])
        stack = [(r, c)]; seen = {(r, c)}; group = []; libs = set()
        while stack:
            rr, cc = stack.pop(); group.append((rr, cc))
            for nr, nc in self._nbrs(rr, cc):
                v = int(board[nr, nc])
                if v == 0: libs.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc)); stack.append((nr, nc))
        return group, libs

    def legal_moves(self) -> List[int]:
        moves = []
        board = self.board_array()
        for r, c in np.argwhere(board == 0):
            if self.try_play(int(r)*BOARD_SIZE+int(c)) is not None:
                moves.append(int(r)*BOARD_SIZE+int(c))
        moves.append(PASS_MOVE)
        return moves

    def try_play(self, move: int) -> Optional["GoState"]:
        board = self.board_array()
        if move == PASS_MOVE:
            ns = GoState(board.tobytes(), -self.to_play, self.passes+1, self.history, self.move_count+1)
            h = ns._pos_hash()
            return GoState(ns.board, ns.to_play, ns.passes, self.history+(h,), self.move_count+1)
        r, c = divmod(move, BOARD_SIZE)
        if board[r, c] != 0: return None
        board[r, c] = self.to_play
        for nr, nc in self._nbrs(r, c):
            if board[nr, nc] == -self.to_play:
                grp, libs = self._group_libs(board, nr, nc)
                if not libs:
                    for gr, gc in grp: board[gr, gc] = 0
        grp, libs = self._group_libs(board, r, c)
        if not libs: return None
        ns = GoState(board.tobytes(), -self.to_play, 0, self.history, self.move_count+1)
        h = ns._pos_hash()
        if h in self.history: return None
        return GoState(ns.board, ns.to_play, ns.passes, self.history+(h,), self.move_count+1)

    def game_over(self) -> bool:
        return self.passes >= 2 or self.move_count >= MAX_GAME_LEN

    def final_score_black(self) -> float:
        board = self.board_array()
        sb = int(np.sum(board == 1)); sw = int(np.sum(board == -1))
        visited = np.zeros_like(board, dtype=np.uint8); tb = tw = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] != 0 or visited[r, c]: continue
                stack = [(r, c)]; region = []; borders = set(); visited[r, c] = 1
                while stack:
                    rr, cc = stack.pop(); region.append((rr, cc))
                    for nr, nc in self._nbrs(rr, cc):
                        v = int(board[nr, nc])
                        if v == 0 and not visited[nr, nc]:
                            visited[nr, nc] = 1; stack.append((nr, nc))
                        elif v != 0: borders.add(v)
                if borders == {1}: tb += len(region)
                elif borders == {-1}: tw += len(region)
        return (sb + tb) - (sw + tw + KOMI)

    def winner(self) -> int:
        return 1 if self.final_score_black() > 0 else -1


def move_to_str(m: int) -> str:
    if m == PASS_MOVE: return "pass"
    r, c = divmod(m, BOARD_SIZE)
    return f"{chr(ord('A')+c)}{r+1}"


def encode_state(state: GoState) -> np.ndarray:
    board = state.board_array()
    own = (board == state.to_play).astype(np.float32)
    opp = (board == -state.to_play).astype(np.float32)
    turn = np.full_like(own, 1.0 if state.to_play == 1 else 0.0)
    legal = np.zeros_like(own)
    for mv in state.legal_moves():
        if mv != PASS_MOVE:
            r, c = divmod(mv, BOARD_SIZE); legal[r, c] = 1.0
    age = np.full_like(own, min(state.move_count / MAX_GAME_LEN, 1.0))
    return np.stack([own, opp, turn, legal, age], axis=0)


# ── Neural net ─────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class ExpertTower(nn.Module):
    def __init__(self, ch, dd, blocks=1):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.ph = nn.Sequential(nn.Conv2d(ch, 2, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.pf = nn.Linear(2*BOARD_SIZE*BOARD_SIZE, ALL_MOVES)
        self.vh = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
        self.vf1 = nn.Linear(BOARD_SIZE*BOARD_SIZE, 48); self.vf2 = nn.Linear(48, 1)
        self.df = nn.Linear(ch, dd)
    def forward(self, base):
        h = self.blocks(base)
        logits = self.pf(self.ph(h).flatten(1))
        v = torch.tanh(self.vf2(F.relu(self.vf1(self.vh(h).flatten(1))))).squeeze(-1)
        desc = torch.tanh(self.df(F.adaptive_avg_pool2d(h, 1).flatten(1)))
        return logits, v, desc


class GraphTeamNet(nn.Module):
    def __init__(self, ch=24, sb=1, eb=1, dd=32):
        super().__init__()
        self.expert_names = list(EXPERT_NAMES); ne = len(EXPERT_NAMES)
        self.stem = nn.Sequential(nn.Conv2d(5, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU())
        self.shared = nn.Sequential(*[ResBlock(ch) for _ in range(sb)])
        self.experts = nn.ModuleList([ExpertTower(ch, dd, eb) for _ in range(ne)])
        self.expert_token = nn.Parameter(torch.randn(ne, dd)*0.05)
        self.router = nn.Sequential(nn.Linear(ch+3, 64), nn.ReLU(), nn.Linear(64, ne))
        self.edge_logits = nn.Parameter(torch.zeros(ne, ne))
        self.conf_head = nn.Linear(dd, 1)
        self.router_temp = nn.Parameter(torch.tensor(1.0))

    def graph_matrix(self):
        return torch.softmax(self.edge_logits, dim=-1)

    def forward(self, x, return_aux=False):
        ne = len(self.expert_names)
        base = self.shared(self.stem(x))
        pooled = F.adaptive_avg_pool2d(base, 1).flatten(1)
        phase = x[:, 4].mean(dim=(1, 2)).unsqueeze(-1)
        mob = x[:, 3].mean(dim=(1, 2)).unsqueeze(-1)
        turn = x[:, 2].mean(dim=(1, 2)).unsqueeze(-1)
        rl = self.router(torch.cat([pooled, phase, mob, turn], -1))
        pols, vals, descs = [], [], []
        for i, exp in enumerate(self.experts):
            lo, va, de = exp(base)
            de = de + self.expert_token[i].unsqueeze(0)
            pols.append(lo); vals.append(va); descs.append(de)
        ps = torch.stack(pols, 1); vs = torch.stack(vals, 1); ds = torch.stack(descs, 1)
        edges = self.graph_matrix()
        md = torch.einsum("ij,bjd->bid", edges, ds)
        conf = self.conf_head(torch.tanh(ds + md)).squeeze(-1)
        temp = torch.clamp(self.router_temp.abs(), 0.3, 3.0)
        w = torch.softmax((rl + conf) / temp, dim=-1)
        fp = (ps * w.unsqueeze(-1)).sum(1); fv = (vs * w).sum(1)
        if not return_aux: return fp, fv
        return fp, fv, {"weights": w, "router_probs": torch.softmax(rl, -1),
                        "conf": conf, "expert_values": vs}

    def snapshot_graph(self):
        with torch.no_grad():
            return {"experts": list(self.expert_names),
                    "edges": self.graph_matrix().cpu().numpy().tolist(),
                    "temp": float(torch.clamp(self.router_temp.abs(), 0.3, 3.0).cpu())}


def new_team(device):
    net = GraphTeamNet(); net.to(device); net.eval(); return net

def net_to_b64(net):
    bio = io.BytesIO(); torch.save(net.state_dict(), bio)
    return base64.b64encode(gzip.compress(bio.getvalue())).decode("ascii")

def net_from_b64(payload, device):
    net = new_team(device)
    raw = gzip.decompress(base64.b64decode(payload.encode("ascii")))
    net.load_state_dict(torch.load(io.BytesIO(raw), map_location=device, weights_only=False))
    net.eval(); return net

def infer_aux(net, device, state):
    x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(device)
    with torch.no_grad():
        lo, va, aux = net(x, return_aux=True)
    return (lo.squeeze(0).cpu().numpy(), float(va.squeeze(0).cpu()),
            {"weights": aux["weights"].squeeze(0).cpu().numpy().tolist(),
             "conf": aux["conf"].squeeze(0).cpu().numpy().tolist()})


# ── MCTS ───────────────────────────────────────────────────────────────
@dataclass
class TreeNode:
    prior: float; to_play: int; visit_count: int = 0; value_sum: float = 0.0
    children: Dict[int, "TreeNode"] = field(default_factory=dict); expanded: bool = False
    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

class MCTS:
    def __init__(self, net, device, sims=32, c_puct=1.5):
        self.net = net; self.device = device; self.sims = sims; self.c = c_puct

    def _eval(self, state):
        x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lo, va = self.net(x)
            lo = lo.squeeze(0).cpu().numpy(); val = float(va.squeeze(0).cpu())
        legal = state.legal_moves()
        mask = np.zeros(ALL_MOVES, dtype=np.float32); mask[legal] = 1.0
        lo[mask == 0] = -1e9; pr = np.exp(lo - lo.max()); pr *= mask
        s = pr.sum(); pr = mask / max(mask.sum(), 1.0) if s <= 0 else pr / s
        return pr, val

    def _expand(self, node, state, noise=False):
        priors, value = self._eval(state); legal = state.legal_moves()
        if noise and legal:
            n = np.random.dirichlet([0.3]*len(legal))
            for i, mv in enumerate(legal): priors[mv] = 0.75*priors[mv] + 0.25*n[i]
        for mv in legal:
            cs = state.try_play(mv)
            if cs: node.children[mv] = TreeNode(float(priors[mv]), cs.to_play)
        node.expanded = True; return value

    def _select(self, node):
        tv = math.sqrt(max(1, node.visit_count)); best_s = -1e9; best_m = PASS_MOVE; best_c = None
        for mv, ch in node.children.items():
            sc = -ch.value() + self.c * ch.prior * tv / (1+ch.visit_count)
            if sc > best_s: best_s = sc; best_m = mv; best_c = ch
        return best_m, best_c

    def run(self, root_state):
        root = TreeNode(1.0, root_state.to_play)
        if root_state.game_over():
            v = np.zeros(ALL_MOVES, dtype=np.float32); v[PASS_MOVE] = 1.0; return v
        self._expand(root, root_state, noise=True)
        for _ in range(self.sims):
            node = root; state = root_state; path = [node]
            while node.expanded and node.children:
                mv, child = self._select(node)
                ns = state.try_play(mv)
                if not ns: break
                node = child; state = ns; path.append(node)
                if state.game_over(): break
            if state.game_over():
                value = 1.0 if state.winner() == state.to_play else -1.0
            else:
                value = self._expand(node, state)
            for bn in reversed(path):
                bn.visit_count += 1; bn.value_sum += value; value = -value
        visits = np.zeros(ALL_MOVES, dtype=np.float32)
        for mv, ch in root.children.items(): visits[mv] = ch.visit_count
        return visits


def sample_move(visits, state, temp):
    legal = state.legal_moves(); pr = np.zeros_like(visits); pr[legal] = visits[legal]
    if pr.sum() <= 0: pr[legal] = 1.0
    if temp <= 1e-4:
        mv = int(np.argmax(pr)); oh = np.zeros_like(pr); oh[mv] = 1.0; return mv, oh
    pp = pr ** (1.0/temp); s = pp.sum()
    if s <= 0: pp[legal] = 1.0; s = pp.sum()
    pp /= s; return int(np.random.choice(np.arange(ALL_MOVES), p=pp)), pp


# ── Self-play ──────────────────────────────────────────────────────────
@dataclass
class Sample:
    state_planes: np.ndarray; policy: np.ndarray; z: float

class ReplayBuffer:
    def __init__(self, cap=50000):
        self.cap = cap; self.data: List[Sample] = []; self.lock = threading.Lock()
    def add(self, batch):
        with self.lock:
            self.data.extend(batch)
            if len(self.data) > self.cap: self.data = self.data[-self.cap:]
    def size(self):
        with self.lock: return len(self.data)
    def sample_batch(self, bs):
        with self.lock:
            if len(self.data) < bs: return None
            idx = np.random.choice(len(self.data), bs, replace=False)
            ch = [self.data[i] for i in idx]
        return (np.stack([s.state_planes for s in ch]),
                np.stack([s.policy for s in ch]),
                np.array([s.z for s in ch], dtype=np.float32))


def self_play_game(net, device, sims, temp_moves=10):
    state = GoState.new(); samples = []; moves = []; gate_trace = []
    mcts = MCTS(net, device, sims)
    while not state.game_over():
        _, _, aux = infer_aux(net, device, state)
        gate_trace.append(aux["weights"])
        visits = mcts.run(state)
        temp = 1.0 if state.move_count < temp_moves else 1e-6
        mv, policy = sample_move(visits, state, temp)
        samples.append((encode_state(state), policy.astype(np.float32), state.to_play))
        moves.append(move_to_str(mv))
        ns = state.try_play(mv)
        if not ns: ns = state.try_play(PASS_MOVE)
        if not ns: break
        state = ns
    winner = state.winner()
    return {"winner": winner, "score_black": state.final_score_black(),
            "num_moves": len(moves), "moves": moves,
            "samples": [Sample(s, p, 1.0 if pl == winner else -1.0) for s, p, pl in samples],
            "avg_gate": np.mean(gate_trace, axis=0).tolist() if gate_trace else [0.0]*len(EXPERT_NAMES)}


def do_selfplay_job(model_b64: str, device: str, sims: int, games: int) -> Dict[str, Any]:
    """Run N self-play games. This is what workers execute."""
    net = net_from_b64(model_b64, device)
    results = []; packed = []; gate_means = []
    for gi in range(games):
        res = self_play_game(net, device, sims)
        results.append({k: v for k, v in res.items() if k != "samples"})
        gate_means.append(res.get("avg_gate", [0.0]*len(EXPERT_NAMES)))
        for s in res["samples"]:
            packed.append([s.state_planes.tolist(), s.policy.tolist(), float(s.z)])
    return {"results": results, "samples": compress_obj(packed),
            "avg_gate": np.mean(gate_means, axis=0).tolist() if gate_means else [0.0]*len(EXPERT_NAMES)}


# ── Training ───────────────────────────────────────────────────────────
def train_team(net, replay, device, steps=100, bs=64, lr=1e-3):
    if replay.size() < bs:
        return {"steps": 0, "total_loss": None}
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pl, vl, tl = [], [], []
    for _ in range(steps):
        b = replay.sample_batch(bs)
        if not b: break
        x, p_tgt, z = [torch.from_numpy(a).to(device) for a in b]
        logits, value = net(x)
        ploss = -(p_tgt * F.log_softmax(logits, -1)).sum(-1).mean()
        vloss = F.mse_loss(value, z); loss = ploss + vloss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        pl.append(float(ploss.cpu())); vl.append(float(vloss.cpu())); tl.append(float(loss.cpu()))
    net.eval()
    return {"steps": len(tl),
            "policy_loss": float(np.mean(pl)) if pl else None,
            "value_loss": float(np.mean(vl)) if vl else None,
            "total_loss": float(np.mean(tl)) if tl else None,
            "graph": net.snapshot_graph()}


# ── Evaluation ─────────────────────────────────────────────────────────
def eval_move(net, device, state, sims):
    mcts = MCTS(net, device, sims, 1.3); visits = mcts.run(state)
    legal = state.legal_moves(); masked = np.zeros_like(visits); masked[legal] = visits[legal]
    return PASS_MOVE if masked.sum() <= 0 else int(np.argmax(masked))

def play_match(nb, nw, db, dw, sims):
    state = GoState.new()
    while not state.game_over():
        mv = eval_move(nb if state.to_play == 1 else nw,
                       db if state.to_play == 1 else dw, state, sims)
        ns = state.try_play(mv)
        if not ns: ns = state.try_play(PASS_MOVE)
        if not ns: break
        state = ns
    return {"winner": state.winner(), "score_black": state.final_score_black()}

def play_vs_random(net, device, sims, games=8, as_black=True):
    wins = 0
    for _ in range(games):
        state = GoState.new()
        while not state.game_over():
            our = (state.to_play == 1 and as_black) or (state.to_play == -1 and not as_black)
            mv = eval_move(net, device, state, sims) if our else random.choice(state.legal_moves())
            ns = state.try_play(mv)
            if not ns: ns = state.try_play(PASS_MOVE)
            if not ns: break
            state = ns
        w = state.winner()
        if (w == 1 and as_black) or (w == -1 and not as_black): wins += 1
    return wins / max(1, games)

def evaluate_pair(nets, device, sims, games):
    wins_a = 0; half = max(1, games//2)
    for _ in range(half):
        r = play_match(nets["A"], nets["B"], device, device, sims)
        if r["winner"] == 1: wins_a += 1
    for _ in range(games - half):
        r = play_match(nets["B"], nets["A"], device, device, sims)
        if r["winner"] == -1: wins_a += 1
    avr = play_vs_random(nets["A"], device, sims, max(4, games//2))
    bvr = play_vs_random(nets["B"], device, sims, max(4, games//2))
    return {"games": games, "wins_A": wins_a, "wins_B": games-wins_a,
            "A_vs_random": round(avr, 3), "B_vs_random": round(bvr, 3)}


class Leaderboard:
    def __init__(self):
        self.ratings: Dict[str, float] = {}
    def ensure(self, name, r=1000.0):
        self.ratings.setdefault(name, r)
    def update(self, a, b, sa, k=24.0):
        self.ensure(a); self.ensure(b)
        ea = 1.0/(1.0+10**((self.ratings[b]-self.ratings[a])/400.0))
        self.ratings[a] += k*(sa-ea); self.ratings[b] += k*((1-sa)-(1-ea))
    def top(self):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# ── Worker HTTP server (same pattern as MiniLab) ──────────────────────
WORKER_CTX = {
    "name": socket.gethostname(),
    "token": "",
    "device": "cpu",
    "version": "exosfear-microgo-kg-1",
}


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "ExosfearMicroGoWorker/1.0"

    def _check_token(self) -> bool:
        expected = WORKER_CTX.get("token", "")
        if not expected: return True
        return self.headers.get("X-Token", "") == expected

    def do_GET(self):
        if self.path.startswith("/health"):
            if not self._check_token():
                self._resp(403, {"ok": False, "error": "bad token"}); return
            self._resp(200, {"ok": True, "role": "microgo_worker",
                             "worker_name": WORKER_CTX["name"],
                             "device": WORKER_CTX["device"],
                             "version": WORKER_CTX["version"],
                             "pid": os.getpid()})
            return
        self._resp(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if not self._check_token():
            self._resp(403, {"ok": False, "error": "bad token"}); return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode())
        except Exception:
            self._resp(400, {"ok": False, "error": "invalid json"}); return

        started = time.time()
        try:
            if self.path == "/selfplay":
                result = do_selfplay_job(
                    model_b64=payload["model"],
                    device=WORKER_CTX["device"],
                    sims=int(payload.get("sims", 20)),
                    games=int(payload.get("games", 2)),
                )
                self._resp(200, {"ok": True, **result,
                                 "worker_name": WORKER_CTX["name"],
                                 "seconds": round(time.time()-started, 2)})
                return
            self._resp(404, {"ok": False, "error": "not found"})
        except Exception as e:
            self._resp(500, {"ok": False, "error": repr(e)})

    def _resp(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a): pass


def start_worker_server(host: str, port: int, token: str, device: str) -> ThreadingHTTPServer:
    WORKER_CTX["token"] = token or ""
    WORKER_CTX["name"] = socket.gethostname()
    WORKER_CTX["device"] = device
    httpd = ThreadingHTTPServer((host, port), WorkerHandler)
    httpd.daemon_threads = True
    return httpd


# ── Coordinator: pushes jobs to workers via HTTP POST ─────────────────
def push_selfplay_to_worker(url: str, token: str, model_b64: str,
                            sims: int, games: int, timeout: int = 600) -> Dict[str, Any]:
    """POST a selfplay job to a worker and get results back."""
    return http_json(url.rstrip("/") + "/selfplay", method="POST",
                     payload={"model": model_b64, "sims": sims, "games": games},
                     token=token, timeout=timeout)


def push_selfplay_distributed(
    model_b64: str, sims: int, total_games: int,
    worker_urls: List[str], token: str,
    include_local: bool, device: str,
) -> List[Dict[str, Any]]:
    """Distribute self-play games across workers + optionally local."""
    workers = (["__local__"] if include_local else []) + list(worker_urls)
    if not workers:
        workers = ["__local__"]

    # Divide games across workers
    games_per = max(1, total_games // len(workers))
    remainder = total_games - games_per * len(workers)
    assignments: List[Tuple[str, int]] = []
    for i, w in enumerate(workers):
        g = games_per + (1 if i < remainder else 0)
        if g > 0:
            assignments.append((w, g))

    results: List[Dict[str, Any]] = []

    for worker, num_games in assignments:
        started = time.time()
        if worker == "__local__":
            res = do_selfplay_job(model_b64, device, sims, num_games)
            res["worker_name"] = "local"
            res["seconds"] = round(time.time() - started, 2)
            results.append(res)
        else:
            try:
                res = push_selfplay_to_worker(worker, token, model_b64, sims, num_games, timeout=1200)
                results.append(res)
            except Exception as exc:
                print(f"  Worker {worker} failed: {exc}")
                print(f"  Running those {num_games} games locally instead...")
                res = do_selfplay_job(model_b64, device, sims, num_games)
                res["worker_name"] = f"local_fallback_for_{worker}"
                res["seconds"] = round(time.time() - started, 2)
                results.append(res)

    return results


# ── Full pipeline ──────────────────────────────────────────────────────
def run_pipeline(cfg: Dict[str, Any]) -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    run_dir = Path(cfg["run_dir"]); ensure_dir(run_dir); ensure_dir(run_dir / "reports")
    json_dump(cfg, run_dir / "config.json")
    device = cfg["device"]
    token = cfg.get("token", "")
    worker_urls = cfg.get("worker_urls", [])
    include_local = cfg.get("include_local", True)

    # Check workers
    if worker_urls:
        short_line("Worker status")
        statuses = ping_workers(worker_urls, token)
        for s in statuses:
            tag = "✓" if s.get("ok") else "✗"
            print(f"  {tag} {s['url']}  {s.get('worker_name', '')}  {s.get('device', '')}")

    nets = {"A": new_team(device), "B": new_team(device)}
    replays = {"A": ReplayBuffer(), "B": ReplayBuffer()}
    lb = Leaderboard(); lb.ensure("A_current"); lb.ensure("B_current")

    rounds = cfg["rounds"]
    spj = cfg["selfplay_jobs_per_round"]
    gpj = cfg["games_per_job"]
    sims = cfg["selfplay_sims"]
    eval_sims = cfg["eval_sims"]
    train_steps = cfg["train_steps"]
    batch_size = cfg["batch_size"]
    lr = cfg["learning_rate"]
    eval_games = cfg["eval_games"]

    total_games_per_net = spj * gpj

    # Stage 0
    section("STAGE 0 EVALUATION")
    s0 = evaluate_pair(nets, device, eval_sims, eval_games)
    json_dump({"stage": "stage0", **s0, "timestamp": now_ts()}, run_dir / "reports" / "stage0.json")
    print(f"  A wins {s0['wins_A']}/{s0['games']}, A-vs-rand={s0['A_vs_random']}, B-vs-rand={s0['B_vs_random']}")

    mid_round = max(1, math.ceil(rounds / 2))

    for ri in range(1, rounds + 1):
        section(f"ROUND {ri}/{rounds}")

        for net_name in ["A", "B"]:
            short_line(f"Self-play: team {net_name}")
            model_b64 = net_to_b64(nets[net_name])

            results = push_selfplay_distributed(
                model_b64, sims, total_games_per_net,
                worker_urls, token, include_local, device,
            )

            total_samples = 0
            total_g = 0
            for res in results:
                packed = decompress_obj(res["samples"])
                samps = [Sample(np.array(i[0], dtype=np.float32),
                                np.array(i[1], dtype=np.float32),
                                float(i[2])) for i in packed]
                replays[net_name].add(samps)
                total_samples += len(samps)
                total_g += len(res.get("results", []))
                wn = res.get("worker_name", "?")
                ws = res.get("seconds", 0)
                print(f"    {wn}: {len(res.get('results',[]))} games, "
                      f"{len(samps)} samples, {ws:.1f}s")

            print(f"  Total: {total_g} games, {total_samples} samples, "
                  f"replay={replays[net_name].size()}")

        # Train
        short_line("Training")
        for n in ["A", "B"]:
            tr = train_team(nets[n], replays[n], device, train_steps, batch_size, lr)
            if tr["total_loss"] is not None:
                print(f"  {n}: loss={tr['total_loss']:.4f} "
                      f"(pol={tr['policy_loss']:.4f} val={tr['value_loss']:.4f})")
            else:
                print(f"  {n}: not enough samples yet")

        # Eval
        short_line("Evaluation")
        ev = evaluate_pair(nets, device, eval_sims, eval_games)
        sa = ev["wins_A"] / max(1, ev["games"])
        lb.update("A_current", "B_current", sa)
        print(f"  A wins {ev['wins_A']}/{ev['games']}, "
              f"A-vs-rand={ev['A_vs_random']}, B-vs-rand={ev['B_vs_random']}")
        print(f"  Leaderboard: {lb.top()[:4]}")

        # Save checkpoints
        for n in ["A", "B"]:
            tag = f"{n}_r{ri:03d}"
            ckpt = run_dir / "checkpoints" / f"{tag}.pt"
            ensure_dir(ckpt.parent)
            torch.save(nets[n].state_dict(), ckpt)
            lb.ensure(tag, lb.ratings[f"{n}_current"])

        # Save round report
        rr = {"round": ri, "eval": ev,
              "replay": {n: replays[n].size() for n in ["A", "B"]},
              "leaderboard": lb.top()[:6]}
        json_dump(rr, run_dir / "reports" / f"round_{ri:03d}.json")

        if ri == mid_round:
            json_dump({"stage": "midstage", **rr, "timestamp": now_ts()},
                      run_dir / "reports" / "midstage.json")
        if ri == rounds:
            json_dump({"stage": "completed", **rr, "timestamp": now_ts()},
                      run_dir / "reports" / "completed.json")

    # Summary
    json_dump({"ratings": lb.top()}, run_dir / "leaderboard.json")
    summary = [
        "# EXOSFEAR MicroGo KG Run", "",
        f"Expert nodes: {', '.join(EXPERT_NAMES)}", "",
        "## Config", "```json", json.dumps(cfg, indent=2), "```", "",
        "## Leaderboard", "```json", json.dumps(lb.top(), indent=2), "```",
    ]
    (run_dir / "RUN_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")

    section("RUN COMPLETE")
    print(f"  Summary: {run_dir / 'RUN_SUMMARY.md'}")
    print(f"  Leaderboard: {lb.top()[:6]}")


# ── Startup wizard ─────────────────────────────────────────────────────
def worker_onboarding() -> None:
    section("START MICROGO WORKER SERVER")
    ips = get_local_ips()
    if ips:
        print("Detected local IPv4 addresses:")
        for ip in ips: print(f"  - {ip}")

    host = prompt_str("Bind host", "0.0.0.0")
    suggested = find_free_port(host, 8765)
    port = prompt_int("Port", suggested, 1, 65535)
    while not port_is_free(host, port):
        print(f"Port {port} is busy.")
        suggested = find_free_port(host, port + 1)
        port = prompt_int("Choose a free port", suggested, 1, 65535)
    token = prompt_generated_token()
    device = prompt_str("Device (cpu/cuda)", choose_device())

    print()
    print("Share one of these URLs with the coordinator:")
    if ips:
        for ip in ips: print(f"  http://{ip}:{port}")
    else:
        print(f"  http://<this-machine-ip>:{port}")
    print(f"Token: {token or '<none>'}")
    print()

    httpd = start_worker_server(host, port, token, device)
    section("WORKER READY — waiting for jobs from coordinator")
    print(json.dumps({"worker_name": WORKER_CTX["name"],
                      "bind": f"http://{host}:{port}",
                      "device": device,
                      "token_enabled": bool(token)}, indent=2))
    print("Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nWorker stopped.")


def prompt_worker_urls_smart(default_port: int, token: str) -> List[str]:
    ips = get_local_ips()
    discovered: List[str] = []
    if ips:
        print("Local machine IPv4 addresses:")
        for ip in ips: print(f"  - {ip}")
        if prompt_bool(f"Scan local /24 for MicroGo workers on port {default_port}", True):
            seen = set()
            for ip in ips:
                short_line(f"Scanning around {ip}/24 ...")
                found = scan_subnet_for_workers(ip, default_port, token)
                for row in found:
                    url = row.get("url", "").rstrip("/")
                    if url and url not in seen:
                        seen.add(url)
                        discovered.append(url)
                        print(f"  ✓ Found: {url}  ({row.get('worker_name', '?')})")
                if not found:
                    print("  No workers found on that subnet.")
    default_text = ", ".join(discovered)
    try:
        raw = input(f"Worker URLs, comma-separated [{default_text}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    text = raw or default_text
    out: List[str] = []
    for part in text.split(","):
        u = normalize_worker_url(part, default_port)
        if u and u not in out: out.append(u)
    return out


def coordinator_onboarding() -> None:
    section("EXOSFEAR MICROGO KG DISTRIBUTED — COORDINATOR")
    print("Start workers on other machines FIRST, then run this.\n")

    run_dir = prompt_str("Run directory", "microgo_exosfear_kg_run")
    device = prompt_str("Device (cpu/cuda)", choose_device())
    rounds = prompt_int("Rounds", 6, 1)
    spj = prompt_int("Self-play jobs per round per net", 2, 1)
    gpj = prompt_int("Games per self-play job", 2, 1)
    selfplay_sims = prompt_int("MCTS sims (self-play)", 20, 1)
    eval_sims = prompt_int("MCTS sims (eval)", 28, 1)
    train_steps = prompt_int("Train steps per round per net", 60, 1)
    batch_size = prompt_int("Batch size", 64, 8)
    lr_str = prompt_str("Learning rate", "0.001")
    lr = float(lr_str)
    eval_games = prompt_int("Eval games per round", 8, 2)

    print("\nCluster setup")
    print("The shared token is a password used on every worker + coordinator.")
    token = prompt_generated_token()
    worker_port = prompt_int("Worker port to scan for", 8765, 1, 65535)
    worker_urls = prompt_worker_urls_smart(worker_port, token)
    include_local = prompt_bool("Also run self-play locally on this machine", True)
    print()

    cfg = {
        "run_dir": run_dir, "device": device, "rounds": rounds,
        "selfplay_jobs_per_round": spj, "games_per_job": gpj,
        "selfplay_sims": selfplay_sims, "eval_sims": eval_sims,
        "train_steps": train_steps, "batch_size": batch_size,
        "learning_rate": lr, "eval_games": eval_games,
        "token": token, "worker_urls": worker_urls,
        "include_local": include_local,
    }
    run_pipeline(cfg)


def inspect_run(run_dir: Path) -> None:
    section("INSPECT RUN")
    for rel in ["config.json", "reports/stage0.json", "reports/midstage.json",
                "reports/completed.json", "leaderboard.json", "RUN_SUMMARY.md"]:
        p = run_dir / rel
        status = "OK" if p.exists() else "MISSING"
        print(f"  {status:7s} {p}")
        if p.exists() and p.suffix == ".json":
            print(f"    {p.read_text(encoding='utf-8')[:300]}")
    summ = run_dir / "RUN_SUMMARY.md"
    if summ.exists():
        print("\n" + summ.read_text(encoding="utf-8"))


def startup_wizard() -> None:
    section("EXOSFEAR MICROGO KG DISTRIBUTED")
    print("Choose mode:")
    print("  1) Coordinator  (start workers first, then run this)")
    print("  2) Worker       (start this first, on each worker machine)")
    print("  3) Inspect      (view results of a previous run)")
    try:
        choice = input("Select [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "1":
        coordinator_onboarding()
    elif choice == "2":
        worker_onboarding()
    elif choice == "3":
        run_dir = prompt_path("Run directory", "microgo_exosfear_kg_run")
        inspect_run(run_dir)
    else:
        print("Unknown choice.")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXOSFEAR MicroGo KG Distributed")
    parser.add_argument("--mode", choices=["menu", "worker", "coordinator", "inspect"],
                        default="menu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-dir", default="microgo_exosfear_kg_run")
    args = parser.parse_args()

    if args.mode == "worker":
        device = args.device or choose_device()
        httpd = start_worker_server(args.host, args.port, args.token, device)
        section("WORKER READY")
        ips = get_local_ips()
        print(f"  Listening on http://{args.host}:{args.port}")
        if ips:
            for ip in ips:
                print(f"  Reachable at http://{ip}:{args.port}")
        print(f"  Device: {device}")
        print(f"  Token: {'enabled' if args.token else 'none'}")
        print("  Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    if args.mode == "coordinator":
        coordinator_onboarding()
        return

    if args.mode == "inspect":
        inspect_run(Path(args.run_dir))
        return

    startup_wizard()


if __name__ == "__main__":
    main()
