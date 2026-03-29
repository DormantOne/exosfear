#!/usr/bin/env python3
"""
EXOSFEAR MicroGo KG Distributed

One-file distributed self-play training lab for 6x6 Micro Go.
Each side is a small graph-team of specialist node-nets:
  opening, tactics, territory, endgame

Workers auto-discover the coordinator via UDP broadcast beacon.
Start coordinator on one machine, worker on another — they find each other.

High risk warning:
- Self-play + MCTS + training can heavily load CPU/GPU.
- Start tiny. Review settings before serious runs.
- Windows Firewall: allow Python through when prompted on first run.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import queue
import random
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request

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
DEFAULT_WORKER_PORT = 8786
DEFAULT_COORD_PORT = 8787
BEACON_PORT = 9876
BEACON_MAGIC = "EXOSFEAR_MICROGO_V1"
MAX_GAME_LEN = 120
SEED = 42
EXPERT_NAMES = ["opening", "tactics", "territory", "endgame"]


# ── missing helpers ────────────────────────────────────────────────────
class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def close_server(server: ThreadingHTTPServer) -> None:
    try:
        server.shutdown()
    except Exception:
        pass


# ── beacon: coordinator broadcasts, worker listens ────────────────────
def start_beacon(coord_url: str, token: str, interval: float = 2.0) -> threading.Thread:
    """Coordinator broadcasts its address + token on LAN via UDP every interval seconds."""
    msg = json.dumps({
        "magic": BEACON_MAGIC,
        "coord_url": coord_url,
        "token": token,
    }).encode("utf-8")

    def _loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # Also try to set SO_REUSEADDR so multiple runs don't collide
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        while True:
            try:
                sock.sendto(msg, ("255.255.255.255", BEACON_PORT))
            except Exception:
                pass
            # Also try subnet-directed broadcast for every local interface
            for ip in local_ipv4_addresses():
                if ip.startswith("127."):
                    continue
                parts = ip.split(".")
                if len(parts) == 4:
                    bcast = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                    try:
                        sock.sendto(msg, (bcast, BEACON_PORT))
                    except Exception:
                        pass
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def discover_coordinator(timeout: float = 120.0) -> Tuple[Optional[str], Optional[str]]:
    """Worker listens for coordinator beacon on UDP. Returns (coord_url, token) or (None, None)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    except Exception:
        pass
    try:
        sock.bind(("0.0.0.0", BEACON_PORT))
    except OSError as exc:
        print(f"  Warning: could not bind UDP port {BEACON_PORT}: {exc}")
        print(f"  Trying alternative port...")
        try:
            sock.bind(("0.0.0.0", BEACON_PORT + 1))
        except OSError:
            print(f"  Could not bind beacon listener. Enter coordinator URL manually.")
            sock.close()
            return None, None
    sock.settimeout(3.0)
    deadline = time.time() + timeout
    print(f"\n  Listening for coordinator beacon (UDP port {BEACON_PORT})...")
    print(f"  Make sure the coordinator is running on another machine.")
    print(f"  Windows/macOS: allow Python through firewall when prompted!\n")
    last_print = 0
    while time.time() < deadline:
        remaining = int(deadline - time.time())
        try:
            data, addr = sock.recvfrom(4096)
            msg = json.loads(data.decode("utf-8"))
            if msg.get("magic") == BEACON_MAGIC:
                coord_url = msg["coord_url"]
                token = msg["token"]
                print(f"\n  ✓ Found coordinator at {coord_url}  (beacon from {addr[0]})")
                sock.close()
                return coord_url, token
        except socket.timeout:
            now = time.time()
            if now - last_print > 3:
                print(f"  Scanning... {remaining}s remaining  ", end="\r", flush=True)
                last_print = now
        except Exception:
            pass
    print(f"\n  No coordinator beacon found within {int(timeout)}s.")
    sock.close()
    return None, None


def verify_coordinator(coord_url: str, token: str) -> bool:
    """Quick HTTP check that coordinator is reachable and token matches."""
    try:
        resp = http_get_json(coord_url.rstrip("/") + "/health", timeout=5.0)
        if resp.get("ok"):
            print(f"  ✓ Coordinator responding at {coord_url}")
            return True
    except Exception as exc:
        print(f"  ✗ Cannot reach coordinator at {coord_url}: {exc}")
    return False


# ── utilities ──────────────────────────────────────────────────────────
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compress_obj(obj: Any) -> str:
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    comp = gzip.compress(raw)
    return base64.b64encode(comp).decode("ascii")


def decompress_obj(s: str) -> Any:
    comp = base64.b64decode(s.encode("ascii"))
    raw = gzip.decompress(comp)
    return pickle.loads(raw)


def free_port(preferred: int = DEFAULT_WORKER_PORT) -> int:
    for p in range(preferred, preferred + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    raise RuntimeError("Could not find a free port")


def local_ipv4_addresses() -> List[str]:
    addrs = {"127.0.0.1"}
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)
        for info in infos:
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addrs.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(addrs)


def my_lan_ip() -> str:
    for ip in local_ipv4_addresses():
        if not ip.startswith("127."):
            return ip
    return "127.0.0.1"


def choose_device(default_cuda: bool = True) -> str:
    return "cuda" if default_cuda and torch.cuda.is_available() else "cpu"


def prompt_default(prompt: str, default: str) -> str:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw if raw else default
    except (EOFError, KeyboardInterrupt):
        return default


def prompt_int(prompt: str, default: int, min_value: Optional[int] = None) -> int:
    while True:
        try:
            raw = input(f"{prompt} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if not raw:
            return default
        try:
            v = int(raw)
            if min_value is not None and v < min_value:
                print(f"  >= {min_value} required")
                continue
            return v
        except ValueError:
            print("  integer required")


def prompt_float(prompt: str, default: float, min_value: Optional[float] = None) -> float:
    while True:
        try:
            raw = input(f"{prompt} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if not raw:
            return default
        try:
            v = float(raw)
            if min_value is not None and v < min_value:
                print(f"  >= {min_value} required")
                continue
            return v
        except ValueError:
            print("  number required")


def prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    tag = "Y/n" if default_yes else "y/N"
    try:
        raw = input(f"{prompt} [{tag}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default_yes
    if not raw:
        return default_yes
    return raw.startswith("y")


def gen_token() -> str:
    return sha256_text(str(time.time()) + str(random.random()))


# ── HTTP helpers ───────────────────────────────────────────────────────
def maybe_json_response(handler: BaseHTTPRequestHandler, code: int, obj: Dict[str, Any]) -> None:
    raw = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


def http_post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    req = urllib_request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    with urllib_request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        st = GoState(board.tobytes(), 1, 0, tuple(), 0)
        h = st.position_hash()
        return GoState(board.tobytes(), 1, 0, (h,), 0)

    def board_array(self) -> np.ndarray:
        return np.frombuffer(self.board, dtype=np.int8).copy().reshape(BOARD_SIZE, BOARD_SIZE)

    def position_hash(self) -> str:
        raw = self.board + bytes([1 if self.to_play == 1 else 2])
        return hashlib.sha1(raw).hexdigest()

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out = []
        if r > 0: out.append((r - 1, c))
        if r + 1 < BOARD_SIZE: out.append((r + 1, c))
        if c > 0: out.append((r, c - 1))
        if c + 1 < BOARD_SIZE: out.append((r, c + 1))
        return out

    def group_and_liberties(self, board: np.ndarray, r: int, c: int):
        color = int(board[r, c])
        stack = [(r, c)]
        seen = {(r, c)}
        group, libs = [], set()
        while stack:
            rr, cc = stack.pop()
            group.append((rr, cc))
            for nr, nc in self.neighbors(rr, cc):
                v = int(board[nr, nc])
                if v == 0:
                    libs.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return group, libs

    def legal_moves(self) -> List[int]:
        moves = []
        board = self.board_array()
        for r, c in np.argwhere(board == 0):
            mv = int(r) * BOARD_SIZE + int(c)
            if self.try_play(mv) is not None:
                moves.append(mv)
        moves.append(PASS_MOVE)
        return moves

    def try_play(self, move: int) -> Optional["GoState"]:
        board = self.board_array()
        if move == PASS_MOVE:
            ns = GoState(board.tobytes(), -self.to_play, self.passes + 1, self.history, self.move_count + 1)
            h = ns.position_hash()
            return GoState(ns.board, ns.to_play, ns.passes, self.history + (h,), self.move_count + 1)
        r, c = divmod(move, BOARD_SIZE)
        if board[r, c] != 0:
            return None
        board[r, c] = self.to_play
        opp = -self.to_play
        for nr, nc in self.neighbors(r, c):
            if board[nr, nc] == opp:
                grp, libs = self.group_and_liberties(board, nr, nc)
                if not libs:
                    for gr, gc in grp:
                        board[gr, gc] = 0
        grp, libs = self.group_and_liberties(board, r, c)
        if not libs:
            return None
        ns = GoState(board.tobytes(), opp, 0, self.history, self.move_count + 1)
        h = ns.position_hash()
        if h in self.history:
            return None
        return GoState(ns.board, ns.to_play, ns.passes, self.history + (h,), self.move_count + 1)

    def game_over(self) -> bool:
        return self.passes >= 2 or self.move_count >= MAX_GAME_LEN

    def final_score_black(self) -> float:
        board = self.board_array()
        sb = int(np.sum(board == 1))
        sw = int(np.sum(board == -1))
        visited = np.zeros_like(board, dtype=np.uint8)
        tb = tw = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] != 0 or visited[r, c]:
                    continue
                stack = [(r, c)]
                region, borders = [], set()
                visited[r, c] = 1
                while stack:
                    rr, cc = stack.pop()
                    region.append((rr, cc))
                    for nr, nc in self.neighbors(rr, cc):
                        v = int(board[nr, nc])
                        if v == 0 and not visited[nr, nc]:
                            visited[nr, nc] = 1
                            stack.append((nr, nc))
                        elif v != 0:
                            borders.add(v)
                if borders == {1}: tb += len(region)
                elif borders == {-1}: tw += len(region)
        return (sb + tb) - (sw + tw + KOMI)

    def winner(self) -> int:
        return 1 if self.final_score_black() > 0 else -1


def move_to_str(move: int) -> str:
    if move == PASS_MOVE:
        return "pass"
    r, c = divmod(move, BOARD_SIZE)
    return f"{chr(ord('A') + c)}{r + 1}"


def encode_state(state: GoState) -> np.ndarray:
    board = state.board_array()
    own = (board == state.to_play).astype(np.float32)
    opp = (board == -state.to_play).astype(np.float32)
    turn = np.full_like(own, 1.0 if state.to_play == 1 else 0.0)
    legal = np.zeros_like(own)
    for mv in state.legal_moves():
        if mv != PASS_MOVE:
            r, c = divmod(mv, BOARD_SIZE)
            legal[r, c] = 1.0
    age = np.full_like(own, min(state.move_count / MAX_GAME_LEN, 1.0))
    return np.stack([own, opp, turn, legal, age], axis=0)


# ── Graph-team neural net ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class ExpertTower(nn.Module):
    def __init__(self, ch: int, dd: int, blocks: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
        self.ph = nn.Sequential(nn.Conv2d(ch, 2, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.pf = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ALL_MOVES)
        self.vh = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
        self.vf1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 48)
        self.vf2 = nn.Linear(48, 1)
        self.df = nn.Linear(ch, dd)

    def forward(self, base):
        h = self.blocks(base)
        logits = self.pf(self.ph(h).flatten(1))
        v = torch.tanh(self.vf2(F.relu(self.vf1(self.vh(h).flatten(1))))).squeeze(-1)
        desc = torch.tanh(self.df(F.adaptive_avg_pool2d(h, 1).flatten(1)))
        return logits, v, desc


class GraphTeamNet(nn.Module):
    def __init__(self, ch=24, sb=1, eb=1, dd=32, expert_names=None):
        super().__init__()
        self.expert_names = expert_names or list(EXPERT_NAMES)
        self.ne = len(self.expert_names)
        self.stem = nn.Sequential(nn.Conv2d(5, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU())
        self.shared = nn.Sequential(*[ResidualBlock(ch) for _ in range(sb)])
        self.experts = nn.ModuleList([ExpertTower(ch, dd, eb) for _ in range(self.ne)])
        self.expert_token = nn.Parameter(torch.randn(self.ne, dd) * 0.05)
        self.router = nn.Sequential(nn.Linear(ch + 3, 64), nn.ReLU(), nn.Linear(64, self.ne))
        self.edge_logits = nn.Parameter(torch.zeros(self.ne, self.ne))
        self.conf_head = nn.Linear(dd, 1)
        self.router_temp = nn.Parameter(torch.tensor(1.0))

    def graph_matrix(self):
        return torch.softmax(self.edge_logits, dim=-1)

    def forward(self, x, return_aux=False):
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
        fp = (ps * w.unsqueeze(-1)).sum(1)
        fv = (vs * w).sum(1)
        if not return_aux:
            return fp, fv
        aux = {"weights": w, "router_logits": rl,
               "router_probs": torch.softmax(rl, -1),
               "conf": conf, "edges": edges.unsqueeze(0).expand(x.shape[0], -1, -1),
               "expert_values": vs}
        return fp, fv, aux

    def snapshot_graph(self):
        with torch.no_grad():
            return {"experts": list(self.expert_names),
                    "edges": self.graph_matrix().cpu().numpy().tolist(),
                    "router_temperature": float(torch.clamp(self.router_temp.abs(), 0.3, 3.0).cpu())}


def new_team(device):
    net = GraphTeamNet(); net.to(device); net.eval(); return net


def net_bytes(net):
    bio = io.BytesIO(); torch.save(net.state_dict(), bio)
    return base64.b64encode(gzip.compress(bio.getvalue())).decode("ascii")


def load_net_from_bytes(payload, device):
    net = new_team(device)
    raw = gzip.decompress(base64.b64decode(payload.encode("ascii")))
    net.load_state_dict(torch.load(io.BytesIO(raw), map_location=device, weights_only=False))
    net.eval(); return net


def infer_with_aux(net, device, state):
    x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(device)
    with torch.no_grad():
        lo, va, aux = net(x, return_aux=True)
    return (lo.squeeze(0).cpu().numpy(), float(va.squeeze(0).cpu()),
            {"weights": aux["weights"].squeeze(0).cpu().numpy().tolist(),
             "router_probs": aux["router_probs"].squeeze(0).cpu().numpy().tolist(),
             "conf": aux["conf"].squeeze(0).cpu().numpy().tolist()})


# ── MCTS ───────────────────────────────────────────────────────────────
@dataclass
class TreeNode:
    prior: float
    to_play: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "TreeNode"] = field(default_factory=dict)
    expanded: bool = False
    def value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net, device, num_simulations=32, c_puct=1.5):
        self.net = net; self.device = device
        self.num_simulations = num_simulations; self.c_puct = c_puct

    def evaluate_state(self, state):
        x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lo, va = self.net(x)
            lo = lo.squeeze(0).cpu().numpy(); val = float(va.squeeze(0).cpu())
        legal = state.legal_moves()
        mask = np.zeros(ALL_MOVES, dtype=np.float32); mask[legal] = 1.0
        lo[mask == 0] = -1e9
        pr = np.exp(lo - lo.max()); pr *= mask
        s = pr.sum()
        pr = mask / max(mask.sum(), 1.0) if s <= 0 else pr / s
        return pr, val

    def expand(self, node, state, root_noise=False):
        priors, value = self.evaluate_state(state)
        legal = state.legal_moves()
        if root_noise and legal:
            noise = np.random.dirichlet([0.3] * len(legal))
            for i, mv in enumerate(legal):
                priors[mv] = 0.75 * priors[mv] + 0.25 * noise[i]
        for mv in legal:
            cs = state.try_play(mv)
            if cs is not None:
                node.children[mv] = TreeNode(prior=float(priors[mv]), to_play=cs.to_play)
        node.expanded = True; return value

    def select_child(self, node):
        tv = math.sqrt(max(1, node.visit_count)); best_s = -1e9; best_m = PASS_MOVE; best_c = None
        for mv, ch in node.children.items():
            sc = -ch.value() + self.c_puct * ch.prior * tv / (1 + ch.visit_count)
            if sc > best_s: best_s = sc; best_m = mv; best_c = ch
        return best_m, best_c

    def run(self, root_state):
        root = TreeNode(prior=1.0, to_play=root_state.to_play)
        if root_state.game_over():
            v = np.zeros(ALL_MOVES, dtype=np.float32); v[PASS_MOVE] = 1.0; return v
        self.expand(root, root_state, root_noise=True)
        for _ in range(self.num_simulations):
            node = root; state = root_state; path = [node]
            while node.expanded and node.children:
                mv, child = self.select_child(node)
                ns = state.try_play(mv)
                if ns is None: break
                node = child; state = ns; path.append(node)
                if state.game_over(): break
            if state.game_over():
                value = 1.0 if state.winner() == state.to_play else -1.0
            else:
                value = self.expand(node, state)
            for bn in reversed(path):
                bn.visit_count += 1; bn.value_sum += value; value = -value
        visits = np.zeros(ALL_MOVES, dtype=np.float32)
        for mv, ch in root.children.items(): visits[mv] = ch.visit_count
        return visits


def sample_move_from_visits(visits, state, temperature):
    legal = state.legal_moves()
    pr = np.zeros_like(visits); pr[legal] = visits[legal]
    if pr.sum() <= 0: pr[legal] = 1.0
    if temperature <= 1e-4:
        mv = int(np.argmax(pr)); oh = np.zeros_like(pr); oh[mv] = 1.0; return mv, oh
    pp = pr ** (1.0 / temperature); s = pp.sum()
    if s <= 0: pp[legal] = 1.0; s = pp.sum()
    pp /= s; mv = int(np.random.choice(np.arange(ALL_MOVES), p=pp)); return mv, pp


# ── Replay and training ───────────────────────────────────────────────
@dataclass
class Sample:
    state_planes: np.ndarray
    policy: np.ndarray
    z: float


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity; self.samples: List[Sample] = []; self.lock = threading.Lock()

    def add_many(self, batch):
        with self.lock:
            self.samples.extend(batch)
            if len(self.samples) > self.capacity:
                self.samples = self.samples[-self.capacity:]

    def size(self):
        with self.lock: return len(self.samples)

    def sample_batch(self, bs):
        with self.lock:
            if len(self.samples) < bs: return None
            idx = np.random.choice(len(self.samples), bs, replace=False)
            ch = [self.samples[i] for i in idx]
        return (np.stack([s.state_planes for s in ch]),
                np.stack([s.policy for s in ch]),
                np.array([s.z for s in ch], dtype=np.float32))


def train_team(net, replay, device, steps=100, batch_size=64, lr=1e-3, wd=1e-4):
    if replay.size() < batch_size:
        return {"steps": 0, "policy_loss": None, "value_loss": None, "total_loss": None}
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    pl, vl, tl = [], [], []
    for _ in range(steps):
        b = replay.sample_batch(batch_size)
        if b is None: break
        x, p_tgt, z = [torch.from_numpy(a).to(device) for a in b]
        logits, value = net(x)
        ploss = -(p_tgt * F.log_softmax(logits, -1)).sum(-1).mean()
        vloss = F.mse_loss(value, z)
        loss = ploss + vloss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        pl.append(float(ploss.cpu())); vl.append(float(vloss.cpu())); tl.append(float(loss.cpu()))
    net.eval()
    return {"steps": len(tl),
            "policy_loss": float(np.mean(pl)) if pl else None,
            "value_loss": float(np.mean(vl)) if vl else None,
            "total_loss": float(np.mean(tl)) if tl else None,
            "graph": net.snapshot_graph()}


# ── Self-play and evaluation ──────────────────────────────────────────
def self_play_game(net, device, sims, temp_moves=10, c_puct=1.5):
    state = GoState.new(); samples = []; moves_sgf = []; gate_trace = []
    mcts = MCTS(net, device, sims, c_puct)
    while not state.game_over():
        _, _, aux = infer_with_aux(net, device, state)
        gate_trace.append(aux["weights"])
        visits = mcts.run(state)
        temp = 1.0 if state.move_count < temp_moves else 1e-6
        move, policy = sample_move_from_visits(visits, state, temp)
        samples.append((encode_state(state), policy.astype(np.float32), state.to_play))
        moves_sgf.append(move_to_str(move))
        ns = state.try_play(move)
        if ns is None:
            ns = state.try_play(PASS_MOVE)
            if ns is None: break
        state = ns
    winner = state.winner()
    return {"winner": winner, "score_black": state.final_score_black(),
            "num_moves": len(moves_sgf),
            "samples": [Sample(s, p, 1.0 if pl == winner else -1.0) for s, p, pl in samples],
            "moves": moves_sgf,
            "avg_gate": np.mean(gate_trace, axis=0).tolist() if gate_trace else [0.0] * len(EXPERT_NAMES)}


def choose_move_for_eval(net, device, state, sims, return_aux=False):
    aux_py = None
    if return_aux: _, _, aux_py = infer_with_aux(net, device, state)
    mcts = MCTS(net, device, sims, 1.3); visits = mcts.run(state)
    legal = state.legal_moves(); masked = np.zeros_like(visits); masked[legal] = visits[legal]
    mv = PASS_MOVE if masked.sum() <= 0 else int(np.argmax(masked))
    return (mv, aux_py) if return_aux else mv


def play_match(nb, nw, db, dw, sims, collect_diag=False):
    state = GoState.new(); moves = []; ddb = []; ddw = []
    while not state.game_over():
        if state.to_play == 1:
            mv, aux = choose_move_for_eval(nb, db, state, sims, collect_diag)
            if collect_diag and aux: ddb.append(aux["weights"])
        else:
            mv, aux = choose_move_for_eval(nw, dw, state, sims, collect_diag)
            if collect_diag and aux: ddw.append(aux["weights"])
        moves.append(move_to_str(mv))
        ns = state.try_play(mv)
        if ns is None:
            ns = state.try_play(PASS_MOVE)
            if ns is None: break
        state = ns
    return {"winner": state.winner(), "score_black": state.final_score_black(),
            "moves": moves, "num_moves": len(moves),
            "diag_black": np.mean(ddb, axis=0).tolist() if ddb else [0.0] * len(EXPERT_NAMES),
            "diag_white": np.mean(ddw, axis=0).tolist() if ddw else [0.0] * len(EXPERT_NAMES)}


def play_vs_random(net, device, sims, games=8, as_black=True):
    wins = 0; diag = []
    for _ in range(games):
        state = GoState.new()
        while not state.game_over():
            our = (state.to_play == 1 and as_black) or (state.to_play == -1 and not as_black)
            if our:
                mv, aux = choose_move_for_eval(net, device, state, sims, True)
                if aux: diag.append(aux["weights"])
            else:
                mv = random.choice(state.legal_moves())
            ns = state.try_play(mv)
            if ns is None:
                ns = state.try_play(PASS_MOVE)
                if ns is None: break
            state = ns
        w = state.winner()
        if (w == 1 and as_black) or (w == -1 and not as_black): wins += 1
    return {"win_rate": wins / max(1, games),
            "avg_gate": np.mean(diag, axis=0).tolist() if diag else [0.0] * len(EXPERT_NAMES)}


# ── Leaderboard ───────────────────────────────────────────────────────
class Leaderboard:
    def __init__(self):
        self.ratings: Dict[str, float] = {}

    def ensure(self, name, rating=1000.0):
        self.ratings.setdefault(name, rating)

    def update(self, a, b, score_a, k=24.0):
        self.ensure(a); self.ensure(b)
        ea = 1.0 / (1.0 + 10 ** ((self.ratings[b] - self.ratings[a]) / 400.0))
        self.ratings[a] += k * (score_a - ea)
        self.ratings[b] += k * ((1.0 - score_a) - (1.0 - ea))

    def top(self):
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# ── Coordinator state and server ──────────────────────────────────────
@dataclass
class Job:
    job_id: str
    kind: str
    net_name: str
    sims: int
    payload: Dict[str, Any]


class CoordinatorState:
    def __init__(self, run_dir, token, device, coord_port=None, open_pairing=True):
        self.run_dir = run_dir; self.token = token; self.device = device
        self.coord_port = coord_port; self.open_pairing = open_pairing
        self.lock = threading.Lock()
        self.jobs: queue.Queue[Job] = queue.Queue()
        self.completed: List[Dict[str, Any]] = []
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.stop = False
        self.nets = {"A": new_team(device), "B": new_team(device)}
        self.replays = {"A": ReplayBuffer(), "B": ReplayBuffer()}
        self.leaderboard = Leaderboard()
        self.leaderboard.ensure("A_current"); self.leaderboard.ensure("B_current")
        self.round_stats: List[Dict[str, Any]] = []

    def add_worker_ping(self, wid, info):
        with self.lock: info["last_seen"] = now_ts(); self.workers[wid] = info

    def next_job(self):
        try: return self.jobs.get_nowait()
        except queue.Empty: return None

    def submit_result(self, result):
        with self.lock: self.completed.append(result)


class CoordinatorHandler(BaseHTTPRequestHandler):
    state: CoordinatorState = None  # type: ignore
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/health"):
            maybe_json_response(self, 200, {"ok": True, "ts": now_ts(),
                "workers": len(self.state.workers), "role": "coordinator",
                "open_pairing": self.state.open_pairing})
            return
        if self.path.startswith("/pair/bootstrap"):
            if not self.state.open_pairing:
                maybe_json_response(self, 403, {"error": "pairing_closed"}); return
            maybe_json_response(self, 200, {"ok": True, "token": self.state.token,
                "coord_port": self.state.coord_port, "open_pairing": True, "ts": now_ts()})
            return
        maybe_json_response(self, 404, {"error": "not_found"})

    def do_POST(self):
        try:
            data = read_json_body(self)
            if self.path == "/worker/pull_job":
                if data.get("token") != self.state.token:
                    maybe_json_response(self, 403, {"error": "bad_token"}); return
                wid = data.get("worker_id", "?")
                self.state.add_worker_ping(wid, {"worker_id": wid,
                    "host": data.get("host"), "port": data.get("port"), "device": data.get("device")})
                job = self.state.next_job()
                if job is None:
                    maybe_json_response(self, 200, {"job": None}); return
                maybe_json_response(self, 200, {"job": {
                    "job_id": job.job_id, "kind": job.kind, "net_name": job.net_name,
                    "sims": job.sims, "payload": job.payload,
                    "model": net_bytes(self.state.nets[job.net_name])}})
                return
            elif self.path == "/worker/submit_result":
                if data.get("token") != self.state.token:
                    maybe_json_response(self, 403, {"error": "bad_token"}); return
                self.state.submit_result(data)
                maybe_json_response(self, 200, {"ok": True}); return
            elif self.path == "/register":
                if data.get("token") != self.state.token:
                    maybe_json_response(self, 403, {"error": "bad_token"}); return
                wid = data.get("worker_id", "?")
                self.state.add_worker_ping(wid, data)
                maybe_json_response(self, 200, {"ok": True}); return
        except Exception as exc:
            maybe_json_response(self, 500, {"error": str(exc)}); return
        maybe_json_response(self, 404, {"error": "not_found"})


def start_coordinator_server(state, host, port):
    CoordinatorHandler.state = state
    srv = ReusableThreadingHTTPServer((host, port), CoordinatorHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv


class SimpleWorkerHandler(BaseHTTPRequestHandler):
    token = ""; info: Dict[str, Any] = {}
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/health"):
            maybe_json_response(self, 200, {"ok": True, **self.info}); return
        maybe_json_response(self, 404, {"error": "not_found"})


def start_worker_health_server(host, port, token, info):
    SimpleWorkerHandler.token = token; SimpleWorkerHandler.info = info
    srv = ReusableThreadingHTTPServer((host, port), SimpleWorkerHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv


def worker_loop(coordinator_url, token, host, port, device, max_idle=999999):
    wid = f"{socket.gethostname()}-{sha256_text(host + str(port) + device)}"
    try:
        http_post_json(coordinator_url + "/register",
            {"token": token, "worker_id": wid, "host": host, "port": port,
             "device": device, "ts": now_ts()}, timeout=10.0)
        print(f"  Registered with coordinator.")
    except Exception as exc:
        print(f"  Register warning: {exc}")
    idle = 0
    jobs_done = 0
    while idle < max_idle:
        try:
            resp = http_post_json(coordinator_url + "/worker/pull_job",
                {"token": token, "worker_id": wid, "host": host, "port": port, "device": device}, timeout=30.0)
        except Exception as exc:
            print(f"  Pull failed: {exc}"); time.sleep(2); idle += 1; continue
        job = resp.get("job")
        if job is None:
            if idle % 10 == 0 and idle > 0:
                print(f"  Waiting for jobs... (idle {idle}s, completed {jobs_done} jobs so far)")
            time.sleep(1); idle += 1; continue
        idle = 0; started = time.time()
        print(f"  Got job: {job['job_id']} ({job['kind']})")
        net = load_net_from_bytes(job["model"], device)
        if job["kind"] == "selfplay":
            games = int(job["payload"].get("games", 1))
            results = []; packed = []; gate_means = []
            for gi in range(games):
                print(f"    Playing game {gi+1}/{games}...", end="\r", flush=True)
                res = self_play_game(net, device, int(job["sims"]))
                results.append({k: v for k, v in res.items() if k != "samples"})
                gate_means.append(res.get("avg_gate", [0.0] * len(EXPERT_NAMES)))
                for s in res["samples"]:
                    packed.append([s.state_planes.tolist(), s.policy.tolist(), float(s.z)])
            elapsed = time.time() - started
            out = {"token": token, "worker_id": wid, "job_id": job["job_id"],
                   "kind": "selfplay_result", "net_name": job["net_name"],
                   "results": results, "samples": compress_obj(packed),
                   "elapsed_sec": elapsed,
                   "avg_gate": np.mean(gate_means, axis=0).tolist() if gate_means else [0.0] * len(EXPERT_NAMES)}
            jobs_done += 1
            print(f"  Job {job['job_id']} done: {games} games in {elapsed:.1f}s  (total jobs: {jobs_done})")
        else:
            out = {"token": token, "worker_id": wid, "job_id": job["job_id"],
                   "kind": "unknown", "net_name": job["net_name"], "elapsed_sec": time.time() - started}
        try:
            http_post_json(coordinator_url + "/worker/submit_result", out, timeout=120.0)
        except Exception as exc:
            print(f"  Submit failed: {exc}"); time.sleep(2)


def wait_for_results(state, expected, poll=1.0):
    t0 = time.time()
    last_print = 0
    while True:
        with state.lock:
            got = len(state.completed)
            if got >= expected:
                out = list(state.completed); state.completed.clear(); return out
        now = time.time()
        if now - last_print > 5:
            print(f"  Waiting for results: {got}/{expected} (workers: {len(state.workers)}, {now-t0:.0f}s)")
            last_print = now
        time.sleep(poll)


def save_checkpoint(run_dir, name, net, ridx):
    tag = f"{name}_r{ridx:03d}"
    ckpt = run_dir / "checkpoints" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt)
    safe_json_dump(run_dir / "graphs" / f"{tag}.json", net.snapshot_graph())
    return tag


def evaluate_current_pair(state, sims, games):
    wins_a = 0; half = max(1, games // 2)
    da, db = [], []
    for _ in range(half):
        res = play_match(state.nets["A"], state.nets["B"], state.device, state.device, sims, True)
        wins_a += 1 if res["winner"] == 1 else 0
        da.append(res["diag_black"]); db.append(res["diag_white"])
    for _ in range(games - half):
        res = play_match(state.nets["B"], state.nets["A"], state.device, state.device, sims, True)
        if res["winner"] == -1: wins_a += 1
        db.append(res["diag_black"]); da.append(res["diag_white"])
    arb = play_vs_random(state.nets["A"], state.device, sims, max(4, games // 2), True)
    arw = play_vs_random(state.nets["A"], state.device, sims, max(4, games // 2), False)
    brb = play_vs_random(state.nets["B"], state.device, sims, max(4, games // 2), True)
    brw = play_vs_random(state.nets["B"], state.device, sims, max(4, games // 2), False)
    return {"games": games, "wins_A": wins_a, "wins_B": games - wins_a, "draws": 0,
            "A_vs_random": {"as_black": arb["win_rate"], "as_white": arw["win_rate"]},
            "B_vs_random": {"as_black": brb["win_rate"], "as_white": brw["win_rate"]},
            "avg_gate": {
                "A": np.mean(da, axis=0).tolist() if da else [0.0] * len(EXPERT_NAMES),
                "B": np.mean(db, axis=0).tolist() if db else [0.0] * len(EXPERT_NAMES)}}


def build_stage_report(run_dir, stage_name, ridx, state, eval_summary, extra=None):
    report = {"stage": stage_name, "timestamp": now_ts(), "round": ridx,
              "leaderboard": state.leaderboard.top(),
              "worker_count": len(state.workers), "workers": list(state.workers.values()),
              "evaluation": eval_summary,
              "graphs": {"A": state.nets["A"].snapshot_graph(), "B": state.nets["B"].snapshot_graph()}}
    if extra: report.update(extra)
    safe_json_dump(run_dir / "reports" / f"{stage_name}.json", report)
    return report


def write_run_summary(run_dir, config, stage_reports):
    lines = ["# EXOSFEAR MicroGo KG Distributed Run", "",
             f"Expert node-nets per team: {', '.join(EXPERT_NAMES)}", "",
             "## Config", "```json", json.dumps(config, indent=2), "```", ""]
    for sn in ["stage0", "midstage", "completed"]:
        r = stage_reports.get(sn)
        if r:
            lines += [f"## {sn}", "```json", json.dumps(r, indent=2), "```", ""]
    save_text(run_dir / "RUN_SUMMARY.md", "\n".join(lines))


# ── Coordinator pipeline ──────────────────────────────────────────────
def coordinator_pipeline(config):
    set_global_seed(int(config.get("seed", SEED)))
    run_dir = Path(config["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
    safe_json_dump(run_dir / "config.json", config)
    state = CoordinatorState(run_dir, config["token"], config["device"],
                             int(config["coord_port"]), config.get("open_pairing", True))
    srv = start_coordinator_server(state, config["coord_host"], int(config["coord_port"]))
    lan_ip = my_lan_ip()
    port = config["coord_port"]
    coord_url = f"http://{lan_ip}:{port}"

    # ── Start UDP beacon so workers auto-discover ──
    start_beacon(coord_url, config["token"], interval=2.0)

    print(f"\n{'='*60}")
    print(f"  COORDINATOR RUNNING")
    print(f"  HTTP : {coord_url}")
    print(f"  Token: {config['token']}")
    print(f"  Device: {config['device']}")
    print(f"  UDP beacon broadcasting on port {BEACON_PORT}")
    print(f"{'='*60}")
    print(f"\n  >>> On OTHER machines, just run:")
    print(f"  >>> python {os.path.basename(__file__)} --mode worker")
    print(f"  >>> (they will auto-discover this coordinator)")
    print(f"{'='*60}\n")

    local_ws = None
    if config.get("local_worker"):
        lwp = int(config["local_worker_port"])
        local_ws = start_worker_health_server("0.0.0.0", lwp, config["token"],
            {"host": "127.0.0.1", "port": lwp, "device": config["device"], "role": "local_worker"})
        threading.Thread(target=worker_loop, daemon=True,
            kwargs={"coordinator_url": f"http://127.0.0.1:{port}",
                    "token": config["token"], "host": "127.0.0.1",
                    "port": lwp, "device": config["device"]}).start()
        print(f"  Local worker started on port {lwp}")

    stage_reports: Dict[str, Dict] = {}
    print("\n  Running stage-0 evaluation...")
    s0 = evaluate_current_pair(state, int(config["eval_sims"]), int(config["stage_eval_games"]))
    stage_reports["stage0"] = build_stage_report(run_dir, "stage0", 0, state, s0)
    print(f"  Stage 0: A wins {s0['wins_A']}/{s0['games']}, "
          f"A vs random {s0['A_vs_random']}, B vs random {s0['B_vs_random']}")

    rounds = int(config["rounds"])
    spj = int(config["selfplay_jobs_per_round"])
    gpj = int(config["games_per_job"])
    ss = int(config["selfplay_sims"])
    ts = int(config["train_steps_per_round"])
    bs = int(config["batch_size"])
    lr = float(config["learning_rate"])
    mid = max(1, math.ceil(rounds / 2))

    for ri in range(1, rounds + 1):
        print(f"\n{'─'*50}")
        print(f"  Round {ri}/{rounds}")
        expected = 0
        for nn in ["A", "B"]:
            for ji in range(spj):
                expected += 1
                state.jobs.put(Job(f"r{ri:03d}_{nn}_{ji}", "selfplay", nn, ss, {"games": gpj}))
        print(f"  Queued {expected} self-play jobs, waiting for workers...")
        results = wait_for_results(state, expected)
        by_net = {"A": 0, "B": 0}; ts_net = {"A": 0, "B": 0}
        gbn: Dict[str, List] = {"A": [], "B": []}
        for res in results:
            nn = res["net_name"]
            packed = decompress_obj(res["samples"])
            samps = [Sample(np.array(i[0], dtype=np.float32), np.array(i[1], dtype=np.float32), float(i[2])) for i in packed]
            state.replays[nn].add_many(samps)
            ts_net[nn] += len(samps); by_net[nn] += len(res.get("results", []))
            if res.get("avg_gate"): gbn[nn].append(res["avg_gate"])
        print(f"  Games: A={by_net['A']} B={by_net['B']}  Samples: A={ts_net['A']} B={ts_net['B']}")

        tr = {}
        for n in ["A", "B"]:
            tr[n] = train_team(state.nets[n], state.replays[n], state.device, ts, bs, lr)
            if tr[n]["total_loss"] is not None:
                print(f"  Train {n}: loss={tr[n]['total_loss']:.4f} (pol={tr[n]['policy_loss']:.4f} val={tr[n]['value_loss']:.4f})")

        ev = evaluate_current_pair(state, int(config["eval_sims"]), int(config["round_eval_games"]))
        sa = ev["wins_A"] / max(1, ev["games"])
        state.leaderboard.update("A_current", "B_current", sa)
        sna = save_checkpoint(run_dir, "A", state.nets["A"], ri)
        snb = save_checkpoint(run_dir, "B", state.nets["B"], ri)
        state.leaderboard.ensure(sna, state.leaderboard.ratings["A_current"])
        state.leaderboard.ensure(snb, state.leaderboard.ratings["B_current"])
        print(f"  Eval: A wins {ev['wins_A']}/{ev['games']}  "
              f"A-vs-rand {ev['A_vs_random']}  B-vs-rand {ev['B_vs_random']}")
        print(f"  Leaderboard: {state.leaderboard.top()[:4]}")

        rr = {"round": ri, "games": by_net, "samples": ts_net,
              "replay": {n: state.replays[n].size() for n in ["A","B"]},
              "train": tr, "eval": ev,
              "selfplay_gate": {n: np.mean(gbn[n], 0).tolist() if gbn[n] else [0.0]*len(EXPERT_NAMES) for n in ["A","B"]}}
        state.round_stats.append(rr)
        safe_json_dump(run_dir / "rounds" / f"round_{ri:03d}.json", rr)

        if ri == mid:
            stage_reports["midstage"] = build_stage_report(run_dir, "midstage", ri, state, ev,
                {"replay": {n: state.replays[n].size() for n in ["A","B"]}, "train": tr})
        if ri == rounds:
            stage_reports["completed"] = build_stage_report(run_dir, "completed", ri, state, ev,
                {"replay": {n: state.replays[n].size() for n in ["A","B"]}, "train": tr})

    write_run_summary(run_dir, config, stage_reports)
    safe_json_dump(run_dir / "leaderboard.json", {"ratings": state.leaderboard.top()})
    print(f"\n{'='*60}")
    print(f"  Run complete!  Summary: {run_dir / 'RUN_SUMMARY.md'}")
    print(f"  Leaderboard: {state.leaderboard.top()}")
    print(f"{'='*60}")
    srv.shutdown()
    if local_ws: close_server(local_ws)


# ── Turnkey config builders ───────────────────────────────────────────
def default_coordinator_config() -> Dict[str, Any]:
    token = gen_token()
    coord_port = free_port(DEFAULT_COORD_PORT)
    lwp = free_port(DEFAULT_WORKER_PORT)
    return {
        "token": token,
        "coord_host": "0.0.0.0",
        "coord_port": coord_port,
        "run_dir": "microgo_exosfear_kg_run",
        "device": choose_device(),
        "seed": SEED,
        "rounds": 6,
        "selfplay_jobs_per_round": 2,
        "games_per_job": 2,
        "selfplay_sims": 20,
        "eval_sims": 28,
        "train_steps_per_round": 60,
        "batch_size": 64,
        "learning_rate": 0.001,
        "stage_eval_games": 8,
        "round_eval_games": 8,
        "local_worker": True,
        "local_worker_port": lwp,
        "worker_urls": [],
        "open_pairing": True,
    }


def coordinator_wizard() -> Dict[str, Any]:
    cfg = default_coordinator_config()
    print(f"\n  Coordinator quick-setup (press Enter to accept defaults)\n")
    print(f"  Token        : {cfg['token']}")
    print(f"  Port         : {cfg['coord_port']}")
    print(f"  Device       : {cfg['device']}")
    print(f"  Run dir      : {cfg['run_dir']}")
    print(f"  Rounds       : {cfg['rounds']}")
    print(f"  Local worker : yes (port {cfg['local_worker_port']})")
    print(f"  Beacon       : UDP port {BEACON_PORT} (workers auto-discover)")
    print()
    if not prompt_yes_no("Use these defaults?", True):
        cfg["coord_port"] = prompt_int("Coordinator port", cfg["coord_port"], 1)
        cfg["device"] = prompt_default("Device (cpu/cuda)", cfg["device"])
        cfg["run_dir"] = prompt_default("Run directory", cfg["run_dir"])
        cfg["rounds"] = prompt_int("Rounds", cfg["rounds"], 1)
        cfg["selfplay_jobs_per_round"] = prompt_int("Self-play jobs/round/net", cfg["selfplay_jobs_per_round"], 1)
        cfg["games_per_job"] = prompt_int("Games per job", cfg["games_per_job"], 1)
        cfg["selfplay_sims"] = prompt_int("MCTS sims (self-play)", cfg["selfplay_sims"], 1)
        cfg["eval_sims"] = prompt_int("MCTS sims (eval)", cfg["eval_sims"], 1)
        cfg["train_steps_per_round"] = prompt_int("Train steps/round/net", cfg["train_steps_per_round"], 1)
        cfg["local_worker"] = prompt_yes_no("Run local worker?", True)
        if cfg["local_worker"]:
            cfg["local_worker_port"] = prompt_int("Local worker port", cfg["local_worker_port"], 1)
    return cfg


# ── CLI ────────────────────────────────────────────────────────────────
def print_banner():
    print("=" * 68)
    print("  EXOSFEAR MicroGo KG Distributed")
    print("  6x6 self-play lab · two graph-team players · LAN auto-discovery")
    print(f"  Expert nodes: {', '.join(EXPERT_NAMES)}")
    print("  Coordinator broadcasts UDP beacon; workers auto-find it.")
    print("  Warning: can be compute-intensive. Start small.")
    print("=" * 68)


def parse_args():
    ap = argparse.ArgumentParser(description="EXOSFEAR MicroGo KG Distributed")
    ap.add_argument("--mode", choices=["coordinator", "worker", "inspect"],
                    default=None, help="Skip interactive menu")
    ap.add_argument("--run-dir", default="microgo_exosfear_kg_run")
    ap.add_argument("--coord-url", default=None, help="Coordinator URL (worker mode, skips beacon)")
    ap.add_argument("--token", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def run_worker(args):
    """Turnkey worker: auto-discover coordinator, auto-fetch token, start."""
    device = args.device or choose_device()
    host = args.host or my_lan_ip()
    port = args.port or free_port(DEFAULT_WORKER_PORT)

    coord_url = args.coord_url
    token = args.token

    # Step 1: find coordinator
    if coord_url and token:
        print(f"  Using provided coordinator: {coord_url}")
    elif coord_url and not token:
        print(f"  Coordinator URL given, fetching token...")
        try:
            info = http_get_json(coord_url.rstrip("/") + "/pair/bootstrap", timeout=5.0)
            token = info.get("token", "")
            if token:
                print(f"  ✓ Token fetched from coordinator.")
            else:
                print(f"  ✗ No token returned. Check coordinator.")
                return
        except Exception as exc:
            print(f"  ✗ Cannot reach {coord_url}: {exc}")
            return
    else:
        # Auto-discover via beacon
        coord_url, token = discover_coordinator(timeout=120.0)
        if not coord_url:
            print("\n  Auto-discovery failed. You can also run with:")
            print(f"    python {os.path.basename(__file__)} --mode worker --coord-url http://COORDINATOR_IP:PORT")
            manual = prompt_default("Or enter coordinator URL now (blank to quit)", "")
            if not manual.strip():
                return
            coord_url = manual.strip()
            try:
                info = http_get_json(coord_url.rstrip("/") + "/pair/bootstrap", timeout=5.0)
                token = info.get("token", "")
            except Exception as exc:
                print(f"  ✗ Cannot reach {coord_url}: {exc}")
                return
            if not token:
                print(f"  ✗ No token. Check coordinator.")
                return

    # Step 2: verify connection
    coord_url = coord_url.rstrip("/")
    if not verify_coordinator(coord_url, token):
        print("  Retrying in 3s...")
        time.sleep(3)
        if not verify_coordinator(coord_url, token):
            print("  Cannot connect. Check firewall / network.")
            print("  On Windows: allow Python through Windows Firewall.")
            return

    # Step 3: start
    srv = start_worker_health_server(host, port, token,
        {"host": host, "port": port, "device": device, "role": "worker"})
    print(f"\n{'='*60}")
    print(f"  WORKER RUNNING")
    print(f"  Coordinator : {coord_url}")
    print(f"  This worker : http://{host}:{port}")
    print(f"  Device      : {device}")
    print(f"  Waiting for jobs from coordinator...")
    print(f"{'='*60}\n")
    try:
        worker_loop(coord_url, token, host, port, device)
    except KeyboardInterrupt:
        print("\n  Worker stopped.")
    close_server(srv)


def inspector(run_dir: Path) -> None:
    print(f"Inspecting {run_dir}")
    for name in ["config.json"]:
        p = run_dir / name
        if p.exists():
            print(f"\n--- {name} ---")
            print(p.read_text(encoding="utf-8"))
    for name in ["stage0", "midstage", "completed"]:
        p = run_dir / "reports" / f"{name}.json"
        if p.exists():
            print(f"\n--- {name} ---")
            print(p.read_text(encoding="utf-8"))
    lb = run_dir / "leaderboard.json"
    if lb.exists():
        print(f"\n--- leaderboard ---")
        print(lb.read_text(encoding="utf-8"))
    summ = run_dir / "RUN_SUMMARY.md"
    if summ.exists():
        print(f"\n{'─'*60}")
        print(summ.read_text(encoding="utf-8"))


def main():
    print_banner()
    args = parse_args()

    mode = args.mode
    if mode is None:
        print("\nChoose mode:")
        print("  1) Coordinator  (start here first)")
        print("  2) Worker       (start on other machines — auto-discovers coordinator)")
        print("  3) Inspect      (view results of a previous run)")
        choice = prompt_default("Select", "1")
        mode = {"1": "coordinator", "2": "worker", "3": "inspect"}.get(choice, "coordinator")

    if mode == "coordinator":
        cfg = coordinator_wizard()
        coordinator_pipeline(cfg)

    elif mode == "worker":
        run_worker(args)

    else:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            run_dir = Path(prompt_default("Run directory to inspect", str(run_dir)))
        inspector(run_dir)


if __name__ == "__main__":
    main()
