#!/usr/bin/env python3
"""
EXOSFEAR MicroGo KG Distributed

One-file distributed self-play training lab for 6x6 Micro Go.
Each side is not a single monolithic net; it is a small graph-team:

- opening expert node-net
- tactics expert node-net
- territory expert node-net
- endgame expert node-net

A router and learnable graph edges combine their opinions.
Workers generate distributed self-play over LAN using plain HTTP.
The coordinator owns checkpoints, replay buffers, training, leaderboard,
and stage reports.

High risk warning:
- This script can become compute-intensive.
- Self-play, MCTS, and training can heavily load CPU/GPU and may make a machine sluggish.
- Review settings before trying a serious run.
- Start tiny.
- This is experimental code and should be reviewed carefully before broad use.
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
import queue
import random
import socket
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
except Exception as exc:  # pragma: no cover
    print("This script requires PyTorch. Install with: pip install torch numpy", file=sys.stderr)
    raise

# Avoid CPU oversubscription on tiny MCTS workloads.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

BOARD_SIZE = 6
PASS_MOVE = BOARD_SIZE * BOARD_SIZE
ALL_MOVES = BOARD_SIZE * BOARD_SIZE + 1
KOMI = 3.5
DEFAULT_WORKER_PORT = 8786
DEFAULT_COORD_PORT = 8787
MAX_GAME_LEN = 120
SEED = 42
EXPERT_NAMES = ["opening", "tactics", "territory", "endgame"]


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
    import pickle
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    comp = gzip.compress(raw)
    return base64.b64encode(comp).decode("ascii")


def decompress_obj(s: str) -> Any:
    import pickle
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
    host = socket.gethostname()
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
        for info in infos:
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        if ip:
            addrs.add(ip)
        s.close()
    except Exception:
        pass
    return sorted(addrs)


def choose_device(default_cuda: bool = True) -> str:
    return "cuda" if default_cuda and torch.cuda.is_available() else "cpu"


def prompt_default(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def prompt_int(prompt: str, default: int, min_value: Optional[int] = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if min_value is not None and v < min_value:
                print(f"Please enter an integer >= {min_value}")
                continue
            return v
        except ValueError:
            print("Please enter an integer")


def prompt_float(prompt: str, default: float, min_value: Optional[float] = None) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = float(raw)
            if min_value is not None and v < min_value:
                print(f"Please enter a number >= {min_value}")
                continue
            return v
        except ValueError:
            print("Please enter a number")


def prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    raw = input(f"{prompt} [{default}]: ").strip().lower()
    if not raw:
        return default_yes
    return raw.startswith("y")


def gen_token() -> str:
    return sha256_text(str(time.time()) + str(random.random()))


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
    req = urllib_request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    with urllib_request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# -----------------------------
# Go rules
# -----------------------------
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
        state = GoState(board.tobytes(), 1, 0, tuple(), 0)
        h = state.position_hash()
        return GoState(board.tobytes(), 1, 0, (h,), 0)

    def board_array(self) -> np.ndarray:
        return np.frombuffer(self.board, dtype=np.int8).copy().reshape(BOARD_SIZE, BOARD_SIZE)

    def position_hash(self) -> str:
        raw = self.board + bytes([1 if self.to_play == 1 else 2])
        return hashlib.sha1(raw).hexdigest()

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out = []
        if r > 0:
            out.append((r - 1, c))
        if r + 1 < BOARD_SIZE:
            out.append((r + 1, c))
        if c > 0:
            out.append((r, c - 1))
        if c + 1 < BOARD_SIZE:
            out.append((r, c + 1))
        return out

    def group_and_liberties(self, board: np.ndarray, r: int, c: int) -> Tuple[List[Tuple[int, int]], set]:
        color = int(board[r, c])
        stack = [(r, c)]
        seen = {(r, c)}
        group: List[Tuple[int, int]] = []
        libs = set()
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
        moves: List[int] = []
        board = self.board_array()
        empties = np.argwhere(board == 0)
        for r, c in empties:
            mv = int(r) * BOARD_SIZE + int(c)
            if self.try_play(mv) is not None:
                moves.append(mv)
        moves.append(PASS_MOVE)
        return moves

    def try_play(self, move: int) -> Optional["GoState"]:
        board = self.board_array()
        if move == PASS_MOVE:
            new_state = GoState(board.tobytes(), -self.to_play, self.passes + 1, self.history, self.move_count + 1)
            h = new_state.position_hash()
            return GoState(new_state.board, new_state.to_play, new_state.passes, self.history + (h,), self.move_count + 1)
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
        new_state = GoState(board.tobytes(), opp, 0, self.history, self.move_count + 1)
        h = new_state.position_hash()
        if h in self.history:
            return None
        return GoState(new_state.board, new_state.to_play, new_state.passes, self.history + (h,), self.move_count + 1)

    def game_over(self) -> bool:
        return self.passes >= 2 or self.move_count >= MAX_GAME_LEN

    def final_score_black(self) -> float:
        board = self.board_array()
        stones_black = int(np.sum(board == 1))
        stones_white = int(np.sum(board == -1))
        visited = np.zeros_like(board, dtype=np.uint8)
        terr_black = 0
        terr_white = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] != 0 or visited[r, c]:
                    continue
                stack = [(r, c)]
                region = []
                borders = set()
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
                if borders == {1}:
                    terr_black += len(region)
                elif borders == {-1}:
                    terr_white += len(region)
        return (stones_black + terr_black) - (stones_white + terr_white + KOMI)

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
    turn = np.full_like(own, 1.0 if state.to_play == 1 else 0.0, dtype=np.float32)
    legal = np.zeros_like(own, dtype=np.float32)
    for mv in state.legal_moves():
        if mv == PASS_MOVE:
            continue
        r, c = divmod(mv, BOARD_SIZE)
        legal[r, c] = 1.0
    move_age = np.full_like(own, min(state.move_count / MAX_GAME_LEN, 1.0), dtype=np.float32)
    return np.stack([own, opp, turn, legal, move_age], axis=0)


# -----------------------------
# Graph-team neural net
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.b2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(x + out)


class ExpertTower(nn.Module):
    def __init__(self, channels: int, desc_dim: int, blocks: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ALL_MOVES)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 48)
        self.value_fc2 = nn.Linear(48, 1)
        self.desc_fc = nn.Linear(channels, desc_dim)

    def forward(self, base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.blocks(base)
        p = self.policy_head(h).flatten(1)
        logits = self.policy_fc(p)
        v = self.value_head(h).flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        desc = torch.tanh(self.desc_fc(pooled))
        return logits, value, desc


class GraphTeamNet(nn.Module):
    def __init__(
        self,
        channels: int = 24,
        shared_blocks: int = 1,
        expert_blocks: int = 1,
        desc_dim: int = 32,
        expert_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.expert_names = expert_names or list(EXPERT_NAMES)
        self.num_experts = len(self.expert_names)
        self.stem = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(*[ResidualBlock(channels) for _ in range(shared_blocks)])
        self.experts = nn.ModuleList([ExpertTower(channels, desc_dim, blocks=expert_blocks) for _ in range(self.num_experts)])
        self.expert_token = nn.Parameter(torch.randn(self.num_experts, desc_dim) * 0.05)
        self.router = nn.Sequential(
            nn.Linear(channels + 3, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts),
        )
        self.edge_logits = nn.Parameter(torch.zeros(self.num_experts, self.num_experts))
        self.conf_head = nn.Linear(desc_dim, 1)
        self.router_temp = nn.Parameter(torch.tensor(1.0))

    def graph_matrix(self) -> torch.Tensor:
        return torch.softmax(self.edge_logits, dim=-1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        base = self.shared(self.stem(x))
        pooled = F.adaptive_avg_pool2d(base, 1).flatten(1)
        phase = x[:, 4].mean(dim=(1, 2)).unsqueeze(-1)
        mobility = x[:, 3].mean(dim=(1, 2)).unsqueeze(-1)
        turn = x[:, 2].mean(dim=(1, 2)).unsqueeze(-1)
        router_in = torch.cat([pooled, phase, mobility, turn], dim=-1)
        router_logits = self.router(router_in)

        pols: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []
        descs: List[torch.Tensor] = []
        for idx, expert in enumerate(self.experts):
            logits_i, value_i, desc_i = expert(base)
            desc_i = desc_i + self.expert_token[idx].unsqueeze(0)
            pols.append(logits_i)
            vals.append(value_i)
            descs.append(desc_i)
        pol_stack = torch.stack(pols, dim=1)
        val_stack = torch.stack(vals, dim=1)
        desc_stack = torch.stack(descs, dim=1)
        edges = self.graph_matrix()
        mixed_desc = torch.einsum("ij,bjd->bid", edges, desc_stack)
        conf = self.conf_head(torch.tanh(desc_stack + mixed_desc)).squeeze(-1)
        temp = torch.clamp(self.router_temp.abs(), min=0.3, max=3.0)
        weights = torch.softmax((router_logits + conf) / temp, dim=-1)
        final_policy = torch.sum(pol_stack * weights.unsqueeze(-1), dim=1)
        final_value = torch.sum(val_stack * weights, dim=1)
        if not return_aux:
            return final_policy, final_value
        aux = {
            "weights": weights,
            "router_logits": router_logits,
            "router_probs": torch.softmax(router_logits, dim=-1),
            "conf": conf,
            "edges": edges.unsqueeze(0).expand(x.shape[0], -1, -1),
            "expert_values": val_stack,
        }
        return final_policy, final_value, aux

    def snapshot_graph(self) -> Dict[str, Any]:
        with torch.no_grad():
            edges = self.graph_matrix().detach().cpu().numpy().tolist()
            return {
                "experts": list(self.expert_names),
                "edges": edges,
                "router_temperature": float(torch.clamp(self.router_temp.abs(), min=0.3, max=3.0).detach().cpu()),
            }


def new_team(device: str) -> GraphTeamNet:
    net = GraphTeamNet()
    net.to(device)
    net.eval()
    return net


def net_bytes(net: nn.Module) -> str:
    bio = io.BytesIO()
    torch.save(net.state_dict(), bio)
    return base64.b64encode(gzip.compress(bio.getvalue())).decode("ascii")


def load_net_from_bytes(payload: str, device: str) -> GraphTeamNet:
    net = new_team(device)
    raw = gzip.decompress(base64.b64decode(payload.encode("ascii")))
    state = torch.load(io.BytesIO(raw), map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def infer_with_aux(net: GraphTeamNet, device: str, state: GoState) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value, aux = net(x, return_aux=True)
    return (
        logits.squeeze(0).detach().cpu().numpy(),
        float(value.squeeze(0).detach().cpu()),
        {
            "weights": aux["weights"].squeeze(0).detach().cpu().numpy().tolist(),
            "router_probs": aux["router_probs"].squeeze(0).detach().cpu().numpy().tolist(),
            "conf": aux["conf"].squeeze(0).detach().cpu().numpy().tolist(),
        },
    )


# -----------------------------
# MCTS
# -----------------------------
@dataclass
class TreeNode:
    prior: float
    to_play: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "TreeNode"] = field(default_factory=dict)
    expanded: bool = False

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net: GraphTeamNet, device: str, num_simulations: int = 32, c_puct: float = 1.5):
        self.net = net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def evaluate_state(self, state: GoState) -> Tuple[np.ndarray, float]:
        x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            logits_np = logits.squeeze(0).detach().cpu().numpy()
            val = float(value.squeeze(0).detach().cpu())
        legal = state.legal_moves()
        mask = np.zeros(ALL_MOVES, dtype=np.float32)
        mask[legal] = 1.0
        logits_np[mask == 0] = -1e9
        probs = np.exp(logits_np - np.max(logits_np))
        probs *= mask
        s = float(probs.sum())
        if s <= 0:
            probs = mask / max(mask.sum(), 1.0)
        else:
            probs /= s
        return probs, val

    def expand(self, node: TreeNode, state: GoState, root_noise: bool = False) -> float:
        priors, value = self.evaluate_state(state)
        legal = state.legal_moves()
        if root_noise and legal:
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(legal))
            for i, mv in enumerate(legal):
                priors[mv] = 0.75 * priors[mv] + 0.25 * noise[i]
        for mv in legal:
            child_state = state.try_play(mv)
            if child_state is None:
                continue
            node.children[mv] = TreeNode(prior=float(priors[mv]), to_play=child_state.to_play)
        node.expanded = True
        return value

    def select_child(self, node: TreeNode) -> Tuple[int, TreeNode]:
        total_visits = math.sqrt(max(1, node.visit_count))
        best_score = -1e9
        best_mv = PASS_MOVE
        best_child = None
        for mv, child in node.children.items():
            q = -child.value()
            u = self.c_puct * child.prior * total_visits / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_mv = mv
                best_child = child
        assert best_child is not None
        return best_mv, best_child

    def run(self, root_state: GoState) -> np.ndarray:
        root = TreeNode(prior=1.0, to_play=root_state.to_play)
        if root_state.game_over():
            visits = np.zeros(ALL_MOVES, dtype=np.float32)
            visits[PASS_MOVE] = 1.0
            return visits
        self.expand(root, root_state, root_noise=True)
        for _ in range(self.num_simulations):
            node = root
            state = root_state
            path = [node]
            while node.expanded and node.children:
                mv, child = self.select_child(node)
                next_state = state.try_play(mv)
                if next_state is None:
                    break
                node = child
                state = next_state
                path.append(node)
                if state.game_over():
                    break
            if state.game_over():
                winner = state.winner()
                value = 1.0 if winner == state.to_play else -1.0
            else:
                value = self.expand(node, state, root_noise=False)
            for back_node in reversed(path):
                back_node.visit_count += 1
                back_node.value_sum += value
                value = -value
        visits = np.zeros(ALL_MOVES, dtype=np.float32)
        for mv, child in root.children.items():
            visits[mv] = child.visit_count
        return visits


def sample_move_from_visits(visits: np.ndarray, state: GoState, temperature: float) -> Tuple[int, np.ndarray]:
    legal = state.legal_moves()
    probs = np.zeros_like(visits, dtype=np.float32)
    probs[legal] = visits[legal]
    if probs.sum() <= 0:
        probs[legal] = 1.0
    if temperature <= 1e-4:
        mv = int(np.argmax(probs))
        onehot = np.zeros_like(probs)
        onehot[mv] = 1.0
        return mv, onehot
    pow_probs = probs.copy() ** (1.0 / temperature)
    s = pow_probs.sum()
    if s <= 0:
        pow_probs[legal] = 1.0
        s = pow_probs.sum()
    pow_probs /= s
    mv = int(np.random.choice(np.arange(ALL_MOVES), p=pow_probs))
    return mv, pow_probs


# -----------------------------
# Replay and training
# -----------------------------
@dataclass
class Sample:
    state_planes: np.ndarray
    policy: np.ndarray
    z: float


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.samples: List[Sample] = []
        self.lock = threading.Lock()

    def add_many(self, batch: List[Sample]) -> None:
        with self.lock:
            self.samples.extend(batch)
            if len(self.samples) > self.capacity:
                self.samples = self.samples[-self.capacity:]

    def size(self) -> int:
        with self.lock:
            return len(self.samples)

    def sample_batch(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        with self.lock:
            if len(self.samples) < batch_size:
                return None
            idx = np.random.choice(len(self.samples), size=batch_size, replace=False)
            chosen = [self.samples[i] for i in idx]
        x = np.stack([s.state_planes for s in chosen], axis=0)
        p = np.stack([s.policy for s in chosen], axis=0)
        z = np.array([s.z for s in chosen], dtype=np.float32)
        return x, p, z


def train_team(
    net: GraphTeamNet,
    replay: ReplayBuffer,
    device: str,
    steps: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    if replay.size() < batch_size:
        return {"steps": 0, "policy_loss": None, "value_loss": None, "total_loss": None}
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    pol_losses: List[float] = []
    val_losses: List[float] = []
    total_losses: List[float] = []
    for _ in range(steps):
        batch = replay.sample_batch(batch_size)
        if batch is None:
            break
        x_np, p_np, z_np = batch
        x = torch.from_numpy(x_np).to(device)
        p_tgt = torch.from_numpy(p_np).to(device)
        z = torch.from_numpy(z_np).to(device)
        logits, value = net(x)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(p_tgt * log_probs).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, z)
        loss = policy_loss + value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        pol_losses.append(float(policy_loss.detach().cpu()))
        val_losses.append(float(value_loss.detach().cpu()))
        total_losses.append(float(loss.detach().cpu()))
    net.eval()
    return {
        "steps": len(total_losses),
        "policy_loss": float(np.mean(pol_losses)) if pol_losses else None,
        "value_loss": float(np.mean(val_losses)) if val_losses else None,
        "total_loss": float(np.mean(total_losses)) if total_losses else None,
        "graph": net.snapshot_graph(),
    }


# -----------------------------
# Self-play and evaluation
# -----------------------------
def self_play_game(net: GraphTeamNet, device: str, sims: int, temperature_moves: int = 10, c_puct: float = 1.5) -> Dict[str, Any]:
    state = GoState.new()
    samples: List[Tuple[np.ndarray, np.ndarray, int]] = []
    moves_sgf: List[str] = []
    gate_trace: List[List[float]] = []
    mcts = MCTS(net, device=device, num_simulations=sims, c_puct=c_puct)
    while not state.game_over():
        _, _, aux = infer_with_aux(net, device, state)
        gate_trace.append(aux["weights"])
        visits = mcts.run(state)
        temp = 1.0 if state.move_count < temperature_moves else 1e-6
        move, policy = sample_move_from_visits(visits, state, temp)
        samples.append((encode_state(state), policy.astype(np.float32), state.to_play))
        moves_sgf.append(move_to_str(move))
        next_state = state.try_play(move)
        if next_state is None:
            next_state = state.try_play(PASS_MOVE)
            if next_state is None:
                break
        state = next_state
    winner = state.winner()
    final_samples = [Sample(state_planes=s, policy=p, z=1.0 if pl == winner else -1.0) for s, p, pl in samples]
    avg_gate = np.mean(np.array(gate_trace), axis=0).tolist() if gate_trace else [0.0] * len(EXPERT_NAMES)
    return {
        "winner": winner,
        "score_black": state.final_score_black(),
        "num_moves": len(moves_sgf),
        "samples": final_samples,
        "moves": moves_sgf,
        "avg_gate": avg_gate,
    }


def choose_move_for_eval(net: GraphTeamNet, device: str, state: GoState, sims: int, return_aux: bool = False):
    aux_py = None
    if return_aux:
        _, _, aux_py = infer_with_aux(net, device, state)
    mcts = MCTS(net, device=device, num_simulations=sims, c_puct=1.3)
    visits = mcts.run(state)
    legal = state.legal_moves()
    masked = np.zeros_like(visits)
    masked[legal] = visits[legal]
    mv = PASS_MOVE if masked.sum() <= 0 else int(np.argmax(masked))
    return (mv, aux_py) if return_aux else mv


def play_match(
    net_black: GraphTeamNet,
    net_white: GraphTeamNet,
    device_black: str,
    device_white: str,
    sims: int,
    collect_diag: bool = False,
) -> Dict[str, Any]:
    state = GoState.new()
    moves: List[str] = []
    diag_black: List[List[float]] = []
    diag_white: List[List[float]] = []
    while not state.game_over():
        if state.to_play == 1:
            mv, aux = choose_move_for_eval(net_black, device_black, state, sims, return_aux=collect_diag)
            if collect_diag and aux is not None:
                diag_black.append(aux["weights"])
        else:
            mv, aux = choose_move_for_eval(net_white, device_white, state, sims, return_aux=collect_diag)
            if collect_diag and aux is not None:
                diag_white.append(aux["weights"])
        moves.append(move_to_str(mv))
        nxt = state.try_play(mv)
        if nxt is None:
            nxt = state.try_play(PASS_MOVE)
            if nxt is None:
                break
        state = nxt
    return {
        "winner": state.winner(),
        "score_black": state.final_score_black(),
        "moves": moves,
        "num_moves": len(moves),
        "diag_black": np.mean(np.array(diag_black), axis=0).tolist() if diag_black else [0.0] * len(EXPERT_NAMES),
        "diag_white": np.mean(np.array(diag_white), axis=0).tolist() if diag_white else [0.0] * len(EXPERT_NAMES),
    }


def play_vs_random(net: GraphTeamNet, device: str, sims: int, games: int = 8, as_black: bool = True) -> Dict[str, Any]:
    wins = 0
    diag: List[List[float]] = []
    for _ in range(games):
        state = GoState.new()
        while not state.game_over():
            our_turn = (state.to_play == 1 and as_black) or (state.to_play == -1 and not as_black)
            if our_turn:
                mv, aux = choose_move_for_eval(net, device, state, sims, return_aux=True)
                if aux is not None:
                    diag.append(aux["weights"])
            else:
                mv = int(random.choice(state.legal_moves()))
            nxt = state.try_play(mv)
            if nxt is None:
                nxt = state.try_play(PASS_MOVE)
                if nxt is None:
                    break
            state = nxt
        winner = state.winner()
        if (winner == 1 and as_black) or (winner == -1 and not as_black):
            wins += 1
    return {
        "win_rate": wins / max(1, games),
        "avg_gate": np.mean(np.array(diag), axis=0).tolist() if diag else [0.0] * len(EXPERT_NAMES),
    }


# -----------------------------
# Leaderboard
# -----------------------------
class Leaderboard:
    def __init__(self):
        self.ratings: Dict[str, float] = {}

    def ensure(self, name: str, rating: float = 1000.0) -> None:
        self.ratings.setdefault(name, rating)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, a: str, b: str, score_a: float, k: float = 24.0) -> None:
        self.ensure(a)
        self.ensure(b)
        ea = self.expected(self.ratings[a], self.ratings[b])
        self.ratings[a] += k * (score_a - ea)
        self.ratings[b] += k * ((1.0 - score_a) - (1.0 - ea))

    def top(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# -----------------------------
# Coordinator state and server
# -----------------------------
@dataclass
class Job:
    job_id: str
    kind: str
    net_name: str
    sims: int
    payload: Dict[str, Any]


class CoordinatorState:
    def __init__(self, run_dir: Path, token: str, device: str):
        self.run_dir = run_dir
        self.token = token
        self.device = device
        self.lock = threading.Lock()
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.completed: List[Dict[str, Any]] = []
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.stop = False
        self.nets: Dict[str, GraphTeamNet] = {"A": new_team(device), "B": new_team(device)}
        self.replays: Dict[str, ReplayBuffer] = {"A": ReplayBuffer(), "B": ReplayBuffer()}
        self.leaderboard = Leaderboard()
        self.leaderboard.ensure("A_current")
        self.leaderboard.ensure("B_current")
        self.round_stats: List[Dict[str, Any]] = []

    def add_worker_ping(self, worker_id: str, info: Dict[str, Any]) -> None:
        with self.lock:
            info["last_seen"] = now_ts()
            self.workers[worker_id] = info

    def next_job(self) -> Optional[Job]:
        try:
            return self.jobs.get_nowait()
        except queue.Empty:
            return None

    def submit_result(self, result: Dict[str, Any]) -> None:
        with self.lock:
            self.completed.append(result)


class CoordinatorHandler(BaseHTTPRequestHandler):
    state: CoordinatorState = None  # type: ignore

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            maybe_json_response(self, 200, {"ok": True, "ts": now_ts(), "workers": len(self.state.workers)})
            return
        maybe_json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        try:
            data = read_json_body(self)
            if self.path == "/worker/pull_job":
                if data.get("token") != self.state.token:
                    maybe_json_response(self, 403, {"error": "bad_token"})
                    return
                worker_id = data.get("worker_id", "unknown")
                self.state.add_worker_ping(worker_id, {
                    "worker_id": worker_id,
                    "host": data.get("host"),
                    "port": data.get("port"),
                    "device": data.get("device"),
                })
                job = self.state.next_job()
                if job is None:
                    maybe_json_response(self, 200, {"job": None})
                    return
                maybe_json_response(self, 200, {
                    "job": {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "net_name": job.net_name,
                        "sims": job.sims,
                        "payload": job.payload,
                        "model": net_bytes(self.state.nets[job.net_name]),
                    }
                })
                return
            elif self.path == "/worker/submit_result":
                if data.get("token") != self.state.token:
                    maybe_json_response(self, 403, {"error": "bad_token"})
                    return
                self.state.submit_result(data)
                maybe_json_response(self, 200, {"ok": True})
                return
            elif self.path == "/register":
                if data.get("token") != self.state.token:
                    maybe_json_response(self, 403, {"error": "bad_token"})
                    return
                worker_id = data.get("worker_id", "unknown")
                self.state.add_worker_ping(worker_id, data)
                maybe_json_response(self, 200, {"ok": True})
                return
        except Exception as exc:
            maybe_json_response(self, 500, {"error": str(exc)})
            return
        maybe_json_response(self, 404, {"error": "not_found"})


def start_coordinator_server(state: CoordinatorState, host: str, port: int) -> ThreadingHTTPServer:
    CoordinatorHandler.state = state
    server = ThreadingHTTPServer((host, port), CoordinatorHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class SimpleWorkerHandler(BaseHTTPRequestHandler):
    token: str = ""
    info: Dict[str, Any] = {}

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            maybe_json_response(self, 200, {"ok": True, **self.info})
            return
        maybe_json_response(self, 404, {"error": "not_found"})


def start_worker_health_server(host: str, port: int, token: str, info: Dict[str, Any]) -> ThreadingHTTPServer:
    SimpleWorkerHandler.token = token
    SimpleWorkerHandler.info = info
    server = ThreadingHTTPServer((host, port), SimpleWorkerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def scan_for_workers(port: int, timeout: float = 0.35) -> List[str]:
    ips = local_ipv4_addresses()
    candidates: List[str] = []
    seen_subnets = set()
    for ip in ips:
        if ip.startswith("127."):
            continue
        parts = ip.split(".")
        if len(parts) != 4:
            continue
        prefix = ".".join(parts[:3])
        if prefix in seen_subnets:
            continue
        seen_subnets.add(prefix)
        for i in range(1, 255):
            candidates.append(f"http://{prefix}.{i}:{port}")
    found: List[str] = []
    q_urls: "queue.Queue[str]" = queue.Queue()
    for url in candidates:
        q_urls.put(url)
    lock = threading.Lock()

    def worker_scan() -> None:
        while True:
            try:
                url = q_urls.get_nowait()
            except queue.Empty:
                return
            try:
                resp = http_get_json(url + "/health", timeout=timeout)
                if resp.get("ok"):
                    with lock:
                        found.append(url)
            except Exception:
                pass
            finally:
                q_urls.task_done()

    threads = [threading.Thread(target=worker_scan, daemon=True) for _ in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(found)


def worker_loop(coordinator_url: str, token: str, host: str, port: int, device: str, max_idle: int = 999999) -> None:
    worker_id = f"{socket.gethostname()}-{sha256_text(host + str(port) + device)}"
    try:
        http_post_json(coordinator_url + "/register", {
            "token": token,
            "worker_id": worker_id,
            "host": host,
            "port": port,
            "device": device,
            "ts": now_ts(),
        }, timeout=10.0)
    except Exception as exc:
        print(f"Initial register warning: {exc}")
    idle = 0
    while idle < max_idle:
        try:
            resp = http_post_json(coordinator_url + "/worker/pull_job", {
                "token": token,
                "worker_id": worker_id,
                "host": host,
                "port": port,
                "device": device,
            }, timeout=30.0)
        except Exception as exc:
            print(f"Worker pull failed: {exc}")
            time.sleep(2.0)
            idle += 1
            continue
        job = resp.get("job")
        if job is None:
            time.sleep(1.0)
            idle += 1
            continue
        idle = 0
        started = time.time()
        net = load_net_from_bytes(job["model"], device=device)
        out: Dict[str, Any]
        if job["kind"] == "selfplay":
            games = int(job["payload"].get("games", 1))
            results = []
            packed_samples: List[List[Any]] = []
            gate_means: List[List[float]] = []
            for _ in range(games):
                res = self_play_game(net, device=device, sims=int(job["sims"]))
                results.append({k: v for k, v in res.items() if k != "samples"})
                gate_means.append(res.get("avg_gate", [0.0] * len(EXPERT_NAMES)))
                for s in res["samples"]:
                    packed_samples.append([s.state_planes.tolist(), s.policy.tolist(), float(s.z)])
            out = {
                "token": token,
                "worker_id": worker_id,
                "job_id": job["job_id"],
                "kind": "selfplay_result",
                "net_name": job["net_name"],
                "results": results,
                "samples": compress_obj(packed_samples),
                "elapsed_sec": time.time() - started,
                "avg_gate": np.mean(np.array(gate_means), axis=0).tolist() if gate_means else [0.0] * len(EXPERT_NAMES),
            }
        else:
            out = {
                "token": token,
                "worker_id": worker_id,
                "job_id": job["job_id"],
                "kind": "unknown",
                "net_name": job["net_name"],
                "elapsed_sec": time.time() - started,
            }
        try:
            http_post_json(coordinator_url + "/worker/submit_result", out, timeout=120.0)
        except Exception as exc:
            print(f"Submit failed: {exc}")
            time.sleep(2.0)


def wait_for_results(state: CoordinatorState, expected_jobs: int, poll_sec: float = 1.0) -> List[Dict[str, Any]]:
    while True:
        with state.lock:
            if len(state.completed) >= expected_jobs:
                out = list(state.completed)
                state.completed.clear()
                return out
        time.sleep(poll_sec)


def save_checkpoint(run_dir: Path, name: str, net: GraphTeamNet, round_idx: int) -> str:
    tag = f"{name}_r{round_idx:03d}"
    ckpt = run_dir / "checkpoints" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ckpt)
    safe_json_dump(run_dir / "graphs" / f"{tag}.json", net.snapshot_graph())
    return tag


def build_stage_report(run_dir: Path, stage_name: str, round_idx: int, state: CoordinatorState, eval_summary: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    report = {
        "stage": stage_name,
        "timestamp": now_ts(),
        "round": round_idx,
        "leaderboard": state.leaderboard.top(),
        "worker_count": len(state.workers),
        "workers": list(state.workers.values()),
        "evaluation": eval_summary,
        "graphs": {
            "A": state.nets["A"].snapshot_graph(),
            "B": state.nets["B"].snapshot_graph(),
        },
    }
    if extra:
        report.update(extra)
    safe_json_dump(run_dir / "reports" / f"{stage_name}.json", report)
    return report


def evaluate_current_pair(state: CoordinatorState, sims: int, games: int) -> Dict[str, Any]:
    wins_a = 0
    draws = 0
    diag_a: List[List[float]] = []
    diag_b: List[List[float]] = []
    half = max(1, games // 2)
    for _ in range(half):
        res = play_match(state.nets["A"], state.nets["B"], state.device, state.device, sims=sims, collect_diag=True)
        wins_a += 1 if res["winner"] == 1 else 0
        diag_a.append(res["diag_black"])
        diag_b.append(res["diag_white"])
    for _ in range(games - half):
        res = play_match(state.nets["B"], state.nets["A"], state.device, state.device, sims=sims, collect_diag=True)
        if res["winner"] == -1:
            wins_a += 1
        diag_b.append(res["diag_black"])
        diag_a.append(res["diag_white"])
    a_rand_b = play_vs_random(state.nets["A"], state.device, sims=sims, games=max(4, games // 2), as_black=True)
    a_rand_w = play_vs_random(state.nets["A"], state.device, sims=sims, games=max(4, games // 2), as_black=False)
    b_rand_b = play_vs_random(state.nets["B"], state.device, sims=sims, games=max(4, games // 2), as_black=True)
    b_rand_w = play_vs_random(state.nets["B"], state.device, sims=sims, games=max(4, games // 2), as_black=False)
    return {
        "games": games,
        "wins_A": wins_a,
        "wins_B": games - wins_a - draws,
        "draws": draws,
        "A_vs_random": {"as_black": a_rand_b["win_rate"], "as_white": a_rand_w["win_rate"]},
        "B_vs_random": {"as_black": b_rand_b["win_rate"], "as_white": b_rand_w["win_rate"]},
        "avg_gate": {
            "A": np.mean(np.array(diag_a), axis=0).tolist() if diag_a else [0.0] * len(EXPERT_NAMES),
            "B": np.mean(np.array(diag_b), axis=0).tolist() if diag_b else [0.0] * len(EXPERT_NAMES),
        },
    }


def write_run_summary(run_dir: Path, config: Dict[str, Any], stage_reports: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# EXOSFEAR MicroGo KG Distributed Run",
        "",
        "This run uses two graph-team players (A and B).",
        f"Each team has expert node-nets: {', '.join(EXPERT_NAMES)}.",
        "Self-play is distributed over LAN workers. Training remains centralized on the coordinator.",
        "",
        "## Config",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
    ]
    for stage_name in ["stage0", "midstage", "completed"]:
        rep = stage_reports.get(stage_name)
        if not rep:
            continue
        lines.extend([
            f"## {stage_name}",
            "```json",
            json.dumps(rep, indent=2),
            "```",
            "",
        ])
    save_text(run_dir / "RUN_SUMMARY.md", "\n".join(lines))


# -----------------------------
# Coordinator pipeline
# -----------------------------
def coordinator_pipeline(config: Dict[str, Any]) -> None:
    set_global_seed(int(config.get("seed", SEED)))
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_json_dump(run_dir / "config.json", config)
    state = CoordinatorState(run_dir=run_dir, token=config["token"], device=config["device"])
    server = start_coordinator_server(state, config["coord_host"], int(config["coord_port"]))
    print(f"Coordinator listening on http://{config['coord_host']}:{config['coord_port']}")

    local_worker_server = None
    if config.get("local_worker"):
        host = config["coord_host"] if config["coord_host"] != "0.0.0.0" else "0.0.0.0"
        local_worker_server = start_worker_health_server(host, int(config["local_worker_port"]), config["token"], {
            "host": host,
            "port": int(config["local_worker_port"]),
            "device": config["device"],
            "role": "local_worker",
        })
        threading.Thread(
            target=worker_loop,
            kwargs={
                "coordinator_url": f"http://127.0.0.1:{config['coord_port']}",
                "token": config["token"],
                "host": "127.0.0.1",
                "port": int(config["local_worker_port"]),
                "device": config["device"],
            },
            daemon=True,
        ).start()

    if config.get("worker_urls"):
        print("Configured worker URLs:")
        for url in config["worker_urls"]:
            try:
                print(f"  - {url}: {http_get_json(url + '/health', timeout=2.0)}")
            except Exception as exc:
                print(f"  - {url}: not responding ({exc})")

    stage_reports: Dict[str, Dict[str, Any]] = {}
    stage0 = evaluate_current_pair(state, sims=int(config["eval_sims"]), games=int(config["stage_eval_games"]))
    stage_reports["stage0"] = build_stage_report(run_dir, "stage0", 0, state, stage0)
    print(json.dumps(stage_reports["stage0"], indent=2))

    rounds = int(config["rounds"])
    selfplay_jobs_per_round = int(config["selfplay_jobs_per_round"])
    games_per_job = int(config["games_per_job"])
    selfplay_sims = int(config["selfplay_sims"])
    train_steps = int(config["train_steps_per_round"])
    batch_size = int(config["batch_size"])
    lr = float(config["learning_rate"])
    mid_round = max(1, math.ceil(rounds / 2))

    for round_idx in range(1, rounds + 1):
        print(f"\n===== Round {round_idx}/{rounds} =====")
        worker_times: Dict[str, float] = {}
        expected_jobs = 0
        for net_name in ["A", "B"]:
            for job_ix in range(selfplay_jobs_per_round):
                expected_jobs += 1
                state.jobs.put(Job(
                    job_id=f"r{round_idx:03d}_{net_name}_{job_ix}",
                    kind="selfplay",
                    net_name=net_name,
                    sims=selfplay_sims,
                    payload={"games": games_per_job},
                ))
        results = wait_for_results(state, expected_jobs)
        by_net = {"A": 0, "B": 0}
        total_samples = {"A": 0, "B": 0}
        gate_by_net: Dict[str, List[List[float]]] = {"A": [], "B": []}
        for res in results:
            net_name = res["net_name"]
            worker_id = res.get("worker_id", "unknown")
            worker_times[worker_id] = worker_times.get(worker_id, 0.0) + float(res.get("elapsed_sec", 0.0))
            packed_samples = decompress_obj(res["samples"])
            samples: List[Sample] = []
            for item in packed_samples:
                samples.append(Sample(
                    state_planes=np.array(item[0], dtype=np.float32),
                    policy=np.array(item[1], dtype=np.float32),
                    z=float(item[2]),
                ))
            state.replays[net_name].add_many(samples)
            total_samples[net_name] += len(samples)
            by_net[net_name] += len(res.get("results", []))
            if res.get("avg_gate"):
                gate_by_net[net_name].append(res["avg_gate"])
        train_stats = {}
        for name in ["A", "B"]:
            train_stats[name] = train_team(
                state.nets[name],
                state.replays[name],
                state.device,
                steps=train_steps,
                batch_size=batch_size,
                lr=lr,
            )
        eval_summary = evaluate_current_pair(state, sims=int(config["eval_sims"]), games=int(config["round_eval_games"]))
        score_a = eval_summary["wins_A"] / max(1, eval_summary["games"])
        state.leaderboard.update("A_current", "B_current", score_a)
        snap_a = save_checkpoint(run_dir, "A", state.nets["A"], round_idx)
        snap_b = save_checkpoint(run_dir, "B", state.nets["B"], round_idx)
        state.leaderboard.ensure(snap_a, state.leaderboard.ratings["A_current"])
        state.leaderboard.ensure(snap_b, state.leaderboard.ratings["B_current"])

        round_record = {
            "round": round_idx,
            "games_by_net": by_net,
            "samples_by_net": total_samples,
            "replay_sizes": {"A": state.replays['A'].size(), "B": state.replays['B'].size()},
            "train": train_stats,
            "eval": eval_summary,
            "selfplay_avg_gate": {
                "A": np.mean(np.array(gate_by_net["A"]), axis=0).tolist() if gate_by_net["A"] else [0.0] * len(EXPERT_NAMES),
                "B": np.mean(np.array(gate_by_net["B"]), axis=0).tolist() if gate_by_net["B"] else [0.0] * len(EXPERT_NAMES),
            },
            "worker_elapsed_sec": worker_times,
        }
        state.round_stats.append(round_record)
        safe_json_dump(run_dir / "rounds" / f"round_{round_idx:03d}.json", round_record)
        print(json.dumps(round_record, indent=2))

        if round_idx == mid_round:
            stage_reports["midstage"] = build_stage_report(
                run_dir,
                "midstage",
                round_idx,
                state,
                eval_summary,
                extra={"replay_sizes": {"A": state.replays['A'].size(), "B": state.replays['B'].size()}, "train": train_stats},
            )
        if round_idx == rounds:
            stage_reports["completed"] = build_stage_report(
                run_dir,
                "completed",
                round_idx,
                state,
                eval_summary,
                extra={"replay_sizes": {"A": state.replays['A'].size(), "B": state.replays['B'].size()}, "train": train_stats},
            )
    write_run_summary(run_dir, config, stage_reports)
    safe_json_dump(run_dir / "leaderboard.json", {"ratings": state.leaderboard.top()})
    print(f"Run complete. Summary at {run_dir / 'RUN_SUMMARY.md'}")
    server.shutdown()
    if local_worker_server is not None:
        local_worker_server.shutdown()


# -----------------------------
# Wizard / UX
# -----------------------------
def print_banner() -> None:
    print("=" * 76)
    print("EXOSFEAR MicroGo KG Distributed")
    print("6x6 self-play lab with two graph-team players, LAN workers, and leaderboard")
    print(f"Each team has specialist node-nets: {', '.join(EXPERT_NAMES)}")
    print("High risk: self-play + MCTS + training can become compute-intensive.")
    print("Review settings. Start small.")
    print("=" * 76)


def coordinator_wizard() -> Dict[str, Any]:
    print("\nCoordinator setup")
    token = prompt_default("Shared token (same on all machines)", gen_token())
    coord_host = prompt_default("Coordinator bind host", "0.0.0.0")
    coord_port = prompt_int("Coordinator port", free_port(DEFAULT_COORD_PORT), 1)
    run_dir = prompt_default("Run directory", "microgo_exosfear_kg_run")
    device = prompt_default("Device (cpu/cuda)", choose_device())
    rounds = prompt_int("Rounds", 6, 1)
    selfplay_jobs_per_round = prompt_int("Self-play jobs per round per net", 2, 1)
    games_per_job = prompt_int("Games per self-play job", 2, 1)
    selfplay_sims = prompt_int("MCTS simulations per self-play move", 20, 1)
    eval_sims = prompt_int("MCTS simulations per eval move", 28, 1)
    train_steps = prompt_int("Training gradient steps per round per net", 60, 1)
    batch_size = prompt_int("Batch size", 64, 8)
    lr = prompt_float("Learning rate", 0.001, 1e-6)
    stage_eval_games = prompt_int("Stage 0 eval games", 8, 2)
    round_eval_games = prompt_int("Per-round eval games", 8, 2)
    local_worker = prompt_yes_no("Also run a local worker on this coordinator machine?", True)
    local_worker_port = free_port(DEFAULT_WORKER_PORT) if local_worker else 0
    if local_worker:
        local_worker_port = prompt_int("Local worker port", local_worker_port, 1)
    scan_port = prompt_int("Scan for workers on port", DEFAULT_WORKER_PORT, 1)
    do_scan = prompt_yes_no("Scan the local /24 subnet for workers now?", True)
    discovered = []
    if do_scan:
        print("Scanning... this can take a bit.")
        discovered = scan_for_workers(scan_port)
        if discovered:
            print("Discovered workers:")
            for url in discovered:
                print(f"  - {url}")
        else:
            print("No workers auto-discovered.")
    manual = prompt_default("Manual worker URLs (comma-separated, blank if none)", "")
    worker_urls = [u.strip() for u in discovered]
    if manual.strip():
        worker_urls.extend([u.strip() for u in manual.split(",") if u.strip()])
    worker_urls = sorted(set(worker_urls))
    safe_json_dump(Path(run_dir) / "worker_targets.json", {"worker_urls": worker_urls})
    print("\nCoordinator quick notes:")
    print("- Start one or more workers on your Mac/Dell with the same token.")
    print(f"- Coordinator URL workers should use: http://<COORDINATOR_IP>:{coord_port}")
    print(f"- Expert nodes per team: {', '.join(EXPERT_NAMES)}")
    return {
        "token": token,
        "coord_host": coord_host,
        "coord_port": coord_port,
        "run_dir": run_dir,
        "device": device,
        "seed": SEED,
        "rounds": rounds,
        "selfplay_jobs_per_round": selfplay_jobs_per_round,
        "games_per_job": games_per_job,
        "selfplay_sims": selfplay_sims,
        "eval_sims": eval_sims,
        "train_steps_per_round": train_steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "stage_eval_games": stage_eval_games,
        "round_eval_games": round_eval_games,
        "local_worker": local_worker,
        "local_worker_port": local_worker_port,
        "worker_urls": worker_urls,
    }


def worker_wizard() -> Dict[str, Any]:
    print("\nWorker setup")
    ips = local_ipv4_addresses()
    print("Detected local IPv4 addresses:")
    for ip in ips:
        print(f"  - {ip}")
    host_default = next((ip for ip in ips if not ip.startswith("127.")), "0.0.0.0")
    host = prompt_default("Worker bind host", host_default)
    port = prompt_int("Worker port", free_port(DEFAULT_WORKER_PORT), 1)
    device = prompt_default("Device (cpu/cuda)", choose_device())
    token = prompt_default("Shared token (must match coordinator)", gen_token())
    coordinator_url = prompt_default("Coordinator URL", f"http://{host_default}:{DEFAULT_COORD_PORT}")
    return {"host": host, "port": port, "device": device, "token": token, "coordinator_url": coordinator_url}


def inspector(run_dir: Path) -> None:
    print(f"Inspecting {run_dir}")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8")) if (run_dir / "config.json").exists() else None
    if cfg:
        print(json.dumps(cfg, indent=2))
    for name in ["stage0", "midstage", "completed"]:
        p = run_dir / "reports" / f"{name}.json"
        if p.exists():
            print("-" * 76)
            print(name)
            print(p.read_text(encoding="utf-8"))
    summ = run_dir / "RUN_SUMMARY.md"
    if summ.exists():
        print("-" * 76)
        print(summ.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EXOSFEAR MicroGo KG Distributed")
    ap.add_argument("--mode", choices=["wizard", "coordinator", "worker", "inspect"], default="wizard")
    ap.add_argument("--run-dir", default="microgo_exosfear_kg_run")
    ap.add_argument("--coord-url", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main() -> None:
    print_banner()
    args = parse_args()
    if args.mode == "wizard":
        print("Choose mode:")
        print("  1) Coordinator")
        print("  2) Worker")
        print("  3) Inspect existing run")
        choice = prompt_default("Select", "1")
        if choice == "1":
            cfg = coordinator_wizard()
            coordinator_pipeline(cfg)
            return
        elif choice == "2":
            cfg = worker_wizard()
            server = start_worker_health_server(cfg["host"], int(cfg["port"]), cfg["token"], {
                "host": cfg["host"],
                "port": int(cfg["port"]),
                "device": cfg["device"],
                "role": "worker",
            })
            print(f"Worker health server on http://{cfg['host']}:{cfg['port']}")
            print(f"Use token: {cfg['token']}")
            worker_loop(cfg["coordinator_url"], cfg["token"], cfg["host"], int(cfg["port"]), cfg["device"])
            server.shutdown()
            return
        else:
            run_dir = Path(prompt_default("Run directory to inspect", args.run_dir))
            inspector(run_dir)
            return
    if args.mode == "coordinator":
        cfg = coordinator_wizard()
        coordinator_pipeline(cfg)
    elif args.mode == "worker":
        host = args.host or prompt_default("Worker bind host", next((ip for ip in local_ipv4_addresses() if not ip.startswith('127.')), '0.0.0.0'))
        port = args.port or prompt_int("Worker port", free_port(DEFAULT_WORKER_PORT), 1)
        device = args.device or prompt_default("Device (cpu/cuda)", choose_device())
        token = args.token or prompt_default("Shared token", gen_token())
        coord_url = args.coord_url or prompt_default("Coordinator URL", f"http://127.0.0.1:{DEFAULT_COORD_PORT}")
        server = start_worker_health_server(host, int(port), token, {"host": host, "port": int(port), "device": device, "role": "worker"})
        print(f"Worker health server on http://{host}:{port}")
        worker_loop(coord_url, token, host, int(port), device)
        server.shutdown()
    else:
        inspector(Path(args.run_dir))


if __name__ == "__main__":
    main()
