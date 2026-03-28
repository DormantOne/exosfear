# exosfear.py — a self-growing knowledge graph of neural nets (FIXED)
# warning - i did not complete the run and it locked up my computer. so review and use with extreme caution if at all 
# pip install torch && python exosfear.py

import torch, torch.nn as nn, torch.nn.functional as F, os, random

# === STEP 1: eat every text file you have ===

def eat(dirs=[".", "~/Documents", "~/Desktop"], max_chars=2_000_000, max_files=500):
    t = ""
    files_read = 0

    SKIP = {".git", "node_modules", "__pycache__", "venv", ".venv",
            "env", ".env", "site-packages", ".tox", "dist", "build",
            ".cargo", ".rustup", "target", ".cache", "Library",
            "bin", "obj", ".vs", ".idea", "Pods", ".gradle"}

    for d in dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for root, subdirs, files in os.walk(d):
            # Prune heavy directories IN PLACE — stops os.walk from entering them
            subdirs[:] = [s for s in subdirs if s not in SKIP and not s.startswith('.')]

            for fname in files:
                if not fname.endswith(('.txt', '.py', '.md', '.csv', '.json', '.html')):
                    continue
                try:
                    path = os.path.join(root, fname)
                    # read() with a limit, not read()[:50000] which reads the whole file first
                    with open(path, errors="ignore") as f:
                        chunk = f.read(50_000)
                    t += chunk + "\n"
                    files_read += 1

                    if files_read % 50 == 0:
                        print(f"  📄 {files_read} files read, {len(t):,} chars so far...")

                    if files_read >= max_files or len(t) >= max_chars:
                        print(f"  ✅ Hit cap at {files_read} files, {len(t):,} chars.")
                        return t
                except:
                    pass

    print(f"  📄 Done: {files_read} files, {len(t):,} chars")
    return t or "the quick brown fox jumps over the lazy dog\n" * 5000


# === STEP 2: The Core Architecture ===

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        # FIX: compute LayerNorm once and reuse for Q, K, V
        nx = self.n1(x)
        x = x + self.attn(nx, nx, nx, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class Node(nn.Module):
    def __init__(self, V, d=64, h=4, L=2, ctx=64):
        super().__init__()
        self.te = nn.Embedding(V, d)
        self.pe = nn.Embedding(ctx, d)
        self.blocks = nn.Sequential(*[Block(d, h) for _ in range(L)])
        self.out = nn.Linear(d, V)
        self.ctx = ctx
        self.d = d

    def _add_pe(self, embs):
        """Single source of truth for positional encoding."""
        positions = torch.arange(embs.size(1), device=embs.device)
        return embs + self.pe(positions)

    def encode(self, x):
        """Token indices → deep latent features (no output head).
        Used by neighbors to produce 'thoughts' for message passing."""
        return self.blocks(self._add_pe(self.te(x)))

    def forward(self, x=None, embs=None):
        """Full forward: tokens or pre-built embeddings → logits."""
        if embs is None:
            embs = self.te(x)
        return self.out(self.blocks(self._add_pe(embs)))

    @torch.no_grad()
    def speak(self, seed, n=400, temp=0.8):
        x = seed
        for _ in range(n):
            logits = self(x[:, -self.ctx:])[:, -1, :]
            p = F.softmax(logits / temp, dim=-1)
            x = torch.cat([x, torch.multinomial(p, 1)], 1)
        return x


class Router(nn.Module):
    def __init__(self, num_nodes, d, num_channels=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Linear(128, num_nodes * num_channels)
        )

    def forward(self, query_emb):
        shape = list(query_emb.shape[:-1]) + [self.num_nodes, self.num_channels]
        scores = self.net(query_emb).view(*shape)
        # Softmax over nodes: which node is best to ask, per channel?
        return F.softmax(scores, dim=-2)


class KnowledgeGraph(nn.Module):
    def __init__(self, num_nodes=4, V=0, d_base=64, ctx_base=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.V = V
        self.d_base = d_base
        self.nodes = nn.ModuleList()
        self.routers = nn.ModuleList()

        for i in range(num_nodes):
            ctx = ctx_base + i * 16
            L = 2 + (i // 2)
            self.nodes.append(Node(V, d=d_base, h=4, L=L, ctx=ctx))
            self.routers.append(Router(num_nodes, d=d_base))

        # FIX: Projection layers to bridge representation spaces.
        # Neighbor thoughts are deep features (post-LN, post-attn, post-MLP).
        # Lead node embeddings are raw. Can't add them directly.
        self.msg_projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_base), nn.Linear(d_base, d_base))
            for _ in range(num_nodes)
        ])

        # FIX: Learnable gates initialized near zero.
        # sigmoid(-2.0) ≈ 0.12 — message starts weak, model must learn to open it.
        # Prevents random router weights from destabilizing early training.
        self.msg_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.0)) for _ in range(num_nodes)
        ])

    def _min_ctx(self):
        return min(n.ctx for n in self.nodes)

    def collaborative_forward(self, x, lead_idx):
        """
        Differentiable graph message passing for training.
        Gradients flow through router edges AND through neighbor blocks.
        """
        lead = self.nodes[lead_idx]
        num_channels = self.routers[lead_idx].num_channels

        # FIX: Crop to lead's context window (prevents PE table overflow)
        x_lead = x[:, -lead.ctx:]
        seq_len = x_lead.size(1)
        embs = lead.te(x_lead)

        # Router computes soft edge weights
        route = self.routers[lead_idx](embs)  # (B, S, Nodes, Channels)

        msg = torch.zeros_like(embs)

        for j in range(self.num_nodes):
            if j == lead_idx:
                continue

            neighbor = self.nodes[j]
            # FIX: Each neighbor gets its own context crop
            x_j = x[:, -neighbor.ctx:]
            j_thought = neighbor.encode(x_j)

            # Align sequence lengths (neighbors may have different ctx)
            if j_thought.size(1) >= seq_len:
                j_thought = j_thought[:, -seq_len:, :]
            else:
                pad_len = seq_len - j_thought.size(1)
                pad = torch.zeros(
                    j_thought.size(0), pad_len, j_thought.size(2),
                    device=x.device
                )
                j_thought = torch.cat([pad, j_thought], dim=1)

            # Sum channel weights → total attention to this neighbor
            weight = route[:, :, j, :].sum(dim=-1, keepdim=True)  # (B, S, 1)
            msg = msg + weight * j_thought

        # FIX: Normalize message magnitude (was scaling with num_channels)
        msg = msg / num_channels

        # FIX: Project into lead's embedding space and gate
        msg = self.msg_projs[lead_idx](msg)
        gate = torch.sigmoid(self.msg_gates[lead_idx])
        embs = embs + gate * msg

        # Lead node processes enriched embeddings
        return lead(embs=embs)

    @torch.no_grad()
    def collaborative_generate(self, prompt_tokens, hops=6):
        """Inference: the graph thinks out loud, one token per hop."""
        device = next(self.parameters()).device
        x = prompt_tokens.unsqueeze(0).to(device)
        trace = []

        for step in range(hops):
            lead_idx = step % self.num_nodes
            lead = self.nodes[lead_idx]
            num_channels = self.routers[lead_idx].num_channels
            x_crop = x[:, -lead.ctx:]

            embs = lead.te(x_crop)
            query_emb = embs[:, -1:, :]  # (1, 1, d)
            route = self.routers[lead_idx](query_emb)  # (1, 1, Nodes, Channels)

            msg = torch.zeros_like(query_emb)
            for j in range(self.num_nodes):
                if j == lead_idx:
                    continue

                total_weight = route[0, 0, j, :].sum().item()
                if total_weight > 0.5:
                    neighbor = self.nodes[j]
                    j_thought = neighbor.encode(x[:, -neighbor.ctx:])[:, -1:, :]
                    msg = msg + total_weight * j_thought
                    trace.append(f"N{lead_idx}←N{j}({total_weight:.2f})")

            # Same normalization + projection + gating as training path
            msg = msg / num_channels
            msg = self.msg_projs[lead_idx](msg)
            gate = torch.sigmoid(self.msg_gates[lead_idx])
            embs[:, -1:, :] = embs[:, -1:, :] + gate * msg

            logits = lead(embs=embs)[:, -1, :]
            next_tok = torch.multinomial(F.softmax(logits / 0.75, dim=-1), 1)
            x = torch.cat([x, next_tok], dim=1)

        return x[0], " → ".join(trace[-8:]) or "independent thought"


# === STEP 3: Training Functions ===

def learn_local(node, data, steps=150, bs=32):
    """Individual node trains on its own data partition."""
    opt = torch.optim.AdamW(node.parameters(), lr=3e-4)
    C = node.ctx
    if len(data) < C + 2:
        print("    ⚠️  Not enough data for this node's context window, skipping.")
        return 999.0

    for i in range(steps):
        ix = torch.randint(len(data) - C - 1, (bs,))
        xb = torch.stack([data[j:j + C] for j in ix])
        yb = torch.stack([data[j + 1:j + C + 1] for j in ix])
        loss = F.cross_entropy(
            node(xb).reshape(-1, node.out.out_features), yb.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(node.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if i % max(steps // 3, 1) == 0:
            print(f"    [Local] step {i:3d}/{steps}  loss {loss.item():.3f}")

    return loss.item()


def learn_global(graph, data, steps=100, bs=32):
    """
    Whole graph trains end-to-end. Loss backpropagates through router edges
    into neighbor transformer blocks, training who-talks-to-whom.
    """
    opt = torch.optim.AdamW(graph.parameters(), lr=3e-4)
    # FIX: Use the minimum context across ALL nodes
    C = graph._min_ctx()
    if len(data) < C + 2:
        print("    ⚠️  Not enough data for global training, skipping.")
        return 999.0

    for i in range(steps):
        ix = torch.randint(len(data) - C - 1, (bs,))
        xb = torch.stack([data[j:j + C] for j in ix])
        yb = torch.stack([data[j + 1:j + C + 1] for j in ix])

        lead_idx = random.randint(0, graph.num_nodes - 1)

        logits = graph.collaborative_forward(xb, lead_idx)
        loss = F.cross_entropy(logits.reshape(-1, graph.V), yb.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(graph.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        if i % max(steps // 3, 1) == 0:
            gate_val = torch.sigmoid(graph.msg_gates[lead_idx]).item()
            print(f"  [Global] step {i:3d}/{steps}  loss {loss.item():.3f}  "
                  f"gate[N{lead_idx}]={gate_val:.3f}")

    return loss.item()


# === STEP 4: THE GRAPH GROWTH LOOP ===

if __name__ == "__main__":
    print("🔍 Eating your files to seed the graph...")
    raw = eat()
    chars = sorted(set(raw))
    V = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    enc = lambda s: torch.tensor([c2i[c] for c in s if c in c2i])
    dec = lambda t: "".join(chars[i] for i in t)

    data = enc(raw)
    print(f"📚 {len(data):,} chars | Vocab {V}\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(dev)

    graph = KnowledgeGraph(num_nodes=4, V=V, d_base=64, ctx_base=64).to(dev)

    # Each node gets a clone so they can diverge and specialize
    node_datas = [data.clone() for _ in range(graph.num_nodes)]

    total_params = sum(p.numel() for p in graph.parameters())
    print(f"🧠 Graph: {graph.num_nodes} nodes, {total_params:,} total parameters")
    for i, node in enumerate(graph.nodes):
        np_ = sum(p.numel() for p in node.parameters())
        print(f"   Node {i}: ctx={node.ctx}, L={len(node.blocks)}, params={np_:,}")
    print()

    for g in range(8):
        print(f"\n{'='*60}")
        print(f"🧬 GEN {g} — GRAPH EVOLUTION")
        print(f"{'='*60}")

        # --- PHASE 1: LOCAL TRAINING ---
        print("\n--- Phase 1: Local Node Training ---")
        for i, node in enumerate(graph.nodes):
            print(f"\n  🧠 Node {i} (ctx={node.ctx}, L={len(node.blocks)}, "
                  f"data={len(node_datas[i]):,} tokens)")
            loss = learn_local(node, node_datas[i], steps=120 + g * 50)

            # Self-feeding: node generates text and eats it
            with torch.no_grad():
                seed = node_datas[i][-node.ctx:].unsqueeze(0)
                taste = dec(node.speak(seed, n=300)[0].tolist())
                bonus = enc(taste).to(dev)
                # FIX: Tighter quality gate — loss 3.4 on char-level is still noise
                if loss < 2.5 and len(bonus) > 100:
                    node_datas[i] = torch.cat([node_datas[i], bonus[-600:]])
                    print(f"    📈 +{min(600, len(bonus))} synthetic tokens "
                          f"(loss={loss:.2f})")
                else:
                    print(f"    ⏸️  No self-feed (loss={loss:.2f}, need <2.5)")

        # --- PHASE 2: GLOBAL GRAPH LEARNING ---
        print("\n--- Phase 2: Global Graph Training ---")
        mixed = torch.cat([random.choice(node_datas)[-2000:] for _ in range(4)])
        print(f"  📊 Mixed dataset: {len(mixed):,} tokens")
        learn_global(graph, mixed, steps=80 + g * 20)

        # --- PHASE 3: COLLABORATIVE INFERENCE ---
        print("\n--- Phase 3: Collaborative Inference ---")
        lines = [l for l in raw.splitlines() if len(l.strip()) > 10]
        prompt = random.choice(lines[:2000])[:80] if lines else "explain the universe"
        prompt_tokens = enc(prompt).to(dev)

        if len(prompt_tokens) < 2:
            prompt = "the quick brown fox"
            prompt_tokens = enc(prompt).to(dev)

        result, trace = graph.collaborative_generate(prompt_tokens, hops=12)
        output = dec(result.tolist())

        print(f"\n  💬 PROMPT: \"{prompt[:60]}...\"")
        print(f"  📝 OUTPUT: \"{output[-250:]}\"")
        print(f"  🔗 TRACE:  {trace}")

        # --- PHASE 4: FEED COLLECTIVE OUTPUT BACK ---
        # FIX: Original used `for nd in node_datas: nd = torch.cat(...)`
        # which rebinds a local variable — list is never mutated.
        # Every generation's collaborative output was silently thrown away.
        collab_bonus = enc(output).to(dev)
        if len(collab_bonus) > 50:
            for i in range(len(node_datas)):
                node_datas[i] = torch.cat([node_datas[i], collab_bonus[-600:]])
            print(f"\n  🔄 Fed +{min(600, len(collab_bonus))} collaborative tokens "
                  f"back to all nodes")

        # --- PHASE 5: CHECKPOINT ---
        torch.save({
            "gen": g,
            "graph_state": graph.state_dict(),
            "V": V,
            "chars": chars
        }, f"graph_gen{g}.pt")

        # Print diagnostics
        gates = [torch.sigmoid(graph.msg_gates[i]).item() for i in range(graph.num_nodes)]
        data_sizes = [len(nd) for nd in node_datas]
        print(f"\n  💾 Saved graph_gen{g}.pt")
        print(f"  🚪 Gates: {['%.3f' % gv for gv in gates]}")
        print(f"  📦 Data:  {['%dk' % (s // 1000) for s in data_sizes]}")

    print(f"\n{'='*60}")
    print("✅ Done. Your living knowledge graph has learned its own topology.")
    print(f"{'='*60}")
