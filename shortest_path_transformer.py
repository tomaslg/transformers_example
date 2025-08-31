#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:15:27 2025

@author: tomaslg
"""

import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


# ----------------------------
# Graph sampling utilities
# ----------------------------

def uniform_int(a: int, b: int) -> int:
    return int(random.randint(a, b))

def sample_random_graph(
    n_max: int,
    directed: bool = False,
    w_low: float = 1.0,
    w_high: float = 10.0,
) -> Tuple[nx.Graph, int, int]:
    """
    Sample a connected random graph.

    - N ~ Uniform_discrete( floor(log(n_max)), n_max ), N >= 2
    - density D ~ Uniform(1, log(N)); target edges E ~= D * N
    - Build a random spanning chain (ensures connectivity), then add random edges up to E
    - Edge weights ~ Uniform[w_low, w_high]
    - Return G, s, t
    """
    n_min = max(2, int(math.floor(math.log(max(3, n_max)))))
    N = random.randint(n_min, max(n_min, n_max))

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(N))

    # --- random spanning tree via random permutation chain ---
    order = random.sample(range(N), N)  # a random permutation
    for i in range(N - 1):
        u, v = order[i], order[i + 1]
        G.add_edge(u, v)
        if not directed:
            # undirected graph needs only one edge; NetworkX handles symmetry
            pass
        else:
            # ensure weak connectivity both ways to simplify training
            G.add_edge(v, u)

    # --- target edge count from density ---
    D = random.uniform(1.0, max(1.0, math.log(N)))
    E_target = int(D * N)

    # clamp to max possible edges
    max_possible = N * (N - 1) if directed else N * (N - 1) // 2
    E_target = max(min(E_target, max_possible), G.number_of_edges())

    # --- add random extra edges until reaching E_target ---
    def try_add(u: int, v: int):
        if u == v:
            return
        if directed:
            if not G.has_edge(u, v):
                G.add_edge(u, v)
        else:
            # for undirected, avoid duplicate in either direction
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                G.add_edge(u, v)

    while G.number_of_edges() < E_target:
        u = random.randrange(N)
        v = random.randrange(N)
        try_add(u, v)

    # --- assign random positive weights ---
    for u, v in G.edges():
        G[u][v]["weight"] = random.uniform(w_low, w_high)

    # --- random distinct start/target ---
    s, t = random.sample(range(N), 2)
    return G, s, t



# ----------------------------
# Feature engineering
# ----------------------------

def graph_to_tensors(G: nx.Graph, s: int, t: int, device: torch.device):
    """
    Build:
    - node_feats : [N, F] -> [deg_norm, sum_w_norm, min_w_norm, is_start, is_target]
    - adj_w      : [N, N] weighted adjacency (0 if no edge)
    """
    N = G.number_of_nodes()
    node_feats = []
    sum_w_all = 0.0
    degs = []
    sums = []
    mins = []
    for i in range(N):
        nbrs = list(G.neighbors(i))
        deg = len(nbrs)
        degs.append(deg)
        if deg == 0:
            sums.append(0.0)
            mins.append(0.0)
        else:
            wlist = [G[i][j]["weight"] for j in nbrs]
            sums.append(sum(wlist))
            mins.append(min(wlist))
            sum_w_all += sum(wlist)
    # Normalizations to [0,1]-ish
    max_deg = max(1, max(degs))
    max_sum = max(1e-8, max(sums))
    max_min = max(1e-8, max(mins))
    for i in range(N):
        deg_norm = degs[i] / max_deg
        sum_norm = sums[i] / max_sum
        min_norm = mins[i] / max(1e-8, max_min)
        node_feats.append([deg_norm, sum_norm, min_norm, 1.0 if i == s else 0.0, 1.0 if i == t else 0.0])

    node_feats = torch.tensor(node_feats, dtype=torch.float32, device=device)

    adj_w = torch.zeros((N, N), dtype=torch.float32, device=device)
    for u, v, d in G.edges(data=True):
        w = float(d["weight"])
        adj_w[u, v] = w
        if not G.is_directed():
            adj_w[v, u] = w

    return node_feats, adj_w


# ----------------------------
# Simple GCN layer (edge-weighted)
# ----------------------------

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, X: torch.Tensor, A_w: torch.Tensor) -> torch.Tensor:
        """
        X: [N, F], A_w: [N, N] (edge weights or 0)
        Uses row-normalized weighted messages.
        """
        N = X.size(0)
        deg = A_w.sum(dim=1, keepdim=True)  # [N,1], weighted degree
        deg = torch.clamp(deg, min=1e-8)
        A_norm = A_w / deg  # row-normalized

        msg = A_norm @ X      # [N,F]
        h = self.W_self(X) + self.W_msg(msg)
        h = F.relu(h)
        h = self.bn(h)
        return self.dropout(h)


# ----------------------------
# Encoder: GCN + Transformer
# ----------------------------

class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int = 2, num_layers: int = 2, gcn_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gcn = nn.ModuleList()
        dims = [in_dim] + [d_model] * gcn_layers
        for i in range(gcn_layers):
            self.gcn.append(GCNLayer(dims[i], dims[i+1], dropout=dropout))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout, batch_first=False)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, node_feats: torch.Tensor, adj_w: torch.Tensor) -> torch.Tensor:
        """
        node_feats: [N, F], adj_w: [N, N]
        returns node_embeds: [N, d_model]
        """
        h = node_feats
        for layer in self.gcn:
            h = layer(h, adj_w)  # [N, d_model]

        # Transformer expects [seq_len, batch, d_model]; batch=1
        H = h.unsqueeze(1)  # [N,1,d_model]
        H = self.tf(H)      # [N,1,d_model]
        return H.squeeze(1) # [N,d_model]


# ----------------------------
# Pointer decoder (attention over nodes, masked to neighbors)
# ----------------------------

class PointerDecoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)
        # Optional small RNN for statefulness (feed current node embedding each step)
        self.rnn = nn.GRUCell(d_model, d_model)

    def step_scores(self, dec_h: torch.Tensor, node_embeds: torch.Tensor) -> torch.Tensor:
        """
        dec_h: [d_model]
        node_embeds: [N, d_model]
        returns scores: [N]
        """
        q = self.query(dec_h)                       # [d_model]
        K = self.key(node_embeds)                   # [N,d_model]
        scores = (K @ q) * self.scale               # [N]
        return scores

    def forward(self, node_embeds: torch.Tensor, adj_w: torch.Tensor, start_idx: int, target_idx: int,
                sample: bool = True, temperature: float = 1.0, max_steps: Optional[int] = None) -> Tuple[List[int], List[Tuple[int,int]], torch.Tensor]:
        """
        node_embeds: [N,d_model], adj_w: [N,N], start_idx, target_idx
        Returns: (node_path, edge_path, log_prob_sum)
        """
        device = node_embeds.device
        N = node_embeds.size(0)
        max_steps = max_steps or (2 * N)
        current = start_idx
        node_path = [current]
        edge_path: List[Tuple[int,int]] = []
        logprob_sum = torch.zeros((), device=device)

        # initial decoder hidden is start node embedding
        dec_h = node_embeds[current]
        # We'll update hidden with GRUCell(curr_embed, hidden)
        hidden = dec_h

        for step in range(max_steps):
            # Scores over all nodes; we’ll mask non-neighbors and the current node (no self loops)
            scores = self.step_scores(hidden, node_embeds)  # [N]
            # mask: neighbors are valid (adj_w[current] > 0)
            neighbor_mask = (adj_w[current] > 0).float()    # [N]
            # Optional: avoid staying put immediately
            neighbor_mask[current] = 0.0
            masked_scores = scores.clone()
            masked_scores[neighbor_mask == 0] = -1e9
            # If all invalid (isolated), stop
            if torch.all(masked_scores <= -1e8):
                break

            # Temperature softmax
            if temperature is None or temperature <= 0:
                probs = F.softmax(masked_scores, dim=0)
            else:
                probs = F.softmax(masked_scores / temperature, dim=0)

            if sample:
                next_node = int(torch.multinomial(probs, num_samples=1).item())
                logprob_sum = logprob_sum + torch.log(torch.clamp(probs[next_node], min=1e-9))
            else:
                next_node = int(torch.argmax(probs).item())
                # (greedy has no gradient contribution)

            edge_path.append((current, next_node))
            node_path.append(next_node)

            # Update RNN hidden
            hidden = self.rnn(node_embeds[next_node], hidden)

            current = next_node
            if current == target_idx:
                break

        return node_path, edge_path, logprob_sum


# ----------------------------
# Full model
# ----------------------------

@dataclass
class ModelConfig:
    in_dim: int = 5
    d_model: int = 128
    nhead: int = 4
    enc_layers: int = 2
    gcn_layers: int = 2
    dropout: float = 0.1
    lr: float = 3e-4

class SPPolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = GraphEncoder(
            in_dim=cfg.in_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.enc_layers,
            gcn_layers=cfg.gcn_layers,
            dropout=cfg.dropout,
        )
        self.decoder = PointerDecoder(cfg.d_model)

    def forward(self, node_feats: torch.Tensor, adj_w: torch.Tensor, s: int, t: int,
                sample: bool = True, temperature: float = 1.0, max_steps: Optional[int] = None):
        node_embeds = self.encoder(node_feats, adj_w)  # [N,d_model]
        return self.decoder(node_embeds, adj_w, s, t, sample=sample, temperature=temperature, max_steps=max_steps)


# ----------------------------
# Utilities: cost, dijkstra, save/load
# ----------------------------

def path_cost(G: nx.Graph, edges: List[Tuple[int, int]]) -> float:
    c = 0.0
    for u, v in edges:
        # If non-existent due to bug, add big penalty
        if not G.has_edge(u, v):
            c += 1e6
        else:
            c += float(G[u][v]["weight"])
    return c

def dijkstra_path_and_cost(G: nx.Graph, s: int, t: int) -> Tuple[List[Tuple[int,int]], float]:
    try:
        nodes = nx.shortest_path(G, s, t, weight="weight")
        edges = list(zip(nodes[:-1], nodes[-1:]))  # wrong; fix below
    except nx.NetworkXNoPath:
        return [], float("inf")
    # Build edge list properly
    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
    return edges, path_cost(G, edges)

def save_checkpoint(path: str, model: SPPolicy, cfg: ModelConfig):
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(payload, path)
    print(f"[info] saved checkpoint to {path}")

def load_checkpoint(path: str, device: torch.device) -> Tuple[SPPolicy, ModelConfig]:
    payload = torch.load(path, map_location=device)
    cfg_dict = payload.get("config", {})
    cfg = ModelConfig(**cfg_dict)
    model = SPPolicy(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print(f"[info] loaded checkpoint from {path}")
    return model, cfg


# ----------------------------
# Training loop (REINFORCE)
# ----------------------------

def train(
    n_max: int,
    epochs: int,
    save_path: str,
    device: torch.device,
    eval_every: int = 200,
    directed: bool = False,
    seed: Optional[int] = None,
    resume: bool = False,
):

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    cfg = ModelConfig()
    if resume:
        model, cfg = load_checkpoint(save_path, device)
    else:
        model = SPPolicy(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # moving average baseline on reward (negative cost)
    baseline = None
    beta = 0.9  # EMA decay
    temperature = 1.0
    entropy_coef = 1e-3

    for step in range(1, epochs + 1):
        model.train()

        # Sample fresh graph
        G, s, t = sample_random_graph(n_max=n_max, directed=directed)
        node_feats, adj_w = graph_to_tensors(G, s, t, device)

        # Rollout: sample policy
        node_path, edge_path, logprob_sum = model(node_feats, adj_w, s, t, sample=True, temperature=temperature)
        cost = path_cost(G, edge_path)
        reward = -cost

        # Greedy baseline path (no gradient) to stabilize learning
        with torch.no_grad():
            node_path_g, edge_path_g, _ = model(node_feats, adj_w, s, t, sample=False)
            cost_g = path_cost(G, edge_path_g)
            reward_g = -cost_g

        # Optional EMA baseline as a backup
        if baseline is None:
            baseline = reward_g
        else:
            baseline = beta * baseline + (1 - beta) * reward_g

        advantage = (reward - baseline)

        # Entropy for exploration (based on the last step distribution would be best; here we approximate)
        # We'll re-run a single greedy step distribution to estimate entropy (cheap & simple)
        # (Optional: you can modify the decoder to return per-step probs to do this exactly.)
        entropy_bonus = 0.0

        # loss = -(advantage.detach()) * logprob_sum - entropy_coef * entropy_bonus
        loss = -advantage * logprob_sum - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            print(f"[train] step {step:05d} | reward {reward:.3f} | baseline {baseline:.3f} | adv {advantage:.3f} | len {len(edge_path)}")

        if step % eval_every == 0:
            evaluate(model, n_max=n_max, device=device, directed=directed)

    save_checkpoint(save_path, model, cfg)
    return model


# ----------------------------
# Evaluation against Dijkstra
# ----------------------------

def evaluate(model: SPPolicy, n_max: int, device: torch.device, directed: bool = False, trials: int = 50):
    model.eval()
    gap_sum = 0.0
    exact = 0
    solved = 0
    for _ in range(trials):
        G, s, t = sample_random_graph(n_max=n_max, directed=directed)
        node_feats, adj_w = graph_to_tensors(G, s, t, device)
        with torch.no_grad():
            _, edge_path, _ = model(node_feats, adj_w, s, t, sample=False)
        cost_model = path_cost(G, edge_path)
        edges_opt, cost_opt = dijkstra_path_and_cost(G, s, t)
        if math.isfinite(cost_opt):
            solved += 1
            if cost_model <= cost_opt + 1e-6:
                exact += 1
            gap = (cost_model - cost_opt) / max(1e-8, cost_opt)
            gap_sum += max(0.0, gap)  # only positive gaps; negative means we matched or beat (rare due to float tol)
    if solved == 0:
        print("[eval] Dijkstra failed on all cases (disconnected?)")
        return
    print(f"[eval] trials={trials} | exact={exact}/{solved} ({100*exact/solved:.1f}%) | avg gap={100*gap_sum/solved:.2f}%")

# ----------------------------
# CLI & main
# ----------------------------

def main():
    # runfile('/home/tomaslg/Projects/LM/shortest_path_transformer.py', args='--n_max 60 --epochs 2000 --save_path sp_ptrnet.pt', wdir='/home/tomaslg/Projects/LM')
    # runfile('/home/tomaslg/Projects/LM/shortest_path_transformer.py', args='--n_max 60 --epochs 0 --save_path sp_ptrnet.pt --eval_only', wdir='/home/tomaslg/Projects/LM')

    parser = argparse.ArgumentParser(description="Transformer Pointer Policy for Shortest Path (unsupervised RL).")
    parser.add_argument("--n_max", type=int, default=50, help="Max number of nodes for random graphs.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training iterations (each samples a new graph).")
    parser.add_argument("--save_path", type=str, default="sp_ptrnet.pt", help="Where to save (and load) the checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if it exists.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and just evaluate the saved model.")
    parser.add_argument("--directed", action="store_true", help="Use directed graphs (default: undirected).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    if args.eval_only:
        model, _ = load_checkpoint(args.save_path, device)
        evaluate(model, n_max=args.n_max, device=device, directed=args.directed)
        return

    # Train (optionally resume)
    model = None
    if args.resume:
        try:
            model, _ = load_checkpoint(args.save_path, device)
        except Exception as e:
            print(f"[warn] resume failed: {e} — starting fresh.")

    if model is None:
        model = train(
            n_max=args.n_max,
            epochs=max(0, args.epochs),
            save_path=args.save_path,
            device=device,
            directed=args.directed,
            seed=args.seed,
            resume=False,
        )
    else:
        # continue training from loaded model
        print("[info] continuing training from checkpoint …")
        cfg = ModelConfig()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        # run remaining epochs from CLI
        for k in range(args.epochs):
            # quick wrapper around the same train logic would be cleaner; for brevity, reuse train() next time.
            G, s, t = sample_random_graph(n_max=args.n_max, directed=args.directed)
            node_feats, adj_w = graph_to_tensors(G, s, t, device)
            model.train()
            node_path, edge_path, logprob_sum = model(node_feats, adj_w, s, t, sample=True)
            cost = path_cost(G, edge_path)
            reward = -cost
            loss = -reward * logprob_sum
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if (k+1) % 50 == 0:
                print(f"[train-ctr] step {k+1:05d} | reward {reward:.3f} | len {len(edge_path)}")
        save_checkpoint(args.save_path, model, ModelConfig())

    # Final eval
    evaluate(model, n_max=args.n_max, device=device, directed=args.directed)


if __name__ == "__main__":
    main()
