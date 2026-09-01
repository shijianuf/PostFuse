"""Graph construction for pseudo-dialogue GNN baselines."""
from __future__ import annotations

import torch


def sequential_edges(n: int) -> torch.Tensor:
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.arange(n - 1, dtype=torch.long)
    dst = src + 1
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def dialoguegcn_edges(n: int) -> torch.Tensor:
    """Past / future / sequential edges (single-speaker pseudo-dialogue)."""
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j < i:
                edges.append((j, i))
            elif j > i:
                edges.append((j, i))
            if abs(i - j) == 1:
                edges.append((i, j))
                edges.append((j, i))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = zip(*edges)
    return torch.tensor([src, dst], dtype=torch.long)


def dag_edges(n: int) -> torch.Tensor:
    """Directed acyclic: only past -> current."""
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = [], []
    for i in range(1, n):
        for j in range(i):
            src.append(j)
            dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


def normalize_adj(edge_index: torch.Tensor, n: int, device: torch.device | None = None) -> torch.Tensor:
    dev = device or edge_index.device
    if edge_index.numel() == 0:
        return torch.eye(n, device=dev)
    adj = torch.zeros(n, n, device=dev)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj + torch.eye(n, device=dev)
    deg = adj.sum(dim=1).clamp(min=1.0)
    return adj / deg.unsqueeze(1)
