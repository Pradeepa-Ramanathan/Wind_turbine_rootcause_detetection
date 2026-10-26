import math
import torch
from torch import nn


class ImplicitGraphLearner(nn.Module):
    """Learns a sensor-to-sensor adjacency using trainable node embeddings."""
    def __init__(self, num_nodes, emb_dim=16):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)

    def forward(self, tau=0.5, hard=False):
        theta = torch.sigmoid(self.embedding @ self.embedding.t()).clamp(1e-4, 1 - 1e-4)
        # Soft learned adjacency is deterministic; this keeps dashboard inference
        # reproducible while gradients still train the node embeddings.
        adj = theta
        adj = adj * (1 - torch.eye(adj.shape[0], device=adj.device))
        if hard:
            hard_adj = (adj > 0.5).float()
            adj = hard_adj + adj - adj.detach()
        return adj


class GraphFeatureTransform(nn.Module):
    """First-order Chebyshev-style graph propagation."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adjacency):
        degree = adjacency.sum(-1).clamp_min(1e-6)
        d_inv = torch.diag(degree.pow(-0.5))
        normalized = d_inv @ adjacency @ d_inv
        identity = torch.eye(adjacency.shape[0], device=adjacency.device)
        return self.lin((identity + normalized) @ x)


class KnowledgeFusion(nn.Module):
    """Implicit graph plus attention-weighted explicit knowledge graphs."""
    def __init__(self, num_nodes, explicit_graphs, hidden=32, emb_dim=16):
        super().__init__()
        self.implicit = ImplicitGraphLearner(num_nodes, emb_dim)
        self.base_transform = GraphFeatureTransform(1, hidden)
        self.graph_transforms = nn.ModuleList(
            [GraphFeatureTransform(1, hidden) for _ in explicit_graphs]
        )
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.num_graphs = len(explicit_graphs)
        for i, graph in enumerate(explicit_graphs):
            self.register_buffer(f"explicit_graph_{i}", torch.as_tensor(graph, dtype=torch.float32))

    def graphs(self):
        return [getattr(self, f"explicit_graph_{i}") for i in range(self.num_graphs)]

    def forward(self, node_features, tau=0.5, hard=False, adjacency=None):
        implicit = self.implicit(tau=tau, hard=hard)
        embeddings = [self.base_transform(node_features, implicit).mean(0, keepdim=True)]
        graph_list = self.graphs()
        if not graph_list and adjacency is not None:
            graph_list = [adjacency]
        for i, graph in enumerate(graph_list):
            transform = self.graph_transforms[i] if i < len(self.graph_transforms) else self.base_transform
            embeddings.append(transform(node_features, graph).mean(0, keepdim=True))
        if len(embeddings) > 1:
            stacked = torch.cat(embeddings, dim=0)
            q, k = self.query(stacked), self.key(stacked)
            attention = torch.softmax(q @ k.t() / math.sqrt(stacked.shape[-1]), dim=-1)
            alpha = attention[0, 1:]
        else:
            alpha = implicit.new_empty((0,))
        fused = implicit.clone()
        for i, graph in enumerate(graph_list):
            fused = fused + alpha[i] * graph
        fused = fused / fused.max().clamp_min(1e-6)
        return fused, alpha, implicit


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        kernels = (2, 3, 6, 7)
        cout_k = max(1, cout // len(kernels))
        self.convs = nn.ModuleList([
            nn.Conv2d(cin, cout_k, kernel_size=(1, k), dilation=(1, dilation))
            for k in kernels
        ])

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        min_len = min(item.shape[-1] for item in outputs)
        return torch.cat([item[..., -min_len:] for item in outputs], dim=1)


class TemporalConvModule(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.filter_conv = DilatedInception(channels, channels, dilation)
        self.gate_conv = DilatedInception(channels, channels, dilation)

    def forward(self, x):
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))


class GraphConvModule(nn.Module):
    def __init__(self, channels, depth=2, beta=0.6):
        super().__init__()
        self.depth = depth
        self.beta = beta
        self.mix = nn.Conv2d(channels * (depth + 1), channels, kernel_size=1)

    def forward(self, x, adjacency):
        degree = adjacency.sum(-1).unsqueeze(-1).clamp_min(1.0)
        normalized = adjacency / degree
        initial = x
        current = x
        states = [initial]
        for _ in range(self.depth):
            current = self.beta * initial + (1 - self.beta) * torch.einsum("nm,bcmt->bcnt", normalized, current)
            states.append(current)
        return self.mix(torch.cat(states, dim=1))


class MTGNN(nn.Module):
    """MTGNN-style dilated temporal inception and mix-hop graph convolution."""
    def __init__(self, num_nodes, seq_len, channels=32, layers=2, gc_depth=2, beta=0.6):
        super().__init__()
        self.start = nn.Conv2d(1, channels, kernel_size=1)
        self.temporal_layers = nn.ModuleList()
        self.graph_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        current_len = seq_len
        for layer in range(layers):
            dilation = 2 ** layer
            receptive = dilation * (7 - 1)
            current_len -= receptive
            if current_len <= 0:
                raise ValueError(f"window={seq_len} is too short for {layers} MTGNN layers.")
            self.temporal_layers.append(TemporalConvModule(channels, dilation))
            self.graph_layers.append(GraphConvModule(channels, gc_depth, beta))
            self.skip_layers.append(nn.Conv2d(channels, channels, kernel_size=(1, current_len)))
        self.end1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.end2 = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x, adjacency):
        x = self.start(x)
        skip = 0.0
        for temporal, graph, skip_layer in zip(self.temporal_layers, self.graph_layers, self.skip_layers):
            residual = x
            x = temporal(x)
            residual = residual[..., -x.shape[-1]:]
            x = graph(x, adjacency) + residual
            skip = skip + skip_layer(x)
        x = torch.relu(skip)
        x = torch.relu(self.end1(x))
        return self.end2(x).squeeze(-1).squeeze(1)


class GraphTemporalForecaster(nn.Module):
    """UI-compatible wrapper around KnowledgeFusion + MTGNN.

    Input shape: (batch, time, sensors). Output shape: (batch, sensors).
    """
    def __init__(self, n_nodes, hidden=32, dropout=0.15, explicit_graphs=None, window=24,
                 layers=2, gc_depth=2, beta=0.6, emb_dim=16):
        super().__init__()
        explicit_graphs = explicit_graphs or []
        self.n_nodes = n_nodes
        self.window = window
        self.fusion = KnowledgeFusion(n_nodes, explicit_graphs, hidden, emb_dim)
        self.forecaster = MTGNN(n_nodes, window, hidden, layers, gc_depth, beta)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency=None):
        # x is (B,T,N); MTGNN expects (B,1,N,T)
        node_features = x.mean(dim=(0, 1)).unsqueeze(-1)
        fused, _, _ = self.fusion(node_features, adjacency=adjacency)
        model_input = self.dropout(x.transpose(1, 2).unsqueeze(1))
        return self.forecaster(model_input, fused)


def make_windows(x, window=24):
    import numpy as np
    if len(x) <= window:
        return np.empty((0, window, x.shape[1]), np.float32), np.empty((0, x.shape[1]), np.float32)
    X = np.stack([x[i:i + window] for i in range(len(x) - window)])
    y = np.stack([x[i + window] for i in range(len(x) - window)])
    return X.astype(np.float32), y.astype(np.float32)
