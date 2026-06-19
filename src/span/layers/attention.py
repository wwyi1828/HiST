import torch
import torch.nn as nn
import dgl
from torch.nn import functional as F

def generate_swin_windows(matrix, w, use_shift=False):
    h, width = matrix.size(0), matrix.size(1)

    height_padding = (2 * w - h % (2 * w)) % (2 * w)
    width_padding = (2 * w - width % (2 * w)) % (2 * w)

    if height_padding > 0 or width_padding > 0:
        if use_shift:
            total_height_padding = w + height_padding
            total_width_padding = w + width_padding
            matrix = F.pad(matrix, (w, total_width_padding, w, total_height_padding), 'constant', 0)
        else:
            matrix = F.pad(matrix, (0, width_padding, 0, height_padding), 'constant', 0)

    blocks = matrix.unfold(0, 2 * w, 2 * w).unfold(1, 2 * w, 2 * w)

    block_sums = blocks.sum(dim=[-2, -1])
    return blocks[block_sums > 0]

ATTENTION_CONFIGS = {
    'hybrid': [
        {'graph': 'g1', 'patterns': ['swmsa', 'wmsa', 'lloop'], 'dedup': True},
        {'graph': 'g2', 'patterns': ['g2g', 'l2g', 'g2l', 'gloop'], 'require_global': True},
    ],
    'hybrid_noshift': [
        {'graph': 'g1', 'patterns': ['wmsa', 'lloop'], 'dedup': True},
        {'graph': 'g2', 'patterns': ['g2g', 'l2g', 'g2l', 'gloop'], 'require_global': True},
    ],
    'long': [
        {'graph': 'g1', 'patterns': ['swmsa', 'wmsa', 'lloop'], 'dedup': True},
        {'graph': 'g2', 'patterns': ['g2g', 'l2g', 'g2l', 'gloop'], 'require_global': True},
    ],
    'swin': [
        {'graph': 'g1', 'patterns': ['swmsa', 'lloop']},
        {'graph': 'g2', 'patterns': ['wmsa', 'lloop']},
    ],
    'tradswin': [
        {'graph': 'g1', 'patterns': ['swmsa', 'lloop']},
        {'graph': 'g2', 'patterns': ['wmsa', 'lloop']},
    ],
    'swin_global': [
        {'graph': 'g1', 'patterns': ['swmsa', 'lloop']},
        {'graph': 'g2', 'patterns': ['wmsa', 'lloop']},
        {'graph': 'g2', 'patterns': ['g2g', 'l2g', 'g2l', 'gloop'], 'require_global': True},
    ],
    'lsa': [
        {'graph': 'g1', 'patterns': ['lsa', 'lloop'], 'dedup': True},
        {'graph': 'g2', 'patterns': ['g2g', 'l2g', 'g2l', 'gloop'], 'require_global': True},
    ],
}

class AttentionBuilder(nn.Module):
    def __init__(self, window_size=1, mode='long', mask_padding=False):
        super(AttentionBuilder, self).__init__()
        self.window_size = window_size

        self.mode = mode
        self.mask_padding = mask_padding

    def edges_global2local(self):
        num_original = self.positions.size(0)
        num_globals = self.num_valid_globals
        num_nodes = self.positions.size(0) + self.num_valid_globals

        valid_indices = torch.arange(0, num_original, device=self.positions.device)
        valid_indices = valid_indices[1:-1]
        src_nodes = torch.arange(num_original, num_nodes, device=self.positions.device).repeat_interleave(len(valid_indices))
        dst_nodes = valid_indices.repeat(num_globals)

        return (src_nodes, dst_nodes)

    def edges_local_self_loop(self):
        num_original = self.positions.size(0)
        indices = torch.arange(0, num_original, device=self.positions.device)
        indices = indices[1:-1]
        src_nodes = indices
        dst_nodes = indices

        return (src_nodes, dst_nodes)

    def edges_global_self_loop(self):
        num_globals = self.num_valid_globals
        start_index = self.positions.size(0)
        global_indices = torch.arange(start_index, start_index + num_globals, device=self.positions.device)

        src_nodes = global_indices
        dst_nodes = src_nodes
        return (src_nodes, dst_nodes)

    def edges_local2global(self):
        num_original = self.positions.size(0)
        num_globals = self.num_valid_globals
        num_nodes = self.positions.size(0) + self.num_valid_globals

        valid_indices = torch.arange(0, num_original, device=self.positions.device)
        valid_indices = valid_indices[1:-1]
        src_nodes = valid_indices.repeat_interleave(num_globals)
        dst_nodes = torch.arange(num_original, num_nodes, device=self.positions.device).repeat(len(valid_indices))

        return (src_nodes, dst_nodes)

    def edges_global2global(self):
        num_globals = self.num_valid_globals
        start_global_index = self.positions.size(0)
        num_nodes = self.positions.size(0) + self.num_valid_globals

        window_indices = torch.arange(start_global_index, num_nodes, device=self.positions.device)
        src = window_indices.unsqueeze(1).repeat(1, num_globals).view(-1)
        dst = window_indices.repeat(num_globals)
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        return (src, dst)

    def build_graph(self, mode):

        valid_modes = ['lloop', 'gloop', 'g2g', 'l2g', 'g2l', 'wmsa', 'swmsa', 'lsa']
        assert mode in valid_modes, f"Invalid mode {mode}. Valid modes are: {', '.join(valid_modes)}"

        if mode == 'lloop':
            return self.edges_local_self_loop()
        if mode == 'gloop':
            return self.edges_global_self_loop()
        if mode == 'g2g':
            return self.edges_global2global()
        if mode == 'l2g':
            return self.edges_local2global()
        if mode == 'g2l':
            return self.edges_global2local()
        elif mode == 'lsa':
            return self._build_lsa_attention()
        elif mode in ('wmsa', 'swmsa'):
            num_nodes = self.positions.size(0)
            device = self.positions.device

            if self.mask_padding:
                num_local = self.positions.size(0)
                valid_indices = torch.arange(0, num_local, device=device)
                valid_indices = valid_indices[1:-1]
                node_shift_id = valid_indices + 1
                node_pos = self.positions[valid_indices]
            else:
                node_shift_id = torch.arange(1, self.positions.size(0) + 1, device=device)
                node_pos = self.positions

            adjusted_positions = node_pos
            adjusted_positions = adjusted_positions.short().t()

            chunk_size = (adjusted_positions[0].max() + 1, adjusted_positions[1].max() + 1)
            sparse_tensor = torch.sparse_coo_tensor(adjusted_positions, node_shift_id, size=chunk_size, device=device)
            dense_matrix = sparse_tensor.to_dense()

            if dense_matrix.size(0) < 2 * self.window_size and dense_matrix.size(1) < 2 * self.window_size:
                num_elements = dense_matrix.numel()
                indices = torch.arange(num_elements, device=device)
                src = indices.repeat_interleave(num_elements)
                dst = indices.repeat(num_elements)
                dense_matrix_flat = dense_matrix.flatten()
                global_src = dense_matrix_flat[src]
                global_dst = dense_matrix_flat[dst]
            else:
                use_shift = (mode == 'swmsa')
                edge_matrices = generate_swin_windows(dense_matrix, w=self.window_size, use_shift=use_shift)
                window_size_squared = (2 * self.window_size) ** 2
                window_indices = torch.arange(window_size_squared, device=device)
                src = window_indices.unsqueeze(1).repeat(1, window_size_squared).view(-1)
                dst = window_indices.repeat(window_size_squared)

                edge_matrices_flat = edge_matrices.view(-1, window_size_squared)
                global_src = edge_matrices_flat[:, src].flatten()
                global_dst = edge_matrices_flat[:, dst].flatten()

            mask = (global_src != 0) & (global_dst != 0) & (global_src != global_dst)
            filtered_src = global_src[mask]
            filtered_dst = global_dst[mask]

            return (filtered_src - 1, filtered_dst - 1)

    def _build_lsa_attention(self):
        num_nodes = self.positions.size(0)
        device = self.positions.device
        w = self.window_size

        if self.mask_padding:
            num_local = self.positions.size(0)
            valid_indices = torch.arange(0, num_local, device=device)[1:-1]
            node_shift_id = valid_indices + 1
            node_pos = self.positions[valid_indices]
        else:
            node_shift_id = torch.arange(1, self.positions.size(0) + 1, device=device)
            node_pos = self.positions

        if node_pos.numel() == 0:
            return (torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device))

        adjusted_positions = node_pos.short().t()
        H = int(adjusted_positions[0].max().item()) + 1
        W = int(adjusted_positions[1].max().item()) + 1

        dense_matrix = torch.sparse_coo_tensor(
            adjusted_positions, node_shift_id, size=(H, W), device=device
        ).to_dense()

        if dense_matrix.numel() == 0:
            return (torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device))

        pad = (w, w, w, w)
        padded = F.pad(dense_matrix, pad, mode='constant', value=0)
        K = 2 * w + 1

        windows = padded.unfold(0, K, 1).unfold(1, K, 1)
        H_out, W_out = windows.size(0), windows.size(1)
        if H_out == 0 or W_out == 0:
            return (torch.tensor([], dtype=torch.long, device=device),
                    torch.tensor([], dtype=torch.long, device=device))

        windows_flat = windows.contiguous().view(H_out * W_out, K * K)
        centers = dense_matrix.view(-1)

        src = centers.repeat_interleave(K * K)
        dst = windows_flat.flatten()

        mask = (src != 0) & (dst != 0) & (src != dst)
        src = src[mask] - 1
        dst = dst[mask] - 1

        return (src, dst)

    def forward(self, positions, valid_globals,  mode='hybrid'):

        self.positions = positions
        self.num_valid_globals = valid_globals

        num_nodes = self.positions.size(0) + self.num_valid_globals
        device = self.positions.device
        g1 = dgl.graph(([], []), num_nodes=num_nodes, device=device)
        g2 = dgl.graph(([], []), num_nodes=num_nodes, device=device)

        config = ATTENTION_CONFIGS[mode]

        for head in config:
            if head.get('require_global', False) and valid_globals == 0:
                continue

            src, dst = self._collect_edges(head['patterns'])

            if head.get('dedup', False):
                edge_pairs = torch.stack([src, dst], dim=0)
                edge_pairs = torch.unique(edge_pairs, dim=1)
                src, dst = edge_pairs[0], edge_pairs[1]

            target = g1 if head['graph'] == 'g1' else g2
            target.add_edges(src, dst)

        return g1, g2

    def _collect_edges(self, modes):
        edges = {mode: self.build_graph(mode=mode) for mode in modes}

        valid_modes = [mode for mode in modes if edges[mode][0].numel() > 0 and edges[mode][1].numel() > 0]
        src = torch.cat([edges[mode][0] for mode in valid_modes], dim=0)
        dst = torch.cat([edges[mode][1] for mode in valid_modes], dim=0)
        return src, dst
